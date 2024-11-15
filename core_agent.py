from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

StateType = TypeVar('StateType', bound=BaseModel)
ResponseType = TypeVar('ResponseType', bound=BaseModel)

@dataclass
class ToolDefinition:
    """Structured definition of a tool"""
    name: str
    description: str
    schema: Dict
    function: Optional[callable] = None

class ToolProvider:
    """Base class for tool providers"""
    def prepare_tool_call(self, tool: ToolDefinition) -> Dict:
        raise NotImplementedError
    
    def bind_tools(self, llm: Any, tools: List[ToolDefinition]) -> Any:
        raise NotImplementedError



class AnthropicToolProvider(ToolProvider):
    def prepare_tool_call(self, tool: ToolDefinition) -> Dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.schema
        }
    
    def bind_tools(self, llm: Any, tools: List[ToolDefinition]) -> Any:
        # Convert ToolDefinition objects to dict format that Anthropic expects
        tool_dicts = [self.prepare_tool_call(tool) for tool in tools]
        return llm.bind(tools=tool_dicts)
    

class OpenAIToolProvider(ToolProvider):
    def prepare_tool_call(self, tool: ToolDefinition) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema
            }
        }
    
    def bind_tools(self, llm: Any, tools: List[ToolDefinition]) -> Any:
        # Convert ToolDefinition objects to dict format that OpenAI expects
        tool_dicts = [self.prepare_tool_call(tool) for tool in tools]
        return llm.bind_tools(tool_dicts)
    

class AgentState(BaseModel):
    """Base state model that all agent states should inherit from"""
    conversation_history: List[Dict] = Field(default_factory=list)
    tool_outputs: List[Dict] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)
    last_response: Optional[Dict] = Field(default=None)

class CoreAgent(Generic[StateType, ResponseType]):
    """Core agent implementation with interactive capabilities"""
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Dict],
        system_prompt: str,
        tool_provider: Optional["ToolProvider"] = None
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        
        # Validate tool_provider type first
        if tool_provider is not None and not isinstance(tool_provider, ToolProvider):
            raise ValueError(f"Invalid tool provider type: {type(tool_provider)}. Must be instance of ToolProvider")
        
        # Get tool provider
        self.tool_provider = tool_provider or self._get_default_tool_provider()
        # Bind tools at initialization
        self.bound_llm = self.tool_provider.bind_tools(self.llm, tools) if tools else self.llm
        self.logger = logging.getLogger(__name__)
        # Initialize state
        self.state = AgentState()
        # Add system prompt to initial state
        self.state.conversation_history.append({
            "role": "system",
            "content": self.system_prompt
        })

    def _extract_response(self, response: Any) -> Dict:
        """Extract structured response from LLM output"""
        if hasattr(response, 'tool_calls') and response.tool_calls==[]:
            return {
                'tool_used': False,
                'tool_name': None,
                'message': response.content
            }
        else:
            return {
                'tool_used': True,
                'tool_name': response.tool_calls[0].get('name'),
                'tool_args': response.tool_calls[0].get('args', {})
            }


    def process_message(self, message: str) -> Dict:
        """Process a user message and return raw response"""
        self.state.conversation_history.append({
            "role": "user",
            "content": message
        })

        try:
            llm_response = self.bound_llm.invoke(self.state.conversation_history)
            self.logger.info(f"LLM Response: {llm_response}")

            response_data = self._extract_response(llm_response)
            
            # Mentsük el az utolsó választ
            self.state.last_response = response_data
            
            if response_data['tool_used']:
                self.state.tool_outputs.append(response_data)
                return {"type": "tool_response", "content": response_data}
            else:
                return {"type": "conversation", "content": response_data['message']}

        except Exception as e:
            error_msg = f"LLM API Error: {str(e)}"
            self.state.errors.append(error_msg)
            return {"type": "error", "content": error_msg}

    def conversate(self, message: str) -> str:
        """High-level method for conversation that includes processing and formatting"""
        response = self.process_message(message)
        formatted_response = self._format_response(response)
        
        # Save formatted response to conversation history
        self.state.conversation_history.append({
            "role": "assistant",
            "content": formatted_response
        })
        
        return formatted_response

    def get_state(self) -> AgentState:
        """Get current state"""
        # Return a deep copy of the state
        return AgentState(
            conversation_history=self.state.conversation_history.copy(),
            tool_outputs=self.state.tool_outputs.copy(),
            errors=self.state.errors.copy(),
            metadata=self.state.metadata.copy()
        )

    def _get_default_tool_provider(self) -> "ToolProvider":
        """Get default tool provider based on LLM type"""
        if isinstance(self.llm, ChatAnthropic):
            return AnthropicToolProvider()
        elif isinstance(self.llm, ChatOpenAI):
            return OpenAIToolProvider()
        else:
            raise ValueError(f"No default tool provider for LLM type: {type(self.llm)}")

    def _format_response(self, response_data: Dict) -> str:
        """Format the response data into a human-readable string"""
        if response_data['type'] == 'conversation':
            return response_data['content']
        elif response_data['type'] == 'tool_response':
            tool_args = response_data['content']['tool_args']
            # Általános tool output formázás
            output_lines = [f"Here's the {response_data['content']['tool_name']} result:"]
            for key, value in tool_args.items():
                output_lines.append(f"{key}: {value}")
            return "\n".join(output_lines)
        else:
            return f"Sorry, there was an error: {response_data['content']}"

    def get_last_response(self) -> Optional[Dict]:
        """Az utolsó strukturált válasz lekérése"""
        return self.state.last_response

    @staticmethod
    def print_user_message(message: str):
        """Prints the user message in green."""
        print(f"\033[92mUser: {message}\033[0m")  # Green text

    @staticmethod
    def print_agent_message(message: str):
        """Prints the agent message in blue."""
        print(f"\033[94mAssistant: {message}\033[0m")  # Blue text



