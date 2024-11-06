from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from langgraph.graph import Graph, MessageGraph

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field
import json
import requests
from abc import ABC, abstractmethod
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

os.environ["LANGCHAIN_TRACING"] = "false"
load_dotenv()


# Data Models
class Article(BaseModel):
    title: str
    abstract: Optional[str]
    url: str
    citations: Optional[int]
    publication_date: Optional[datetime]
    source: str


class ResearchState(BaseModel):
    topic: str
    keywords: List[str] = Field(default_factory=list)
    collected_articles: List[Article] = Field(default_factory=list)
    ranked_articles: List[Article] = Field(default_factory=list)
    downloaded_articles: List[Article] = Field(default_factory=list)
    analysis_results: List[Dict] = Field(default_factory=list)
    final_summary: Optional[str] = None
    current_step: str = "keyword_generation"
    needs_human_feedback: bool = False
    feedback_prompt: Optional[str] = None


# Base Agent Class
class ResearchAgent(ABC):
    def __init__(self, llm: Any, tools: Optional[List[Tool]] = None):
        self.llm = llm
        self.tools = tools or []
        self.state = {}

    @abstractmethod
    def process(self, state: ResearchState) -> ResearchState:
        """Process the current state and return updated state"""
        pass

    def requires_human_feedback(self, state: ResearchState) -> bool:
        """Determine if human feedback is needed"""
        return False

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from langgraph.graph import Graph, MessageGraph

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field
import json
import requests
from abc import ABC, abstractmethod
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

os.environ["LANGCHAIN_TRACING"] = "false"
load_dotenv()


# Data Models
class Article(BaseModel):
    title: str
    abstract: Optional[str]
    url: str
    citations: Optional[int]
    publication_date: Optional[datetime]
    source: str


class ResearchState(BaseModel):
    topic: str
    keywords: List[str] = Field(default_factory=list)
    collected_articles: List[Article] = Field(default_factory=list)
    ranked_articles: List[Article] = Field(default_factory=list)
    downloaded_articles: List[Article] = Field(default_factory=list)
    analysis_results: List[Dict] = Field(default_factory=list)
    final_summary: Optional[str] = None
    current_step: str = "keyword_generation"
    needs_human_feedback: bool = False
    feedback_prompt: Optional[str] = None


# Base Agent Class
class ResearchAgent(ABC):
    def __init__(self, llm: Any, tools: Optional[List[Tool]] = None):
        self.llm = llm
        self.tools = tools or []
        self.state = {}

    @abstractmethod
    def process(self, state: ResearchState) -> ResearchState:
        """Process the current state and return updated state"""
        pass

    def requires_human_feedback(self, state: ResearchState) -> bool:
        """Determine if human feedback is needed"""
        return False

    @abstractmethod
    def get_conversation_config(self) -> Dict:
        """Return agent-specific conversation configuration"""
        pass


class KeywordAgent(ResearchAgent):
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.tools = [{
            "type": "function",
            "function": {
                "name": "generate_scholar_queries",
                "description": "Generate Google Scholar search queries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of search queries where each query is a space-separated combination of terms"
                        }
                    },
                    "required": ["queries"]
                }
            }
        }]

    def get_conversation_config(self) -> Dict:
        return {
            "system_prompt": "You are a Google Scholar search specialist...",
            "tools": self.tools,
            "state_key": "keywords",
            "tool_response_key": "queries",
            "display_format": "{}",
            "feedback_template": """Current queries: {current_state}
User feedback: "{feedback}"

Instructions:
- If the user approves (saying ok, good, yes, etc), return the exact same queries to confirm
- If the user wants changes, return the modified queries
- ALWAYS return queries using the generate_scholar_queries tool
- Never respond without using the tool

Analyze the feedback and respond accordingly with the tool."""
        }

    def process(self, state: ResearchState) -> ResearchState:
        prompt = f"""Generate academic search queries for: {state.topic}

        Create effective queries that will find highly-cited papers.
        Each query should combine 3-5 terms with methodology terms.

        Example format:
        - "eggs cholesterol heart disease meta"
        - "eggs cardiovascular health systematic review"

        Include:
        - Methodological terms (meta-analysis, review, trial)
        - Specific health outcomes
        - Key concepts and variables"""

        messages = [
            SystemMessage(content=self.get_conversation_config()["system_prompt"]),
            HumanMessage(content=prompt)
        ]

        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(messages)

        tool_call = response.tool_calls[0]
        search_queries = tool_call["args"]["queries"] # q miert hihvjak queriesnek
        state.keywords = search_queries
        state.current_step = "keyword"
        state.needs_human_feedback = True
        state.feedback_prompt = "\nHere are the suggested search queries:\n" + \
                                "\n".join(f"- {q}" for q in search_queries) + \
                                "\n\nWhat do you think about these queries? Feel free to suggest any changes."

        return state


class DataCollectorAgent(ResearchAgent):
    """Collects articles based on keywords"""

    def __init__(self, llm: Any):
        super().__init__(llm)
        self.apis = {
            "pubmed": self._search_pubmed,
            "arxiv": self._search_arxiv
        }

    def process(self, state: ResearchState) -> ResearchState:
        all_articles = []
        for database, search_func in self.apis.items():
            articles = search_func(state.keywords)
            all_articles.extend(articles)

        state.collected_articles = all_articles
        state.current_step = "ranking"
        return state

    def _search_pubmed(self, keywords: List[str]) -> List[Article]:
        # Simulate PubMed API call
        return [Article(
            title=f"Research on {keyword}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{i}",
            citations=10,
            publication_date=datetime.now(),
            source="PubMed"
        ) for i, keyword in enumerate(keywords)]

    def _search_arxiv(self, keywords: List[str]) -> List[Article]:
        # Simulate arXiv API call
        return [Article(
            title=f"Study of {keyword}",
            url=f"https://arxiv.org/abs/{i}",
            citations=5,
            publication_date=datetime.now(),
            source="arXiv"
        ) for i, keyword in enumerate(keywords)]


class RankingAgent(ResearchAgent):
    """Ranks articles by relevance"""

    def process(self, state: ResearchState) -> ResearchState:
        # Rank based on citations and date
        ranked_articles = sorted(
            state.collected_articles,
            key=lambda x: (x.citations or 0, x.publication_date or datetime.min),
            reverse=True
        )

        state.ranked_articles = ranked_articles
        state.current_step = "download"
        state.needs_human_feedback = True
        state.feedback_prompt = f"Review top 5 ranked articles:\n" + \
                                "\n".join([a.title for a in ranked_articles[:5]])

        return state


class DownloadAgent(ResearchAgent):
    """Downloads article content"""

    def process(self, state: ResearchState) -> ResearchState:
        downloaded_articles = []
        for article in state.ranked_articles:
            # Simulate download
            article.abstract = f"Abstract for {article.title}"
            downloaded_articles.append(article)

        state.downloaded_articles = downloaded_articles
        state.current_step = "analysis"
        return state


class AnalysisAgent(ResearchAgent):
    """Analyzes article content"""

    def __init__(self, llm: Any):
        super().__init__(llm)
        self.system_prompt = """You are a research analyst.
        Analyze academic articles and extract key findings related to the research topic."""

    def process(self, state: ResearchState) -> ResearchState:
        analysis_results = []
        for article in state.downloaded_articles:
            prompt = f"""Analyze this article's relevance to {state.topic}:
            Title: {article.title}
            Abstract: {article.abstract}

            Extract key findings and their relevance."""

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            analysis_results.append({
                "article": article.title,
                "analysis": response.content
            })

        state.analysis_results = analysis_results
        state.current_step = "summary"
        state.needs_human_feedback = True
        state.feedback_prompt = "Please review the analysis results. Are they satisfactory?"

        return state


class SummaryAgent(ResearchAgent):
    """Generates final summary"""

    def __init__(self, llm: Any):
        super().__init__(llm)
        self.system_prompt = """You are a research summarizer.
        Create comprehensive literature reviews from analyzed research findings."""

    def process(self, state: ResearchState) -> ResearchState:
        prompt = f"""Create a literature review for topic: {state.topic}

        Analysis results:
        {json.dumps(state.analysis_results, indent=2)}

        Provide a comprehensive overview including:
        1. Main findings
        2. Consensus points
        3. Contradictions
        4. Research gaps"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        state.final_summary = response.content
        state.current_step = "complete"
        state.needs_human_feedback = True
        state.feedback_prompt = "Please review the final literature review. Is it satisfactory?"

        return state


class ResearchWorkflow:
    """Manages the multi-agent research workflow"""

    def __init__(self, llm_configs: Dict[str, Any]):
        self.agents = {
            "keyword": KeywordAgent(llm_configs.get("keyword", llm_configs.get("default"))),
            # "collector": DataCollectorAgent(llm_configs.get("collector", llm_configs.get("default"))),
            # "ranking": RankingAgent(llm_configs.get("ranking", llm_configs.get("default"))),
            # "download": DownloadAgent(llm_configs.get("download", llm_configs.get("default"))),
            # "analysis": AnalysisAgent(llm_configs.get("analysis", llm_configs.get("default"))),
            # "summary": SummaryAgent(llm_configs.get("summary", llm_configs.get("default")))
        }

        self.workflow = self._build_workflow()
        self.state = None

    def _build_workflow(self) -> MessageGraph:
        """Build the workflow graph"""
        workflow = Graph()

        # Add nodes for each agent
        workflow.add_node("keyword", self.agents["keyword"].process)
        # workflow.add_node("collector", self.agents["collector"].process)
        # workflow.add_node("ranking", self.agents["ranking"].process)
        # workflow.add_node("download", self.agents["download"].process)
        # workflow.add_node("analysis", self.agents["analysis"].process)
        # workflow.add_node("summary", self.agents["summary"].process)

        # Define edges
        workflow.add_edge(START, "keyword")  # start node-ból indulunk
        # workflow.add_edge("keyword", "collector")
        # workflow.add_edge("collector", "ranking")
        # workflow.add_edge("ranking", "download")
        # workflow.add_edge("download", "analysis")
        # workflow.add_edge("analysis", "summary")
        workflow.add_edge("keyword", END)  # és end node-ba érkezünk
        return workflow.compile()

    def _get_human_feedback(self) -> None:
        """Natural conversation-based human feedback"""
        current_agent = self.agents[self.state.current_step]
        config = current_agent.get_conversation_config()

        print(self.state.feedback_prompt)

        while True:
            feedback = input("\nYour feedback: ").strip()

            # Get current state and prepare for conversation
            current_state = getattr(self.state, config["state_key"])
            conversation_prompt = config["feedback_template"].format(
                feedback=feedback,
                current_state=current_state
            )

            messages = [
                SystemMessage(content=config["system_prompt"]),
                HumanMessage(content=conversation_prompt)
            ]

            llm_with_tools = current_agent.llm.bind_tools(config["tools"])
            response = llm_with_tools.invoke(messages)

            tool_call = response.tool_calls[0]
            proposed_results = tool_call["args"]["queries"]

            if isinstance(proposed_results, list) and proposed_results:
                if str(proposed_results[0]).startswith("REJECTED:"):
                    print(
                        "\nI understand you want changes. Please provide more specific feedback about what you'd like to modify.")
                    continue

                if proposed_results == current_state:
                    print("\nI understand you're happy with these results. Let's continue.")
                    self.state.needs_human_feedback = False
                    self.state.current_step = "complete"
                    break
                else:
                    print("\nBased on your feedback, I've modified the queries:")
                    for i, result in enumerate(proposed_results, 1):
                        print(f"{i}. {result}")

                    setattr(self.state, config["state_key"], proposed_results)

    def run(self, topic: str) -> Dict[str, Any]:
        """Run the research workflow"""
        try:
            print(f"Starting research on: {topic}")

            # Initialize state
            self.state = ResearchState(topic=topic)

            while self.state.current_step != "complete":
                # Process current step
                self.state = self.workflow.invoke(self.state)

                # Handle human feedback if needed
                if self.state.needs_human_feedback:
                    self._get_human_feedback()

                # Ha a current_step complete, akkor kilépünk
                if self.state.current_step == "complete":
                    break

            return {
                "topic": self.state.topic,
                "keywords": self.state.keywords,
                # "articles": [a.dict() for a in self.state.ranked_articles],
                # "summary": self.state.final_summary
            }

        except Exception as e:
            print(f"Error in workflow: {str(e)}")
            raise


def main():
    # Configure LLMs for different agents
    llm_configs = {
        "default": ChatOpenAI(model="gpt-4o", temperature=0,
                                max_tokens=None, timeout=None, max_retries=2,
                                # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
                                # base_url="...",
                                # organization="...",
                                # other params...
        ),
        "keyword": ChatOpenAI(model="gpt-4o", temperature=0,
                                max_tokens=None, timeout=None, max_retries=2,
                                # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
                                # base_url="...",
                                # organization="...",
                                # other params...
        ),
        "keyword2": ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0,
                                    max_tokens=1024, timeout=None, max_retries=2,
                                # other params...
        ),
        "analysis": ChatOpenAI(temperature=0.3),
        "summary": ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    # other params...
    ),
    }

    # Initialize workflow
    workflow = ResearchWorkflow(llm_configs)

    # Run research
    results = workflow.run("Health effects of eggs on cardiovascular health")

    # Print results
    print("\nResearch Results:")
    print("Keywords:", results["keywords"])
    # print("Articles analyzed:", len(results["articles"]))
    # print("\nFinal Summary:", results["summary"])


if __name__ == "__main__":
    main()



class KeywordAgent(ResearchAgent):
    def __init__(self, llm: Any):
        super().__init__(llm)

        self.tools = [{
            "type": "function",
            "function": {
                "name": "generate_scholar_queries",
                "description": "Generate Google Scholar search queries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of search queries where each query is a space-separated combination of terms"
                        }
                    },
                    "required": ["queries"]
                }
            }
        }]

        self.system_prompt = """You are a Google Scholar search specialist.
        Generate effective search queries that will find the most relevant and highly-cited papers.
        Each query should be 3-5 keywords separated by spaces."""

    def process(self, state: ResearchState) -> ResearchState:
        prompt = f"""Generate Google Scholar search queries for the topic: {state.topic}

        Create a list of search queries where each query combines 3-5 relevant terms with spaces.

        Format example:
        - "eggs cholesterol heart disease meta"
        - "eggs cardiovascular health systematic review"

        Important:
        - Include methodological terms (meta-analysis, review, trial) for finding high-quality papers
        - Focus on combinations that will find highly-cited papers
        - Keep queries concise (3-5 terms) for best results
        - Combine specific health outcomes with methodology terms"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(messages)

        tool_call = response.tool_calls[0]
        search_queries = tool_call["args"]["queries"]

        state.keywords = search_queries
        state.current_step = "data_collection"
        state.needs_human_feedback = True
        state.feedback_prompt = "Please review these Google Scholar search queries:\n" + \
                                "\n".join(f"- {q}" for q in search_queries) + \
                                "\n\nAre these queries appropriate for finding relevant, highly-cited papers?"

        return state


class DataCollectorAgent(ResearchAgent):
    """Collects articles based on keywords"""

    def __init__(self, llm: Any):
        super().__init__(llm)
        self.apis = {
            "pubmed": self._search_pubmed,
            "arxiv": self._search_arxiv
        }

    def process(self, state: ResearchState) -> ResearchState:
        all_articles = []
        for database, search_func in self.apis.items():
            articles = search_func(state.keywords)
            all_articles.extend(articles)

        state.collected_articles = all_articles
        state.current_step = "ranking"
        return state

    def _search_pubmed(self, keywords: List[str]) -> List[Article]:
        # Simulate PubMed API call
        return [Article(
            title=f"Research on {keyword}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{i}",
            citations=10,
            publication_date=datetime.now(),
            source="PubMed"
        ) for i, keyword in enumerate(keywords)]

    def _search_arxiv(self, keywords: List[str]) -> List[Article]:
        # Simulate arXiv API call
        return [Article(
            title=f"Study of {keyword}",
            url=f"https://arxiv.org/abs/{i}",
            citations=5,
            publication_date=datetime.now(),
            source="arXiv"
        ) for i, keyword in enumerate(keywords)]


class RankingAgent(ResearchAgent):
    """Ranks articles by relevance"""

    def process(self, state: ResearchState) -> ResearchState:
        # Rank based on citations and date
        ranked_articles = sorted(
            state.collected_articles,
            key=lambda x: (x.citations or 0, x.publication_date or datetime.min),
            reverse=True
        )

        state.ranked_articles = ranked_articles
        state.current_step = "download"
        state.needs_human_feedback = True
        state.feedback_prompt = f"Review top 5 ranked articles:\n" + \
                                "\n".join([a.title for a in ranked_articles[:5]])

        return state


class DownloadAgent(ResearchAgent):
    """Downloads article content"""

    def process(self, state: ResearchState) -> ResearchState:
        downloaded_articles = []
        for article in state.ranked_articles:
            # Simulate download
            article.abstract = f"Abstract for {article.title}"
            downloaded_articles.append(article)

        state.downloaded_articles = downloaded_articles
        state.current_step = "analysis"
        return state


class AnalysisAgent(ResearchAgent):
    """Analyzes article content"""

    def __init__(self, llm: Any):
        super().__init__(llm)
        self.system_prompt = """You are a research analyst.
        Analyze academic articles and extract key findings related to the research topic."""

    def process(self, state: ResearchState) -> ResearchState:
        analysis_results = []
        for article in state.downloaded_articles:
            prompt = f"""Analyze this article's relevance to {state.topic}:
            Title: {article.title}
            Abstract: {article.abstract}

            Extract key findings and their relevance."""

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            analysis_results.append({
                "article": article.title,
                "analysis": response.content
            })

        state.analysis_results = analysis_results
        state.current_step = "summary"
        state.needs_human_feedback = True
        state.feedback_prompt = "Please review the analysis results. Are they satisfactory?"

        return state


class SummaryAgent(ResearchAgent):
    """Generates final summary"""

    def __init__(self, llm: Any):
        super().__init__(llm)
        self.system_prompt = """You are a research summarizer.
        Create comprehensive literature reviews from analyzed research findings."""

    def process(self, state: ResearchState) -> ResearchState:
        prompt = f"""Create a literature review for topic: {state.topic}

        Analysis results:
        {json.dumps(state.analysis_results, indent=2)}

        Provide a comprehensive overview including:
        1. Main findings
        2. Consensus points
        3. Contradictions
        4. Research gaps"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        state.final_summary = response.content
        state.current_step = "complete"
        state.needs_human_feedback = True
        state.feedback_prompt = "Please review the final literature review. Is it satisfactory?"

        return state


class ResearchWorkflow:
    """Manages the multi-agent research workflow"""

    def __init__(self, llm_configs: Dict[str, Any]):
        self.agents = {
            "keyword": KeywordAgent(llm_configs.get("keyword", llm_configs.get("default"))),
            # "collector": DataCollectorAgent(llm_configs.get("collector", llm_configs.get("default"))),
            # "ranking": RankingAgent(llm_configs.get("ranking", llm_configs.get("default"))),
            # "download": DownloadAgent(llm_configs.get("download", llm_configs.get("default"))),
            # "analysis": AnalysisAgent(llm_configs.get("analysis", llm_configs.get("default"))),
            # "summary": SummaryAgent(llm_configs.get("summary", llm_configs.get("default")))
        }

        self.workflow = self._build_workflow()
        self.state = None

    def _build_workflow(self) -> MessageGraph:
        """Build the workflow graph"""
        workflow = Graph()

        # Add nodes for each agent
        workflow.add_node("keyword", self.agents["keyword"].process)
        # workflow.add_node("collector", self.agents["collector"].process)
        # workflow.add_node("ranking", self.agents["ranking"].process)
        # workflow.add_node("download", self.agents["download"].process)
        # workflow.add_node("analysis", self.agents["analysis"].process)
        # workflow.add_node("summary", self.agents["summary"].process)

        # Define edges
        workflow.add_edge(START, "keyword")  # start node-ból indulunk
        # workflow.add_edge("keyword", "collector")
        # workflow.add_edge("collector", "ranking")
        # workflow.add_edge("ranking", "download")
        # workflow.add_edge("download", "analysis")
        # workflow.add_edge("analysis", "summary")
        workflow.add_edge("keyword", END)  # és end node-ba érkezünk
        return workflow.compile()

    def _get_human_feedback(self) -> None:
        """Natural conversation-based human feedback"""
        # Get the current agent and its config
        current_agent = self.agents[self.state.current_step]
        config = current_agent.get_conversation_config()

        print("\nHuman feedback required:")
        print(self.state.feedback_prompt)

        while True:
            feedback = input("\nPlease provide your feedback or instructions: ").strip()

            # Get current state from config
            current_state = getattr(self.state, config["state_key"])

            # Use agent's feedback template
            feedback_interpretation_prompt = config["feedback_template"].format(
                feedback=feedback,
                current_state=current_state
            )

            messages = [
                SystemMessage(content=config["system_prompt"]),
                AIMessage(content="Current state:"),
                HumanMessage(content=str(current_state)),
                HumanMessage(content=feedback_interpretation_prompt)
            ]

            llm_with_tools = current_agent.llm.bind_tools(config["tools"])
            response = llm_with_tools.invoke(messages)

            tool_call = response.tool_calls[0]
            proposed_result = tool_call["args"][config.get("tool_response_key", "queries")]

            if isinstance(proposed_result, list) and proposed_result:
                if str(proposed_result[0]).startswith("REJECTED:"):
                    print("\nI understand you're not satisfied. Please provide more specific feedback.")
                    continue

                if proposed_result == current_state:
                    print("\nI understand you're satisfied with these results.")
                    self.state.needs_human_feedback = False
                    self.state.current_step = "complete"
                    break
                else:
                    print("\nBased on your feedback, I've updated the results:")
                    display_format = config.get("display_format", "{}")
                    for i, result in enumerate(proposed_result, 1):
                        print(f"{i}. {display_format.format(result)}")

                    setattr(self.state, config["state_key"], proposed_result)

    def run(self, topic: str) -> Dict[str, Any]:
        """Run the research workflow"""
        try:
            print(f"Starting research on: {topic}")

            # Initialize state
            self.state = ResearchState(topic=topic)

            while self.state.current_step != "complete":
                # Process current step
                self.state = self.workflow.invoke(self.state)

                # Handle human feedback if needed
                if self.state.needs_human_feedback:
                    self._get_human_feedback()

                # Ha a current_step complete, akkor kilépünk
                if self.state.current_step == "complete":
                    break

            return {
                "topic": self.state.topic,
                "keywords": self.state.keywords,
                # "articles": [a.dict() for a in self.state.ranked_articles],
                # "summary": self.state.final_summary
            }

        except Exception as e:
            print(f"Error in workflow: {str(e)}")
            raise


def main():
    # Configure LLMs for different agents
    llm_configs = {
        "default": ChatOpenAI(model="gpt-4o", temperature=0,
                                max_tokens=None, timeout=None, max_retries=2,
                                # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
                                # base_url="...",
                                # organization="...",
                                # other params...
        ),
        "keyword": ChatOpenAI(model="gpt-4o", temperature=0,
                                max_tokens=None, timeout=None, max_retries=2,
                                # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
                                # base_url="...",
                                # organization="...",
                                # other params...
        ),
        "keyword2": ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0,
                                    max_tokens=1024, timeout=None, max_retries=2,
                                # other params...
        ),
        "analysis": ChatOpenAI(temperature=0.3),
        "summary": ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    # other params...
    ),
    }

    # Initialize workflow
    workflow = ResearchWorkflow(llm_configs)

    # Run research
    results = workflow.run("Health effects of eggs on cardiovascular health")

    # Print results
    print("\nResearch Results:")
    print("Keywords:", results["keywords"])
    # print("Articles analyzed:", len(results["articles"]))
    # print("\nFinal Summary:", results["summary"])
