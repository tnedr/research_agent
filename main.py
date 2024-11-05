from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic  # Az Anthropic Claude modell importálása
from langchain.llms import OpenAI  # OpenAI modellek
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from autogen import ConversableAgent
from dotenv import load_dotenv
import os
import pprint
import sys

from autogen.agentchat.utils import gather_usage_summary
load_dotenv()



# Állítsd be az OpenAI és Anthropic API kulcsokat környezeti változókkal vagy közvetlenül itt:
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"

class KeywordAgent:
    def __init__(self, model_type="openai", system_prompt="Generate relevant keywords for text."):
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model_type == "anthropic":
            return ChatAnthropic(model="claude-3-haiku-20240307", system_prompt=self.system_prompt)
        else:
            raise ValueError("Invalid model_type. Use 'openai' or 'anthropic'.")

    def generate_keywords(self, text: str):
        prompt = f"{self.system_prompt} Input: '{text}'"
        response = self.model(prompt)

        # Kulcsszavak formázása
        if hasattr(response, 'text'):
            keywords_text = response.text.strip()
        else:
            keywords_text = response.strip()

        keywords = [keyword.strip() for keyword in keywords_text.split(",")]
        return keywords


# Gráf állapotának meghatározása
class KeywordState(TypedDict):
    text: str
    keywords: List[str]
    confirmed: bool


# Gráf felépítése
graph_builder = StateGraph(KeywordState)


# 1. Csomópont: Kulcsszavak generálása
def keyword_generation(state: KeywordState):
    agent = KeywordAgent(model_type="anthropic")  # vagy 'openai'
    generated_keywords = agent.generate_keywords(state["text"])
    return {"keywords": generated_keywords, "confirmed": False}


graph_builder.add_node("generate_keywords", keyword_generation)


# 2. Csomópont: Emberi ellenőrzés
def human_verification(state: KeywordState):
    print("Please review the keywords:", state["keywords"])
    confirmation = input("Are these keywords acceptable? (y/n): ")
    return {"confirmed": confirmation.lower() == 'y'}


graph_builder.add_node("verify_keywords", human_verification)

# Gráf kapcsolatok
graph_builder.add_edge("generate_keywords", "verify_keywords")
graph_builder.add_conditional_edges("verify_keywords", lambda state: END if state["confirmed"] else "generate_keywords")
graph_builder.add_edge(START, "generate_keywords")

# Gráf futtatása
graph = graph_builder.compile()

if __name__ == "__main__":
    initial_state = KeywordState(text="Your topic or text here", keywords=[], confirmed=False)
    graph.run(initial_state)
