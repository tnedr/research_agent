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


class KeywordAgent(ResearchAgent):
    """Generates research keywords from the topic"""

    def __init__(self, llm: Any):
        super().__init__(llm)
        self.system_prompt = """You are a research keyword specialist. 
        Generate relevant academic keywords for the given research topic.
        Consider both general and specific aspects of the topic.
        Return keywords in order of relevance."""

    def process(self, state: ResearchState) -> ResearchState:
        prompt = f"""Generate research keywords for the topic: {state.topic}
        Consider:
        1. Main concepts
        2. Related terms
        3. Specific aspects
        4. Methodology terms

        Return as a JSON list of strings."""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        keywords = json.loads(response.content)

        state.keywords = keywords
        state.current_step = "data_collection"
        state.needs_human_feedback = True
        state.feedback_prompt = f"Please review these keywords: {keywords}\nAre they appropriate?"

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
        """Get human feedback"""
        print("\nHuman feedback required:")
        print(self.state.feedback_prompt)

        # Simple command-line interface for feedback
        response = input("Enter 'approve' or 'reject': ").strip().lower()

        if response == 'approve':
            self.state.needs_human_feedback = False
        else:
            # Handle rejection by returning to previous step
            step_mapping = {
                "data_collection": "keyword_generation",
                "ranking": "data_collection",
                "download": "ranking",
                "analysis": "download",
                "summary": "analysis",
                "complete": "summary"
            }
            self.state.current_step = step_mapping.get(self.state.current_step,
                                                       self.state.current_step)
            self.state.needs_human_feedback = False

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
        "default": ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
        ),
        "keyword": ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
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
    print("Articles analyzed:", len(results["articles"]))
    print("\nFinal Summary:", results["summary"])


if __name__ == "__main__":
    main()