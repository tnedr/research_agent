from typing import Dict, List, Optional, Any
from datetime import datetime
from langgraph.graph import Graph, MessageGraph
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field
from abc import ABC, abstractmethod
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import pandas as pd
from scholarly import scholarly
import time
import requests

os.environ["LANGCHAIN_TRACING"] = "false"
load_dotenv()


# Data Models
class Article(BaseModel):
    title: str
    abstract: Optional[str]
    url: str
    citations: Optional[int]
    publication_date: Optional[int]
    source: str


class ResearchState(BaseModel):
    topic: str
    keywords: List[str] = Field(default_factory=list)
    articles_by_query: Dict[str, List[Article]] = Field(default_factory=dict)  # query -> [Article]
    all_articles: Dict[str, Article] = Field(default_factory=dict)  # title -> Article
    temp_csv_path: str = "research_articles.csv"  # CSV útvonal
    filtered_csv_path: str = ""  # Ez lesz az új filtered CSV útja

    def generate_filtered_path(self) -> str:
        """Generate filtered CSV path based on original path"""
        base, ext = os.path.splitext(self.temp_csv_path)
        return f"{base}_filtered{ext}"


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
        self.system_prompt = """You are a Google Scholar search specialist with conversation skills.
       When receiving feedback:
       - If the user approves (ok, good, yes), return the exact same queries to confirm
       - If they request changes, modify the queries accordingly
       - ALWAYS use the generate_scholar_queries tool to respond
       - Maintain a helpful, conversational tone"""


    def process(self, state: ResearchState) -> ResearchState:
        # Initial query generation
        initial_prompt = f"""Generate academic search queries for: {state.topic}
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
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=initial_prompt)
        ]

        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(messages)

        # Set initial queries
        state.keywords = response.tool_calls[0]["args"]["queries"]

        # Show initial queries
        print("\nI've generated these search queries for your topic:")
        for q in state.keywords:
            print(f"- {q}")
        print("\nWhat do you think about these? Feel free to suggest any changes, or say 'ok' if they look good.")

        # Handle feedback loop
        while True:
            feedback = input("\nYour feedback: ").strip()

            # Process feedback with conversation
            feedback_prompt = f"""Current search queries: {state.keywords}

           User feedback: "{feedback}"

           Instructions:
           - If the user approves (saying ok, good, yes), return the exact same queries to confirm
           - If they want changes, analyze their feedback and modify the queries
           - ALWAYS return queries using the generate_scholar_queries tool
           - Respond conversationally but keep focus on the academic search task

           Analyze the feedback and respond with appropriate queries."""

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=feedback_prompt)
            ]

            response = llm_with_tools.invoke(messages)
            new_queries = response.tool_calls[0]["args"]["queries"]

            # Check if queries are unchanged (user approved) or modified
            if new_queries == state.keywords:
                print("\nI understand you're satisfied with these queries. Moving forward!")
                break
            else:
                print("\nBased on your feedback, I've modified the queries to:")
                state.keywords = new_queries
                for q in state.keywords:
                    print(f"- {q}")
                print("\nHow do these look now?")

        return state


class ScholarlyAgent(ResearchAgent):
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.max_results = 10

    def _show_results(self, articles: List[Article], new_articles: int, state: ResearchState):
        print(f"\nFound {len(articles)} articles ({new_articles} new)")
        # CSV update azonnal
        self.update_csv(state)


    def process(self, state: ResearchState) -> ResearchState:
        print(f"\nI'll now search for articles using the {len(state.keywords)} queries.")
        total_new = 0

        for i, query in enumerate(state.keywords, 1):
            print(f"\nQuery {i}/{len(state.keywords)}: '{query}'")
            articles = self.fetch_articles(query)

            # Store results
            state.articles_by_query[query] = articles

            # Update unique articles
            new_articles = 0
            for article in articles:
                if article.title not in state.all_articles or \
                        article.citations > state.all_articles[article.title].citations:
                    state.all_articles[article.title] = article
                    new_articles += 1
            total_new += new_articles

            # Show minimal results
            print(f"\nFound {len(articles)} articles ({new_articles} new)")

        # A végén egyszer írjuk ki a CSV-t az összes unique cikkel
        self.update_csv(state)
        print(f"\nTotal unique articles collected: {len(state.all_articles)}")

        return state

    def update_csv(self, state: ResearchState):
        """Update CSV with unique articles, sorted by citations and year"""
        # Use unique articles from state.all_articles
        records = []
        for article in state.all_articles.values():
            # Find which queries found this article
            queries = []
            for query, articles in state.articles_by_query.items():
                if any(a.title == article.title for a in articles):
                    queries.append(query)

            records.append({
                'Citations': article.citations or 0,
                'Year': article.publication_date,
                'Title': article.title,
                'Abstract': article.abstract,
                'URL': article.url,
                'Source Query': '; '.join(queries)  # All queries that found this article
            })

        # Create DataFrame and sort
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(
                by=['Citations', 'Year'],
                ascending=[False, False]
            )

            # Ensure column order
            columns = ['Citations', 'Year', 'Title', 'Abstract', 'URL', 'Source Query']
            df = df[columns]

            # Save to CSV
            df.to_csv(state.temp_csv_path, index=False)
            print(f"Saved {len(df)} unique articles to {state.temp_csv_path}")

    def fetch_articlesOLD(self, query: str, max_results: int = 20) -> List[Article]:
        articles = []
        try:
            time.sleep(1)
            search_query = scholarly.search_pubs(query)
            print(f"\nSearching for: '{query}'")

            for i, result in enumerate(search_query):
                if i >= max_results:
                    break

                bib = result.get('bib', {})
                article = Article(
                    title=bib.get('title', 'N/A'),
                    abstract=bib.get('abstract', 'N/A'),
                    url=result.get('eprint_url', 'N/A'),
                    citations=result.get('num_citations', 0),
                    publication_date=str(bib.get('pub_year')),
                    source="Google Scholar"
                )
                articles.append(article)
                print(f"Found: {article.title} ({article.publication_date}, {article.citations} citations)")

        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")

        return articles

    def fetch_articles(self, query: str, max_results: int = 20) -> List[Article]:
        articles = []
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,url,authors,citationCount,publicationDate"
            }
            response = requests.get(url, params=params)

            if response.status_code == 200:
                results = response.json().get('data', [])

                for item in results:

                    # Process abstract for CSV format
                    raw_abstract = item.get('abstract', 'N/A')
                    if raw_abstract:
                        # Remove newlines and excessive spaces, and truncate if too long
                        processed_abstract = " ".join(raw_abstract.split()).strip()

                    article = Article(
                        title=item.get('title', 'N/A'),
                        abstract=processed_abstract,
                        url=item.get('url', 'N/A'),
                        citations=item.get('citationCount', 0),
                        publication_date=str(item.get('publicationDate', '0000'))[:4],
                        source="Semantic Scholar"
                    )
                    articles.append(article)
                    print(f"Found: {article.title} ({article.publication_date}, {article.citations} citations)")
            else:
                print(f"Error fetching data: {response.status_code}")

        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")

        return articles



    def _show_results(self, articles: List[Article], new_articles: int, state: ResearchState):
        print(f"\nFound {len(articles)} articles ({new_articles} new)")
        print("\nTop articles from this search:")
        sorted_articles = sorted(articles, key=lambda x: x.citations or 0, reverse=True)
        for i, art in enumerate(sorted_articles[:5], 1):
            print(f"{i}. [{art.citations} citations] {art.title}")

    def _show_all_articles(self, state: ResearchState):
        print(f"\nAll unique articles collected so far: {len(state.all_articles)}")
        print("\nTop articles overall:")
        sorted_articles = sorted(
            state.all_articles.values(),
            key=lambda x: x.citations or 0,
            reverse=True
        )
        for i, art in enumerate(sorted_articles[:10], 1):
            print(f"{i}. [{art.citations} citations] {art.title}")


class FilterRankAgent(ResearchAgent):
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.current_year = 2024
        self.min_citations = 5
        self.min_citations_per_year = 0.5
        self.min_year = 1990
        self.max_papers = 20


    def _calculate_citation_metrics(self, row: pd.Series) -> tuple:
        """
        Calculate citation metrics:
        - Citations per year
        - Total citations (normalized)
        - Recency score
        """
        year = row['Year']
        citations = row['Citations']

        # Citations per year
        years_since_pub = self.current_year - year + 1
        citations_per_year = citations / years_since_pub

        # Recency score (0-1, newer is higher)
        recency_score = (year - self.min_year) / (self.current_year - self.min_year)

        return citations_per_year, citations, recency_score

    def _initial_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initial filtering and ranking based on citation metrics.
        No LLM used here.
        """
        print(f"Starting initial filter with {len(df)} articles...")

        # Create working copy
        filtered_df = df.copy()

        # Calculate all metrics
        metrics = []
        for idx, row in filtered_df.iterrows():
            cpy, total_cites, recency = self._calculate_citation_metrics(row)
            metrics.append({
                'Title': row['Title'],
                'citations_per_year': cpy,
                'citation_score': total_cites,
                'recency_score': recency
            })

        # Convert metrics to DataFrame and merge with original
        metrics_df = pd.DataFrame(metrics)
        filtered_df = pd.merge(filtered_df, metrics_df, on='Title')

        # Basic filtering
        filtered_df = filtered_df[
            (filtered_df['citations_per_year'] >= self.min_citations_per_year) &
            (filtered_df['Year'] >= self.min_year)
            ]

        print(f"After basic filtering: {len(filtered_df)} articles")

        if filtered_df.empty:
            print("Warning: No articles passed the initial filter. Relaxing criteria...")
            filtered_df = df.copy()
            filtered_df = pd.merge(filtered_df, metrics_df, on='Title')

        # Normalize scores to 0-1 range
        if len(filtered_df) > 0:
            # Citations per year score
            max_cpy = filtered_df['citations_per_year'].max()
            filtered_df['cpy_score'] = filtered_df['citations_per_year'] / max_cpy if max_cpy > 0 else 0

            # Total citations score
            max_cites = filtered_df['Citations'].max()
            filtered_df['citation_score'] = filtered_df['Citations'] / max_cites if max_cites > 0 else 0

            # Calculate final relevance score
            filtered_df['relevance_score'] = (
                    filtered_df['cpy_score'] * 0.4 +  # Citations per year (largest weight)
                    filtered_df['citation_score'] * 0.3 +  # Total citations
                    filtered_df['recency_score'] * 0.3  # Recency
            )

        # Sort by relevance score and keep top papers
        result_df = filtered_df.sort_values(
            by='relevance_score',
            ascending=False
        ).head(self.max_papers)

        # Display metrics for debugging
        print("\nTop 5 papers with metrics:")
        display_cols = ['Title', 'Year', 'Citations', 'citations_per_year',
                        'cpy_score', 'citation_score', 'recency_score', 'relevance_score']
        pd.set_option('display.max_columns', None)
        print(result_df[display_cols].head().to_string())

        return result_df

    def process(self, state: ResearchState) -> ResearchState:
        """Main processing pipeline"""
        self.state = state

        # Generate filtered CSV path
        state.filtered_csv_path = state.generate_filtered_path()
        print(f"Loading data from {state.temp_csv_path}")
        print(f"Filtered results will be saved to {state.filtered_csv_path}")

        df = pd.read_csv(state.temp_csv_path)

        # 1. Initial filtering with metadata and citation metrics
        light_df = df[['Title', 'Year', 'Citations', 'URL', 'Source Query']]
        filtered_df = self._initial_filter(light_df)

        # 2. Get full data for filtered articles
        full_filtered_df = df[df['Title'].isin(filtered_df['Title'])].copy()

        # 3. Add metrics to full data
        final_df = pd.merge(full_filtered_df,
                            filtered_df[['Title', 'citations_per_year', 'cpy_score',
                                         'citation_score', 'recency_score', 'relevance_score']],
                            on='Title')

        # Reorder columns with metrics right after Citations and Year
        column_order = [
            'Title',
            'Year',
            'Citations',
            'citations_per_year',  # Metrikák kezdődnek
            'cpy_score',
            'citation_score',
            'recency_score',
            'relevance_score',  # Metrikák végződnek
            'Abstract',
            'URL',
            'Source Query'
        ]

        # Ellenőrizzük, hogy minden oszlop szerepel-e a sorrendben
        remaining_cols = [col for col in final_df.columns if col not in column_order]
        column_order.extend(remaining_cols)  # Ha van extra oszlop, azt a végére tesszük

        final_df = final_df[column_order]

        # Save filtered results with metrics
        final_df.to_csv(state.filtered_csv_path, index=False)

        print("\nFiltering and ranking complete!")
        print(f"Original articles: {len(df)}")
        print(f"Filtered articles: {len(final_df)}")
        print(f"Results saved to: {state.filtered_csv_path}")

        # Display the first few rows with key columns to verify ordering
        print("\nFirst few rows of filtered results:")
        display_cols = ['Title', 'Year', 'Citations', 'citations_per_year',
                        'cpy_score', 'relevance_score']
        print(final_df[display_cols].head().to_string())

        return state




class ResearchWorkflow:
    """Manages the multi-agent research workflow"""

    def __init__(self, llm_configs: Dict[str, Any]):
        self.agents = {
            "keyword": KeywordAgent(llm_configs.get("keyword")),
            "scholarly": ScholarlyAgent(llm_configs.get("default"))
        }

        self.workflow = self._build_workflow()
        self.state = None

    def _generate_session_filename(self, topic: str) -> str:
        """Generate unique filename for this research session"""
        # Clean topic for filename
        clean_topic = "".join(x for x in topic if x.isalnum() or x in "_ ").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"research_results_{clean_topic}_{timestamp}.csv"

    def _build_workflow(self) -> MessageGraph:
        """Build the workflow graph"""
        workflow = Graph()
        workflow.add_node("keyword", self.agents["keyword"].process)
        workflow.add_node("scholarly", self.agents["scholarly"].process)

        workflow.add_edge(START, "keyword")
        workflow.add_edge("keyword", "scholarly")
        workflow.add_edge("scholarly", END)

        return workflow.compile()

    def run(self, topic: str) -> Dict[str, Any]:
        """Run the research workflow"""
        try:
            print(f"Starting research on: {topic}")

            # Initialize state with unique CSV path
            self.state = ResearchState(
                topic=topic,
                temp_csv_path=self._generate_session_filename(topic)
            )
            print(f"Results will be saved to: {self.state.temp_csv_path}")

            # Process workflow - each agent handles its own feedback
            self.state = self.workflow.invoke(self.state)

            return {
                "topic": self.state.topic,
                "keywords": self.state.keywords,
                # "articles": [a.dict() for a in self.state.ranked_articles],
                # "summary": self.state.final_summary
            }

        except Exception as e:
            print(f"Error in workflow: {str(e)}")
            raise


def test_keyword_agent():
    print("Keyword Agent Test")
    # Configure LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    # Initialize with a test topic
    topic = "Health effects of eggs on cardiovascular health"
    print(f"\nTesting keyword generation for topic: {topic}")

    print("Testing KeywordAgent...")

    state = ResearchState(
        topic=topic
    )
    agent = KeywordAgent(llm)
    result_state = agent.process(state)

    print("\nGenerated keywords:", result_state.keywords)

    return result_state.keywords


def test_scholarly_agent():
    print("Scholarly Agent Interactive Test")

    keywords = ["eggs cardiovascular health meta-analysis",
                "eggs cholesterol heart disease review"]

    filename = 'test1_results.csv'
    state = ResearchState(
        topic="Interactive Test",
        keywords=keywords,
        temp_csv_path=filename
    )

    agent = ScholarlyAgent(None)
    result_state = agent.process(state)

    print(f"\nResults written to: {result_state.temp_csv_path}")


def test_filter_rank_agent():
    print("FilterRank Agent Test")

    # Test CSV paths
    input_csv = 'research_results_Health_effects_of_eggs_on_cardiovascular_health_20241106_212158.csv'

    # Initial state
    state = ResearchState(
        topic="Health effects of eggs on cardiovascular health",
        keywords=["eggs cardiovascular health meta-analysis",
                  "eggs cholesterol heart disease review"],
        temp_csv_path=input_csv
    )

    # Initialize and run agent
    agent = FilterRankAgent(None)

    print("\nStarting initial filtering and ranking...")
    result_state = agent.process(state)

    # Load and display results
    results_df = pd.read_csv(result_state.filtered_csv_path)
    print(f"\nFiltered down to {len(results_df)} articles")
    print("\nTop 5 ranked articles with metrics:")

    # Display formatted results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    display_cols = ['Title', 'Year', 'Citations', 'citations_per_year',
                    'relevance_score']
    print(results_df[display_cols].head().to_string())

    return result_state

test_filter_rank_agent()
import sys
sys.exit()



def test_workflow():
    # Configure LLMs for different agents
    llm_configs = {
        "default": ChatOpenAI(model="gpt-4", temperature=0),
        "keyword": ChatOpenAI(model="gpt-4", temperature=0)
    }

    # Define the test topic
    test_topic = "Health effects of eggs on cardiovascular health"

    # Initialize the workflow
    workflow = ResearchWorkflow(llm_configs)

    # Run the workflow with the test topic
    print(f"\nRunning workflow test for topic: '{test_topic}'")
    results = workflow.run(test_topic)

    # Print out the results
    print("\nTest Workflow Results:")
    print("Keywords generated:", results["keywords"])

    # If article data is available (make sure to uncomment it in `ResearchWorkflow`):
    if "articles" in results:
        print("\nArticles Summary:")
        for i, article in enumerate(results["articles"], 1):
            print(f"{i}. {article['title']} ({article['citations']} citations)")
            print(f"   URL: {article['url']}")
            print(f"   Abstract: {article['abstract'][:200]}...")  # Print first 200 characters of abstract

test_workflow()
import sys
sys.exit()



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
        )
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
    test_keyword_agent()
    # test_scholarly_agent()
    # main()
