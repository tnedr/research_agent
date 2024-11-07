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
import json
from typing import List

os.environ["LANGCHAIN_TRACING"] = "false"
load_dotenv()


# Data Models
class Article(BaseModel):
    title: str
    authors: str
    abstract: Optional[str]
    tldr: Optional[str]
    url: str
    citations: Optional[int]
    influential_citations: Optional[int]
    publication_year: Optional[int]
    source: str


class ResearchState(BaseModel):
    topic: str
    keywords: List[str] = Field(default_factory=list)
    articles_by_query: Dict[str, List[Article]] = Field(default_factory=dict)  # query -> [Article]
    all_articles: Dict[str, Article] = Field(default_factory=dict)  # title -> Article
    temp_csv_path: str = "research_articles.csv"  # CSV útvonal
    filtered_csv_path: str = ""  # Citation-based filtering results
    final_csv_path: str = ""     # LLM-based filtering results

    def generate_filtered_path(self) -> str:
        """Generate citation-filtered CSV path based on original path"""
        base, ext = os.path.splitext(self.temp_csv_path)
        return f"{base}_citation_filtered{ext}"

    def generate_final_path(self) -> str:
        """Generate final filtered CSV path"""
        base, ext = os.path.splitext(self.temp_csv_path)
        return f"{base}_final_filtered{ext}"


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


class PublicationSearchAgent(ResearchAgent):
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
                # if article.title not in state.all_articles or \
                #         article.citations > state.all_articles[article.title].citations:
                if article.title not in state.all_articles:
                    state.all_articles[article.title] = article
                    new_articles += 1
                else:
                    print('Skipping article, already exist:', article.title)
            total_new += new_articles

            # Show minimal results
            print(f"\nFound {len(articles)} articles with data (among them {new_articles} new)")

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
                'Year': article.publication_year,
                'Title': article.title,
                'Authors': article.authors,
                'Abstract': article.abstract,
                'TLDR': article.tldr,
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
            columns = ['Citations', 'Year', 'Title', 'Authors', 'Abstract', 'TLDR', 'URL', 'Source Query']
            df = df[columns]

            # Save to CSV
            df.to_csv(state.temp_csv_path, index=False)
            print(f"Saved {len(df)} unique articles to {state.temp_csv_path}")

    def fetch_articles(self, query: str, max_results: int = 20) -> List[Article]:
        articles = []

        articles = []
        retries = 5  # Maximum retry attempts
        backoff_factor = 2  # Factor for exponential backoff (1s, 2s, 4s, etc.)
        delay = 1  # Initial delay in seconds

        try:
            for attempt in range(retries):

                try:
                    url = "https://api.semanticscholar.org/graph/v1/paper/search"
                    params = {
                        "query": query,
                        "limit": max_results,
                        # https://api.semanticscholar.org/api-docs#tag/Paper-Data/operation/post_graph_get_papers
                        "fields": "title,tldr,abstract,url,venue,authors,citationCount,influentialCitationCount,publicationDate,year,isOpenAccess,openAccessPdf"
                    }
                    response = requests.get(url, params=params)

                    if response.status_code == 200:
                        results = response.json().get('data', [])

                        for bib in results:
                            # Ellenőrizzük, hogy minden szükséges mező megvan-e
                            raw_title = bib.get('title')
                            raw_abstract = bib.get('abstract')
                            raw_tldr = bib.get('tldr')
                            raw_url = bib.get('url')
                            raw_authors = bib.get('authors')
                            raw_citations = bib.get('citationCount')
                            raw_influential_citations = bib.get('influentialCitationCount') # todo implement this
                            raw_year = bib.get('year')

                            # Csak akkor dolgozzuk fel, ha minden mező létezik és nem None
                            if raw_title and (raw_abstract or raw_tldr) and raw_url and raw_citations is not None:
                                if raw_year is None:
                                    raw_year = 1900
                                # Process abstract for CSV format
                                if raw_abstract is None:
                                    processed_abstract = 'None'
                                else:
                                    processed_abstract = " ".join(raw_abstract.split()).strip()
                                if raw_tldr is None:
                                    processed_tldr = 'None'
                                else:
                                    processed_tldr = raw_tldr['text']
                                if raw_influential_citations is None:
                                    raw_influential_citations = 0
                                processed_authors = ', '.join([author['name'] for author in raw_authors])

                                article = Article(
                                    title=raw_title,
                                    authors=processed_authors,
                                    abstract=processed_abstract,
                                    tldr=processed_tldr,
                                    url=raw_url,
                                    citations=raw_citations,
                                    influential_citations=raw_influential_citations,
                                    publication_year=raw_year,
                                    source="Google Scholar"
                                )
                                articles.append(article)
                                print(f"Found: {article.title} ({article.publication_year}, {article.citations} citations)")
                            else:
                                print(f"Skipping article due to missing fields: {bib}")

                        return articles

                    elif response.status_code == 429:
                        print(f"Error 429: Too many requests. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= backoff_factor  # Increase delay for the next retry

                    else:
                        print(f"Error fetching data: {response.status_code}")
                        break  # Break loop if it's a different error

                except requests.exceptions.RequestException as e:
                    print(f"Request error: {str(e)}")
                    break  # Exit on network error

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

        # Tool for consistent output format
        self.tools = [{
            "type": "function",
            "function": {
                "name": "analyze_paper_relevance",
                "description": "Analyze paper relevance and content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relevance_score": {
                            "type": "number",
                            "description": "Relevance score from 0-1"
                        },
                        "methodology_score": {
                            "type": "number",
                            "description": "Methodology quality score from 0-1"
                        },
                        "findings_score": {
                            "type": "number",
                            "description": "Significance of findings score from 0-1"
                        },
                        "key_findings": {
                            "type": "string",
                            "description": "Brief summary of key findings related to the topic"
                        },
                        "include": {
                            "type": "boolean",
                            "description": "Whether to include this paper in the final set"
                        }
                    },
                    "required": ["relevance_score", "methodology_score",
                                 "findings_score", "key_findings", "include"]
                }
            }
        }]



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

    def _llm_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Content-based filtering and analysis using LLM"""
        system_prompt = """You are a research paper analysis specialist.
        Evaluate papers for their relevance to our research topic and analyze their content.
        Consider:
        1. How directly the paper addresses our topic
        2. Quality and rigor of methodology
        3. Significance and clarity of findings
        4. Usefulness of conclusions for our topic

        Be critical and selective - only include papers that provide substantial value."""

        llm_with_tools = self.llm.bind_tools(self.tools)
        results = []

        for idx, row in df.iterrows():
            print(f"\nAnalyzing paper {idx + 1}/{len(df)}: {row['Title'][:100]}...")

            prompt = f"""Research topic: {self.state.topic}

            Paper Title: {row['Title']}
            TLDR: {row['TLDR']} 
            Abstract: {row['Abstract']}
            Year: {row['Year']}
            Citations: {row['Citations']}

            Analyze this paper's relevance and content.
            - For relevance_score: focus on direct connection to our topic
            - For methodology_score: evaluate study design and rigor
            - For findings_score: assess significance and clarity of results
            - For key_findings: extract main conclusions relevant to our topic
            - For include: be selective, only true if paper adds substantial value

            Use the analyze_paper_relevance function to provide your analysis."""

            try:
                response = llm_with_tools.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ])

                # Get analysis from tool call
                llm_analysis = response.tool_calls[0]["args"]

                # Add scores and analysis to row
                row_dict = row.to_dict()

                content_score = llm_analysis['relevance_score'] * 0.4 + llm_analysis['methodology_score'] * 0.3 + llm_analysis['findings_score'] * 0.3
                d_new = {
                    'llm_relevance': llm_analysis['relevance_score'],
                    'methodology_quality': llm_analysis['methodology_score'],
                    'findings_significance': llm_analysis['findings_score'],
                    'key_findings': llm_analysis['key_findings'],
                    'content_score': content_score
                }

                row_dict.update(d_new)

                # Calculate final score combining metrics and content
                row_dict['final_score'] = (
                        row_dict['relevance_score'] * 0.3 +  # Citation-based
                        row_dict['content_score'] * 0.7  # Content-based
                )

                if llm_analysis['include']:
                    results.append(row_dict)

            except Exception as e:
                print(f"Error in LLM analysis for {row['Title']}: {e}")
                continue

        result_df = pd.DataFrame(results)

        # Sort by final score
        result_df = result_df.sort_values('final_score', ascending=False)

        # Reorder columns
        column_order = [
            'Title',
            'Year',
            'Citations',
            # Citation-based metrics
            'citations_per_year',
            'cpy_score',
            'citation_score',
            'recency_score',
            'relevance_score',
            # Content-based metrics
            'llm_relevance',
            'methodology_quality',
            'findings_significance',
            'content_score',
            'final_score',
            # Content
            'key_findings',
            'Abstract',
            'TLDR',
            'URL',
            'Source Query'
        ]

        # Handle any additional columns
        remaining_cols = [col for col in result_df.columns if col not in column_order]
        column_order.extend(remaining_cols)

        return result_df[column_order]

    def process(self, state: ResearchState) -> ResearchState:
        """Main processing pipeline"""
        self.state = state

        # Generate paths for both steps
        state.filtered_csv_path = state.generate_filtered_path()
        state.final_csv_path = state.generate_final_path()

        print(f"Loading data from {state.temp_csv_path}")
        print(f"Citation-filtered results will be saved to {state.filtered_csv_path}")
        print(f"Final filtered results will be saved to {state.final_csv_path}")

        # 1. Load full data
        df = pd.read_csv(state.temp_csv_path)

        # 2. Initial filtering with citation metrics
        print("\nStep 1: Citation-based filtering...")
        filtered_df = self._initial_filter(df)

        # Save citation-based filtering results
        filtered_df.to_csv(state.filtered_csv_path, index=False)
        print(f"\nSaved {len(filtered_df)} citation-filtered articles to {state.filtered_csv_path}")

        # 3. LLM-based filtering and analysis
        if self.llm:
            print("\nStep 2: Content-based filtering and analysis...")
            final_df = self._llm_filter(filtered_df)

            # Save final results
            final_df.to_csv(state.final_csv_path, index=False)

            print("\nFiltering and analysis complete!")
            print(f"Original articles: {len(df)}")
            print(f"After citation filter: {len(filtered_df)}")
            print(f"After content filter: {len(final_df)}")

            print("\nResults saved to:")
            print(f"1. Citation-filtered: {state.filtered_csv_path}")
            print(f"2. Final results: {state.final_csv_path}")

            # Display sample results from both steps
            print("\nTop 5 papers after citation filtering:")
            citation_cols = ['Title', 'Year', 'Citations',
                             'citations_per_year', 'relevance_score']
            print(filtered_df[citation_cols].head().to_string())

            print("\nTop 5 papers after content filtering:")
            final_cols = ['Title', 'Year', 'Citations',
                          'relevance_score', 'content_score', 'final_score']
            print(final_df[final_cols].head().to_string())
        else:
            # If no LLM available, only save citation-filtered results
            print("\nLLM not provided, stopping after citation-based filtering.")
            print(f"Results saved to: {state.filtered_csv_path}")

        return state




class ResearchWorkflow:
    """Manages the multi-agent research workflow"""

    def __init__(self, llm_configs: Dict[str, Any]):
        self.agents = {
            "keyword": KeywordAgent(llm_configs.get("keyword")),
            "scholarly": PublicationSearchAgent(llm_configs.get("default"))
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


def test_publication_search_agent():
    print("PublicationSearchAgent Interactive Test")

    keywords = ["eggs cardiovascular health meta-analysis",
                "eggs cholesterol heart disease review"]

    filename = 'test1_results.csv'
    state = ResearchState(
        topic="Interactive Test",
        keywords=keywords,
        temp_csv_path=filename
    )

    agent = PublicationSearchAgent(None)
    result_state = agent.process(state)

    print(f"\nResults written to: {result_state.temp_csv_path}")
test_publication_search_agent()
import sys
sys.exit()

def test_filter_rank_agent():
    print("FilterRank Agent Test")

    input_csv = 'test1_results_used.csv'

    state = ResearchState(
        topic="Health effects of eggs on cardiovascular health",
        keywords=["eggs cardiovascular health meta-analysis",
                  "eggs cholesterol heart disease review"],
        temp_csv_path=input_csv
    )

    # Initialize with LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    agent = FilterRankAgent(llm)

    result_state = agent.process(state)

    # Load and display results from both steps
    citation_df = pd.read_csv(result_state.filtered_csv_path)
    final_df = pd.read_csv(result_state.final_csv_path)

    print("\nSummary of filtering steps:")
    print(f"Citation-filtered articles: {len(citation_df)}")
    print(f"Final filtered articles: {len(final_df)}")

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
