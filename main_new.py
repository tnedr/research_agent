from typing import Dict, List, Optional, Any
from datetime import datetime
from langgraph.graph import Graph, MessageGraph
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
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
import logging
from datetime import datetime
import sys




os.environ["LANGCHAIN_TRACING"] = "false"
load_dotenv()



# Set up the logging configuration
logging.basicConfig(
    level=logging.INFO,  # Default level; can be configured as needed
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("research_workflow.log"),  # Logs to a file
        logging.StreamHandler()                        # Logs to the console
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    raw_result_path: str = "raw_results.csv"  # CSV útvonal
    filtered_csv_path: str = ""    # Citation-based filtering results
    title_filtered_path: str = ""  # Title-based filtering results
    synthesis_results_path: str = ""  # Selected relevant papers
    synthesis_md_path: str = ""  # Synthesis markdown document


    def generate_citation_filtered_path(self) -> str:
        base, ext = os.path.splitext(self.raw_result_path)
        return f"{base}_citation_filtered{ext}"

    def generate_title_filtered_path(self) -> str:
        base, ext = os.path.splitext(self.raw_result_path)
        return f"{base}_title_filtered{ext}"

    def generate_synthesis_path(self) -> str:
        base, ext = os.path.splitext(self.raw_results_path)
        return f"{base}_synthesis{ext}"


    def generate_synthesis_paths(self) -> tuple[str, str]:
        """Generate synthesis CSV and markdown paths"""
        base, ext = os.path.splitext(self.raw_results_path)
        return (
            f"{base}_synthesis_results{ext}",
            f"{base}_synthesis.md"
        )


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

# todo lehetne beletenni olyant, hogy a finding vagy a search alapjan uj keywordoket javasol
    # az nagyon tuti lenne, ekkor folyamatosan novekednne az eselye, hogy megtalalja a legjobb
    # papirokat
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

    def process(self, state: ResearchState) -> ResearchState:
        logger.info(f"Starting search for articles using {len(state.keywords)} queries.")
        total_new = 0

        for i, query in enumerate(state.keywords, 1):
            logger.info(f"Query {i}/{len(state.keywords)}: '{query}'")
            articles = self.fetch_articles(query)

            # Store results
            state.articles_by_query[query] = articles

            # Update unique articles
            new_articles = 0
            for article in articles:
                if article.title not in state.all_articles:
                    state.all_articles[article.title] = article
                    new_articles += 1
                else:
                    logger.debug(f"Skipping article, already exists: {article.title}")

            total_new += new_articles
            logger.info(f"Found {len(articles)} articles with data (among them {new_articles} new)")

        # Save unique articles to CSV at the end
        self.update_csv(state)
        logger.info(f"Total unique articles collected: {len(state.all_articles)}")

        return state

    def update_csv(self, state: ResearchState):
        """Update CSV with unique articles, sorted by citations and year."""
        try:
            records = []
            for article in state.all_articles.values():
                # Find which queries found this article
                queries = [query for query, articles in state.articles_by_query.items() if article.title in [a.title for a in articles]]

                records.append({
                    'Citations': article.citations or 0,
                    'Year': article.publication_year,
                    'Title': article.title,
                    'Authors': article.authors,
                    'Abstract': article.abstract,
                    'TLDR': article.tldr,
                    'URL': article.url,
                    'Source Query': '; '.join(queries)
                })

            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values(by=['Citations', 'Year'], ascending=[False, False])
                df.to_csv(state.raw_result_path, index=False)
                logger.info(f"Saved {len(df)} unique articles to {state.raw_result_path}")

        except Exception as e:
            logger.exception(f"Failed to update CSV: {str(e)}")

    def fetch_articles(self, query: str, max_results: int = 20) -> List[Article]:
        """Fetch articles based on the query from an external API."""
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
                        "fields": "title,tldr,abstract,url,venue,authors,citationCount,influentialCitationCount,publicationDate,year,isOpenAccess,openAccessPdf"
                    }
                    response = requests.get(url, params=params)

                    if response.status_code == 200:
                        results = response.json().get('data', [])

                        for bib in results:
                            # Check if all necessary fields are present
                            raw_title = bib.get('title')
                            raw_abstract = bib.get('abstract')
                            raw_tldr = bib.get('tldr')
                            raw_url = bib.get('url')
                            raw_authors = bib.get('authors')
                            raw_citations = bib.get('citationCount')
                            raw_influential_citations = bib.get('influentialCitationCount', 0)
                            raw_year = bib.get('year', 1900)

                            # Ensure mandatory fields are present
                            if raw_title and (raw_abstract or raw_tldr) and raw_url and raw_citations is not None:
                                processed_authors = ', '.join([author['name'] for author in raw_authors])
                                # processed_abstract = raw_abstract.strip() if raw_abstract else 'None'
                                processed_abstract = " ".join(raw_abstract.split()) if raw_abstract else 'None'


                                processed_tldr = raw_tldr['text'] if raw_tldr else 'None'

                                article = Article(
                                    title=raw_title,
                                    authors=processed_authors,
                                    abstract=processed_abstract,
                                    tldr=processed_tldr,
                                    url=raw_url,
                                    citations=raw_citations,
                                    influential_citations=raw_influential_citations,
                                    publication_year=raw_year,
                                    source="Semantic Scholar"
                                )
                                articles.append(article)
                                logger.debug(f"Found: {article.title} ({article.publication_year}, {article.citations} citations)")
                            else:
                                logger.warning(f"Skipping article due to missing fields: {bib}")

                        return articles

                    elif response.status_code == 429:
                        logger.warning(f"Error 429: Too many requests. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= backoff_factor  # Increase delay for the next retry

                    else:
                        logger.error(f"Error fetching data: {response.status_code}")
                        break  # Break loop if it's a different error

                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error: {str(e)}")
                    break  # Exit on network error

        except Exception as e:
            logger.exception(f"Error searching for '{query}': {str(e)}")

        return articles


class CitationFilterAgent(ResearchAgent):
    def __init__(self):
        super().__init__(None)  # Nincs szükség LLM-re ehhez
        self.current_year = 2024
        self.min_citations = 5
        self.min_citations_per_year = 0.5
        self.min_year = 1990
        self.max_papers = 50


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

        logger.debug(f"Metrics for '{row['Title']}': Citations/Year={citations_per_year}, "
                     f"Total Citations={citations}, Recency Score={recency_score}")

        return citations_per_year, citations, recency_score

    def _initial_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initial filtering and ranking based on citation metrics.
        No LLM used here.
        """
        logger.info(f"Starting initial filter with {len(df)} articles...")

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

        logger.info(f"After citation filtering: {len(filtered_df)} articles remaining.")

        if filtered_df.empty:
            logger.warning("No articles passed the citation filter. Relaxing criteria...")
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

        return result_df

    def process(self, state: ResearchState) -> ResearchState:
        """Citation-based filtering process"""
        logger.info(f"Loading data from {state.raw_result_path}")
        state.filtered_csv_path = state.generate_citation_filtered_path()
        logger.info(f"Citation-filtered results will be saved to {state.filtered_csv_path}")

        # Load data
        df = pd.read_csv(state.raw_result_path)

        # Citation-based filtering
        logger.info("Step 1: Starting citation-based filtering...")
        filtered_df = self._initial_filter(df)

        # Save results
        filtered_df.to_csv(state.filtered_csv_path, index=False)
        logger.info(f"Saved {len(filtered_df)} citation-filtered articles to {state.filtered_csv_path}")

        return state


class TitleFilterAgent(ResearchAgent):
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.tools = [{
            "type": "function",
            "function": {
                "name": "analyze_titles_batch",
                "description": "Analyze multiple paper titles for relevance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_analyses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "index": {
                                        "type": "integer",
                                        "description": "Index of the paper in the original list"
                                    },
                                    "relevance_score": {
                                        "type": "number",
                                        "description": "Title relevance score from 0-1"
                                    },
                                    "topic_match": {
                                        "type": "boolean",
                                        "description": "Does title match research topic"
                                    },
                                    "study_type": {
                                        "type": "string",
                                        "description": "Type of study (meta-analysis, review, trial, etc.)"
                                    },
                                    "importance_rank": {
                                        "type": "integer",
                                        "description": "Relative importance rank within the batch (1 being most important)"
                                    },
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Brief explanation of the relevance assessment"
                                    }
                                },
                                "required": ["index", "relevance_score", "topic_match", "study_type", "importance_rank",
                                             "reasoning"]
                            }
                        }
                    },
                    "required": ["paper_analyses"]
                }
            }
        }]

    def _analyze_title_batch(self, papers: List[Dict], topic: str, batch_size: int = 5) -> List[Dict]:
        """Analyze a batch of papers"""
        system_prompt = """You are a research title analysis specialist. 
        Evaluate multiple paper titles for their relevance to the research topic.
        For each paper:
        1. Assess direct topic relevance
        2. Identify study type
        3. Assign relative importance rank within the batch
        Be selective but don't miss valuable papers."""

        papers_info = [
            {
                "index": idx,
                "title": paper['Title'],
                "year": paper['Year'],
                "citations": paper['Citations']
            }
            for idx, paper in enumerate(papers)
        ]

        prompt = f"""Research topic: {topic}

        Analyze these papers for relevance:
        {json.dumps(papers_info, indent=2)}

        For each paper:
        - Assess relevance to: {topic}
        - Identify study type (meta-analysis, review, etc.)
        - Rank importance within this batch
        - Provide brief reasoning

        Return analysis using the analyze_titles_batch function.
        Ensure importance_rank ranges from 1 to {len(papers_info)} with no duplicates."""

        try:
            os.environ["LANGCHAIN_TRACING"] = "false"
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])

            return response.tool_calls[0]["args"]["paper_analyses"]
        except Exception as e:
            print(f"Error in batch analysis: {str(e)}")
            return []

    def process(self, state: ResearchState, batch_size: int = 10) -> ResearchState:
        """Process the citation-filtered papers and filter based on titles"""
        logger.info("Starting title-based filtering...")

        # Generate paths
        base, ext = os.path.splitext(state.filtered_csv_path)
        state.title_filtered_path = f"{base.replace('citation_filtered', 'title_filtered')}{ext}"

        logger.info(f"Loading citation-filtered data from {state.filtered_csv_path}")
        logger.info(f"Title-filtered results will be saved to {state.title_filtered_path}")

        # Load the citation-filtered data
        df = pd.read_csv(state.filtered_csv_path)

        # Initialize statistics
        stats = {
            'total': len(df),
            'processed': 0,
            'included': 0,
            'errors': 0,
            'study_types': {}
        }

        logger.info(f"Analyzing {len(df)} paper titles in batches of {batch_size}...")

        # Process in batches
        results = []

        for start_idx in range(0, len(df), batch_size):
            batch_df = df.iloc[start_idx:start_idx + batch_size]
            batch_papers = batch_df.to_dict('records')

            logger.info(f"Processing batch {start_idx // batch_size + 1}/{(len(df) + batch_size - 1) // batch_size}")

            batch_results = self._analyze_title_batch(batch_papers, state.topic, batch_size)

            # Process batch results
            for analysis in batch_results:
                real_idx = start_idx + analysis['index']

                paper_title = df.iloc[real_idx]['Title']
                logger.debug(f"Analysis for Paper {real_idx + 1}: {paper_title}")
                logger.debug(
                    f"Relevance: {analysis['relevance_score']:.3f}, Topic Match: {'Yes' if analysis['topic_match'] else 'No'}")
                logger.debug(f"Study Type: {analysis['study_type']}, Batch Rank: {analysis['importance_rank']}")
                logger.debug(f"Reasoning: {analysis['reasoning']}")

                # Update statistics
                stats['processed'] += 1
                stats['study_types'][analysis['study_type']] = \
                    stats['study_types'].get(analysis['study_type'], 0) + 1

                if analysis['topic_match'] and analysis['relevance_score'] >= 0.5:
                    stats['included'] += 1
                    results.append({
                        'index': real_idx,
                        'title_relevance': analysis['relevance_score'],
                        'study_type': analysis['study_type'],
                        'batch_rank': analysis['importance_rank'],
                        'reasoning': analysis['reasoning']
                    })

        # Process results
        if results:
            # Sort by relevance score
            results.sort(key=lambda x: (x['title_relevance'], -x['batch_rank']), reverse=True)

            # Create filtered DataFrame
            filtered_indices = [r['index'] for r in results]
            title_filtered_df = df.iloc[filtered_indices].copy()

            # Add analysis results
            title_filtered_df['title_relevance'] = [r['title_relevance'] for r in results]
            title_filtered_df['study_type'] = [r['study_type'] for r in results]
            title_filtered_df['reasoning'] = [r['reasoning'] for r in results]

            title_filtered_df.to_csv(state.title_filtered_path, index=False)
            logger.info(f"Saved {len(title_filtered_df)} title-filtered articles to {state.title_filtered_path}")
        else:
            logger.warning("No papers passed title filtering. Copying citation-filtered data.")
            df.to_csv(state.title_filtered_path, index=False)

        # Print summary
        logger.info("TITLE FILTERING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Papers processed: {stats['processed']}/{stats['total']}")
        logger.info(f"Papers included: {stats['included']} ({stats['included'] / stats['total'] * 100:.1f}%)")
        logger.info(f"Errors: {stats['errors']}")

        logger.info("\nSTUDY TYPES FOUND")

        for study_type, count in sorted(stats['study_types'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{study_type:20} {count:3d} papers")
        logger.info("=" * 80)

        return state


class ContentSynthesisAgent(ResearchAgent):
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.tools = [{
            "type": "function",
            "function": {
                "name": "analyze_paper_findings",
                "description": "Extract relevant findings from a research paper",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Main findings relevant to the research topic"
                        },
                        "methodology": {"type": "string"},
                        "limitations": {"type": "string"},
                        "relevance_explanation": {"type": "string"},
                        "contribution_to_topic": {"type": "string"}
                    },
                    "required": ["key_findings", "methodology", "limitations", "relevance_explanation",
                                 "contribution_to_topic"]
                }
            }
        }]

    def process(self, state: ResearchState) -> ResearchState:
        """Extract findings from title-filtered papers one by one"""
        logger.info("Starting research findings extraction...")

        # Load title-filtered data
        df = pd.read_csv(state.title_filtered_path)

        # Create output paths
        base, _ = os.path.splitext(state.title_filtered_path)
        synthesis_path = f"{base.replace('title_filtered', 'synthesis')}.md"

        system_prompt = """You are a research findings specialist. Your task is to:
        1. Extract key findings specifically related to the research topic
        2. Evaluate the methodology and limitations
        3. Explain how this paper contributes to our understanding of the topic
        Be specific and focus on findings directly relevant to the topic."""

        findings_list = []

        # Process each paper individually
        for idx, row in df.iterrows():
            logger.info(f"\nAnalyzing paper {idx + 1}/{len(df)}:")
            logger.info(f"Title: {row['Title']}")

            paper_prompt = f"""Research topic: {state.topic}

            Analyze this paper for relevant findings:
            Title: {row['Title']}
            Abstract: {row['Abstract']}
            TLDR: {row['TLDR'] if pd.notna(row['TLDR']) else 'None'}
            Study Type: {row['study_type'] if 'study_type' in row else 'Not specified'}
            Year: {row['Year']}

            Extract findings specifically related to: {state.topic}
            Use the analyze_paper_findings function for your analysis."""

            try:
                llm_with_tools = self.llm.bind_tools(self.tools)
                response = llm_with_tools.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=paper_prompt)
                ])

                if response.tool_calls and response.tool_calls[0]["args"]:
                    paper_analysis = response.tool_calls[0]["args"]
                    paper_analysis['title'] = row['Title']
                    paper_analysis['year'] = row['Year']
                    findings_list.append(paper_analysis)

                    # Log findings for this paper
                    logger.info("\nFindings extracted:")
                    for finding in paper_analysis['key_findings']:
                        logger.info(f"- {finding}")

            except Exception as e:
                logger.error(f"Error analyzing paper: {str(e)}")
                continue

        # Create synthesis markdown from all findings
        if findings_list:
            markdown = self._create_synthesis_markdown(findings_list, state.topic)
            with open(synthesis_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            logger.info(f"\nSynthesis saved to: {synthesis_path}")

        return state

    def _create_synthesis_markdown(self, findings_list: List[Dict], topic: str) -> str:
        markdown = f"""# Research Findings Analysis: {topic}

    ## Paper-by-Paper Analysis
    """
        for paper in findings_list:
            markdown += f"\n### {paper['title']} ({paper['year']}):\n"
            markdown += "\n#### Key Findings:\n"
            for finding in paper['key_findings']:
                markdown += f"- {finding}\n"

            markdown += f"\n**Methodology:** {paper['methodology']}\n"
            markdown += f"\n**Limitations:** {paper['limitations']}\n"
            markdown += f"\n**Contribution:** {paper['contribution_to_topic']}\n"
            markdown += f"\n**Relevance:** {paper['relevance_explanation']}\n"
            markdown += "\n---\n"

        return markdown


class EvidenceAnalysisAgent(ResearchAgent):
    def __init__(self, llm: Any):
        super().__init__(llm)
        self.tools = [{
            "type": "function",
            "function": {
                "name": "analyze_research_evidence",
                "description": "Analyze research evidence, conflicts, and gaps",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "evidence_analysis": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "theme": {"type": "string"},
                                    "consensus": {"type": "string"},
                                    "conflicting_findings": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "finding": {"type": "string"},
                                                "supporting_papers": {"type": "array", "items": {"type": "string"}},
                                                "opposing_papers": {"type": "array", "items": {"type": "string"}},
                                                "evidence_strength": {"type": "string"}
                                            }
                                        }
                                    },
                                    "research_gaps": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "methodology_assessment": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "paper_title": {"type": "string"},
                                    "methodology_strength": {"type": "string"},
                                    "limitations": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        }
                    },
                    "required": ["evidence_analysis", "methodology_assessment"]
                }
            }
        }]

    def process(self, state: ResearchState) -> ResearchState:
        """Analyze evidence and create detailed analysis"""
        print("\nStarting evidence analysis...")

        # Load both synthesis and original data
        with open(state.synthesis_path, 'r', encoding='utf-8') as f:
            synthesis_content = f.read()

        df = pd.read_csv(state.final_csv_path)

        # Create analysis file path
        base, _ = os.path.splitext(state.final_csv_path)
        analysis_path = f"{base}_evidence_analysis.md"

        system_prompt = """You are a research evidence analyst specializing in:
        1. Identifying consensus and conflicts in research findings
        2. Evaluating strength of evidence
        3. Assessing methodology quality
        4. Identifying research gaps
        Be critical and thorough in your analysis."""

        # Prepare data for analysis
        papers_data = []
        for _, row in df.iterrows():
            papers_data.append({
                'title': row['Title'],
                'abstract': row['Abstract'],
                'key_findings': row['key_findings'],
                'methodology_quality': row['methodology_quality'],
                'findings_significance': row['findings_significance']
            })

        prompt = f"""Research topic: {state.topic}

        Previous synthesis:
        {synthesis_content}

        Analyze the evidence in these papers:
        {json.dumps(papers_data, indent=2)}

        Focus on:
        1. Identifying areas of consensus and conflict
        2. Evaluating strength of evidence
        3. Assessing methodology quality
        4. Identifying research gaps

        Use the analyze_research_evidence function for your analysis."""

        try:
            # Get analysis from LLM
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])

            # Get analysis from tool call
            analysis = response.tool_calls[0]["args"]

            # Create markdown content
            markdown_content = self._create_markdown(analysis)

            # Save analysis
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"\nEvidence analysis complete!")
            print(f"Results saved to: {analysis_path}")

            return state

        except Exception as e:
            print(f"Error in evidence analysis: {e}")
            raise

    def _create_markdown(self, analysis: Dict) -> str:
        """Create markdown document from analysis results"""
        markdown = """# Evidence Analysis Report

## Theme-based Analysis
"""
        for theme_analysis in analysis['evidence_analysis']:
            markdown += f"\n### {theme_analysis['theme']}\n"
            markdown += f"\nConsensus:\n{theme_analysis['consensus']}\n"

            markdown += "\n#### Conflicting Findings\n"
            for conflict in theme_analysis['conflicting_findings']:
                markdown += f"\n**Finding**: {conflict['finding']}\n"
                markdown += f"- Evidence Strength: {conflict['evidence_strength']}\n"
                markdown += "- Supporting Papers:\n"
                for paper in conflict['supporting_papers']:
                    markdown += f"  - {paper}\n"
                markdown += "- Opposing Papers:\n"
                for paper in conflict['opposing_papers']:
                    markdown += f"  - {paper}\n"

            markdown += "\n#### Research Gaps\n"
            for gap in theme_analysis['research_gaps']:
                markdown += f"- {gap}\n"

        markdown += "\n## Methodology Assessment\n"
        for method in analysis['methodology_assessment']:
            markdown += f"\n### {method['paper_title']}\n"
            markdown += f"Strength: {method['methodology_strength']}\n"
            markdown += "Limitations:\n"
            for limitation in method['limitations']:
                markdown += f"- {limitation}\n"

        return markdown


class ResearchWorkflow:
    """Manages the multi-agent research workflow"""
    # Új folyamat:
    # 1. KeywordAgent -> keywords
    # 2. PublicationSearchAgent -> raw_results.csv
    # 3. CitationFilterAgent -> citation_filtered.csv
    # 4. TitleFilterAgent -> title_filtered.csv
    # 5. ContentSynthesisAgent -> synthesis_results.csv

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
                raw_result_path=self._generate_session_filename(topic)
            )
            print(f"Results will be saved to: {self.state.raw_result_path}")

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
# test_keyword_agent()
# sys.exit()


def test_publication_search_agent():
    print("PublicationSearchAgent Interactive Test")

    keywords = ["eggs cardiovascular health meta-analysis",
                "eggs cholesterol heart disease review"]

    filename = 'test_publication_search_agent_result.csv'
    state = ResearchState(
        topic="Interactive Test",
        keywords=keywords,
        raw_result_path=filename
    )

    agent = PublicationSearchAgent(None)
    result_state = agent.process(state)

    print(f"\nResults written to: {result_state.raw_result_path}")
# test_publication_search_agent()
# sys.exit()



def test_filtering_agents():
    input_csv = 'test_publication_search_agent_result.csv'
    state = ResearchState(
        topic="Health effects of eggs on cardiovascular health",
        keywords=["eggs cardiovascular health meta-analysis",
                 "eggs cholesterol heart disease review"],
        raw_result_path=input_csv
    )

    # 1. Citation filtering
    citation_agent = CitationFilterAgent()
    state = citation_agent.process(state)

    # 2. Title filtering with batch processing
    print("\nStep 2: Title-based filtering")
    print("=" * 80)
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model="llama3-70b-8192",
        groq_api_key=groq_api_key,
        temperature=0,
        max_tokens=1024
    )
    title_agent = TitleFilterAgent(llm)
    state = title_agent.process(state, batch_size=10)  # Set batch size in process call

    return state
# test_filtering_agents()
# sys.exit()


def test_content_synthesis():
    print("ContentSynthesisAgent Test")

    input_csv = 'test_publication_search_agent_result_title_filtered.csv'
    state = ResearchState(
        topic="Health effects of eggs on cardiovascular health",
        keywords=["eggs cardiovascular health meta-analysis",
                  "eggs cholesterol heart disease review"],
        raw_result_path=input_csv,
        title_filtered_path=input_csv  # Set as both temp and title_filtered for test
    )

    # Initialize with Groq LLM
    # groq_api_key = os.getenv("GROQ_API_KEY")
    # llm = ChatGroq(
    #     model="llama3-70b-8192",
    #     groq_api_key=groq_api_key,
    #     temperature=0,
    #     max_tokens=1024
    # )
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    synthesis_agent = ContentSynthesisAgent(llm)

    print("\nStarting synthesis test...")
    print("=" * 80)
    state = synthesis_agent.process(state)

    # Load and display results
    title_df = pd.read_csv(state.title_filtered_path)
    synthesis_df = pd.read_csv(state.synthesis_results_path)

    print("\nSYNTHESIS RESULTS")
    print("=" * 80)
    print(f"Input papers:          {len(title_df)}")
    print(f"Selected papers:       {len(synthesis_df)}")
    print(f"Selection rate:        {len(synthesis_df) / len(title_df) * 100:.1f}%")

    if len(synthesis_df) > 0:
        print("\nTop 5 most relevant papers:")
        print("-" * 80)
        display_cols = ['Title', 'Year', 'relevance_score', 'selection_reasoning']
        print(synthesis_df[display_cols].head().to_string())

    # Check if synthesis markdown was created
    base, _ = os.path.splitext(state.title_filtered_path)
    synthesis_md_path = f"{base.replace('title_filtered', 'synthesis')}.md"

    if os.path.exists(synthesis_md_path):
        with open(synthesis_md_path, 'r', encoding='utf-8') as f:
            synthesis_content = f.read()
            print("\nSynthesis document created successfully")
            print(f"Length: {len(synthesis_content)} characters")
    else:
        print("\nWarning: Synthesis document was not created")

    return state
test_content_synthesis()
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
# test_workflow()
# import sys
# sys.exit()

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
    # test_keyword_agent()
    # test_scholarly_agent()
    main()
