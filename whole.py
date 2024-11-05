# research_agent.py

from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.tools import Tool

# 1. Kulcsszógeneráló ágens
class KeywordAgent:
    def __init__(self, model_name='gpt-4'):
        self.llm = OpenAI(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["topic"],
            template="Generate relevant keywords for the following topic: {topic}"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_keywords(self, topic):
        response = self.chain.invoke({"topic": topic})
        return response.split(', ')

# 2. Adatgyűjtő ágens
class DataAgent:
    def __init__(self):
        self.search_tool = Tool(
            name="PubMed Search",
            func=self.pubmed_search,
            description="Search for articles in PubMed based on keywords"
        )

    def pubmed_search(self, query):
        # Itt implementálhatod a PubMed API keresést
        pass

    def fetch_articles(self, keywords):
        articles = []
        for keyword in keywords:
            result = self.search_tool.run(keyword)
            articles.extend(result)
        return articles

    def rank_articles(self, articles):
        # Rangsorolási logika implementálása
        pass

# 3. Letöltő ágens
class DownloadAgent:
    def __init__(self):
        self.download_tool = Tool(
            name="Article Downloader",
            func=self.download_article,
            description="Download article abstracts or full texts"
        )

    def download_article(self, article_id):
        # Implementálhatod a cikkek letöltését DOI alapján
        pass

    def fetch_abstract(self, article):
        doi = article.get('doi')
        abstract = self.download_tool.run(doi)
        return abstract

# 4. Elemző ágens
class AnalyzeAgent:
    def __init__(self, model_name='gpt-4'):
        self.llm = OpenAI(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["abstract", "question"],
            template="Read the following abstract and answer the question: {abstract} Question: {question}"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def analyze_abstract(self, abstract, question):
        response = self.chain.invoke({"abstract": abstract, "question": question})
        return response

    def summarize_findings(self, findings):
        # Összegzés logika, ha szükséges
        pass

# 5. Fővezérlő funkció
class ResearchController:
    def __init__(self):
        self.keyword_agent = KeywordAgent()
        self.data_agent = DataAgent()
        self.download_agent = DownloadAgent()
        self.analyze_agent = AnalyzeAgent()

    def run_research(self, topic, question):
        # 1. Kulcsszavak generálása
        keywords = self.keyword_agent.generate_keywords(topic)
        print(f"Generált kulcsszavak: {keywords}")

        # 2. Adatgyűjtés és rangsorolás
        articles = self.data_agent.fetch_articles(keywords)
        ranked_articles = self.data_agent.rank_articles(articles)
        print(f"Rangsorolt cikkek: {len(ranked_articles)}")

        # 3. Letöltés
        for article in ranked_articles:
            abstract = self.download_agent.fetch_abstract(article)
            article["abstract"] = abstract  # Absztrakt hozzáadása a cikkhez

        # 4. Elemzés
        findings = []
        for article in ranked_articles:
            finding = self.analyze_agent.analyze_abstract(article["abstract"], question)
            findings.append(finding)
            print(f"Elemzés eredménye: {finding}")

        # 5. Összegzés
        summary = self.analyze_agent.summarize_findings(findings)
        return summary
