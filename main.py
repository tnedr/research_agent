from Bio import Entrez
import time
from scholarly import scholarly
import sys

# Keresés kulcsszavak alapján a Google Scholar-on
search_query = scholarly.search_pubs('eggs and cholesterol')

for result in search_query:
    print(f"Cím: {result['bib']['title']}")
    print(f"Citációk száma: {result['num_citations']}")
    print(f"Absztrakt: {result['bib']['abstract'][:200]}...")

sys.exit()

class ResearchArticle:
    def __init__(self, title, abstract, citations, pmid, year):
        self.title = title
        self.abstract = abstract
        self.citations = citations
        self.pmid = pmid
        self.year = year

    def __repr__(self):
        return f"{self.title} ({self.year}), Citations: {self.citations}"


def fetch_paper_ids(keywords, email, max_results=100):
    Entrez.email = email
    search_handle = Entrez.esearch(db="pubmed", term=keywords, retmax=max_results)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    return search_results["IdList"]


def fetch_paper_details(paper_ids):
    papers = []
    batch_size = 20

    for i in range(0, len(paper_ids), batch_size):
        batch_ids = paper_ids[i:i + batch_size]
        try:
            handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="xml", retmode="xml")
            records = Entrez.read(handle)['PubmedArticle']
            handle.close()

            for record in records:
                article = record['MedlineCitation']['Article']
                title = article['ArticleTitle']
                abstract = article['Abstract']['AbstractText'][0] if 'Abstract' in article else ""
                pmid = record['MedlineCitation']['PMID']
                year = article['Journal']['JournalIssue']['PubDate'].get('Year', 'N/A')

                citation_count = fetch_citation_count(pmid)

                if citation_count > 0:  # Filter out papers with zero citations
                    papers.append(ResearchArticle(title, abstract, citation_count, pmid, year))

            time.sleep(1)  # API korlátozások miatt

        except Exception as e:
            print(f"Hiba történt a következő ID-k feldolgozásakor: {batch_ids}")
            print(f"Hiba: {str(e)}")
            continue

    return papers


def fetch_citation_count(pmid):
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pubmed", linkname="pubmed_pubmed_cites", id=pmid)
        citation_data = Entrez.read(handle)
        handle.close()

        if citation_data[0]['LinkSetDb']:
            citation_count = len(citation_data[0]['LinkSetDb'][0]['Link'])
            print(f"PMID {pmid} has {citation_count} citations.")
            return citation_count
        else:
            print(f"PMID {pmid} has no citation links.")
    except Exception as e:
        print(f"Error retrieving citation count for PMID {pmid}: {e}")
        return 0

    return 0


def fetch_top_cited_papers(keywords, email, max_results=20):
    paper_ids = fetch_paper_ids(keywords, email)
    papers = fetch_paper_details(paper_ids)

    # Rendezés citációk száma alapján és top eredmények visszaadása
    sorted_papers = sorted(papers, key=lambda x: x.citations, reverse=True)
    return sorted_papers[:max_results]


# Példa használat:
if __name__ == "__main__":
    keywords = "cancer"
    email = "your.email@example.com"

    top_papers = fetch_top_cited_papers(keywords, email)

    for i, paper in enumerate(top_papers, 1):
        print(f"\n{i}. Cikk:")
        print(f"Cím: {paper.title}")
        print(f"Citációk száma: {paper.citations}")
        print(f"Év: {paper.year}")
        print(f"PMID: {paper.pmid}")
        print(f"Absztrakt: {paper.abstract[:200]}...")
