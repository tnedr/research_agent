from Bio import Entrez
import time
from scholarly import scholarly
import sys

import pandas as pd
from scholarly import scholarly


# Google Scholar keresés
def fetch_scholar_data(query, max_results=10):
    results = []
    search_query = scholarly.search_pubs(query)

    for i, result in enumerate(search_query):
        if i >= max_results:  # Csak max_results számú találat
            break

        # Releváns adatok kigyűjtése
        title = result['bib'].get('title', 'N/A')
        abstract = result['bib'].get('abstract', 'N/A')
        num_citations = result.get('num_citations', 0)
        pub_year = result['bib'].get('pub_year', 'N/A')
        author = ', '.join(result['bib'].get('author', []))
        url_scholar = result.get('eprint_url', 'N/A')  # Ha van link a cikkhez

        results.append({
            'Title': title,
            'Abstract': abstract,
            'Citations': num_citations,
            'Year': pub_year,
            'Author': author,
            'Google Scholar Link': url_scholar
        })

    return pd.DataFrame(results)


# Használat
query = ("eggs AND cholesterol")
df = fetch_scholar_data(query, max_results=10)

# Az adatkeret megjelenítése
print(df)