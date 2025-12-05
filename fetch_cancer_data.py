import arxiv
import os

# Define Cancer-specific queries
# q-bio.TO = Tissues and Organs (often covers tumors)
# or simple keywords like "Oncology", "Immunotherapy", "Carcinoma"
QUERY = "all:cancer OR all:oncology OR all:immunotherapy OR all:tumor"
MAX_RESULTS = 10
DATA_PATH = "knowledge_base"

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

print(f"🧬 specialized OncoBot Scraper started...")
print(f"🔍 Searching arXiv for {MAX_RESULTS} oncology papers...")

client = arxiv.Client()
search = arxiv.Search(
    query = QUERY,
    max_results = MAX_RESULTS,
    sort_by = arxiv.SortCriterion.Relevance
)

count = 0
for result in client.results(search):
    # Sanitize filename
    safe_title = "".join([c for c in result.title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    filename = f"{safe_title}.pdf"
    filepath = os.path.join(DATA_PATH, filename)
    
    if not os.path.exists(filepath):
        print(f"⬇️ Downloading: {result.title}")
        result.download_pdf(dirpath=DATA_PATH, filename=filename)
        count += 1
    else:
        print(f"✅ Exists: {result.title}")

print(f"🎉 Download complete! Added {count} new cancer research papers.")