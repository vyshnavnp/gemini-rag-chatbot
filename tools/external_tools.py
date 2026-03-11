# tools/external_tools.py — External API tools (ClinicalTrials.gov, PubMed, arXiv).

import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

import arxiv
from langchain_core.tools import tool


@tool
def search_clinical_trials(condition: str, phase: str = "") -> str:
    """
    Search ClinicalTrials.gov for active or recruiting clinical trials
    related to a specific cancer condition or treatment.

    Use this tool when:
    - The user asks about available clinical trials for a cancer type
    - The user asks "are there any trials for [cancer/drug]?"
    - The user wants to know about experimental treatments

    Args:
        condition: The cancer type or treatment to search for.
                   Examples: "lung cancer", "pembrolizumab melanoma",
                   "triple negative breast cancer immunotherapy"
        phase:     Optional trial phase filter. Use "1", "2", "3", or "4".
                   Leave empty to return trials of all phases.

    Returns:
        A formatted string listing up to 5 matching clinical trials with
        their title, status, phase, and ClinicalTrials.gov identifier (NCT ID).
        Returns an error message if the API call fails.
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,BriefSummary",
        "pageSize": "5",
        "format": "json",
    }

    if phase:
        params["filter.phase"] = f"PHASE{phase}"

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        return f"ClinicalTrials.gov API call failed: {str(e)}"

    studies = data.get("studies", [])
    if not studies:
        return (
            f"No recruiting clinical trials found on ClinicalTrials.gov "
            f"for: {condition}"
        )

    results = [f"Clinical trials found for '{condition}':\n"]

    for study in studies:
        protocol = study.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        desc_module = protocol.get("descriptionModule", {})
        design_module = protocol.get("designModule", {})

        nct_id = id_module.get("nctId", "N/A")
        title = id_module.get("briefTitle", "N/A")
        status = status_module.get("overallStatus", "N/A")
        phases = design_module.get("phases", ["N/A"])
        phase_str = ", ".join(phases) if phases else "N/A"
        summary = desc_module.get("briefSummary", "No summary available.")

        if len(summary) > 250:
            summary = summary[:250] + "..."

        results.append(
            f"NCT ID : {nct_id}\n"
            f"Title  : {title}\n"
            f"Status : {status}\n"
            f"Phase  : {phase_str}\n"
            f"Summary: {summary}\n"
        )

    return "\n".join(results)


@tool
def fetch_pubmed_abstracts(query: str) -> str:
    """
    Search PubMed for recent oncology research papers and return their
    titles and abstracts.

    Use this tool when:
    - The user asks about "latest research" or "recent studies"
    - The user wants to know what the current literature says about a topic
    - The local knowledge base does not have enough up-to-date information
    - The user asks about a very specific drug, mutation, or therapy that
      requires recent published evidence

    This uses NCBI's free E-utilities API and requires no authentication.

    Args:
        query: The PubMed search query string.
               Examples: "KRAS mutation colorectal cancer 2024",
               "CAR-T cell therapy multiple myeloma"

    Returns:
        A formatted string with up to 3 paper titles and their abstracts.
        Returns an error message if the API call fails.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    search_params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": "3",
        "sort": "relevance",
        "retmode": "json",
    })
    search_url = f"{base_url}/esearch.fcgi?{search_params}"

    try:
        with urllib.request.urlopen(search_url, timeout=10) as response:
            search_data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        return f"PubMed search failed: {str(e)}"

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return f"No PubMed results found for: {query}"

    fetch_params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(id_list),
        "rettype": "abstract",
        "retmode": "xml",
    })
    fetch_url = f"{base_url}/efetch.fcgi?{fetch_params}"

    try:
        with urllib.request.urlopen(fetch_url, timeout=10) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        return f"PubMed abstract fetch failed: {str(e)}"

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        return f"Failed to parse PubMed response XML: {str(e)}"

    results = [f"PubMed abstracts for '{query}':\n"]

    for article in root.findall(".//PubmedArticle"):
        title_el = article.find(".//ArticleTitle")
        title = title_el.text if title_el is not None else "No title"

        abstract_parts = article.findall(".//AbstractText")
        if abstract_parts:
            abstract = " ".join(
                part.text for part in abstract_parts if part.text
            )
        else:
            abstract = "Abstract not available."

        if len(abstract) > 400:
            abstract = abstract[:400] + "..."

        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None else "Unknown year"

        results.append(
            f"Title ({year}): {title}\n"
            f"Abstract: {abstract}\n"
        )

    if len(results) == 1:
        return f"No abstract content found for query: {query}"

    return "\n".join(results)


@tool
def summarize_arxiv_paper(arxiv_id: str) -> str:
    """
    Fetch the metadata and abstract of a specific arXiv paper by its ID
    and return a structured summary.

    Use this tool when:
    - The user mentions a specific arXiv paper ID (e.g., "2301.12345")
    - The user wants to know more about a specific paper the bot referenced
    - The user asks to "look up the paper" or "find the original study"

    This does NOT download the full PDF -- it returns the title, authors,
    abstract, and publication date from the arXiv API. This is fast and
    does not require storage.

    Args:
        arxiv_id: The arXiv paper identifier string.
                  Examples: "2301.12345", "arxiv:2301.12345", "2301.12345v2"
                  The "arxiv:" prefix and version suffix are stripped automatically.

    Returns:
        A formatted string with the paper's title, authors, abstract,
        and a direct link to the paper.
    """
    clean_id = arxiv_id.strip()
    if clean_id.lower().startswith("arxiv:"):
        clean_id = clean_id[6:]
    # Remove version suffix like "v2"
    if "v" in clean_id.split(".")[-1]:
        clean_id = clean_id.rsplit("v", 1)[0]

    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[clean_id])
        results = list(client.results(search))
    except Exception as e:
        return f"arXiv lookup failed: {str(e)}"

    if not results:
        return f"No arXiv paper found with ID: {arxiv_id}"

    paper = results[0]

    # Format authors -- show first 3, then "et al." if more exist.
    authors = [author.name for author in paper.authors]
    if len(authors) > 3:
        author_str = ", ".join(authors[:3]) + " et al."
    else:
        author_str = ", ".join(authors)

    # Truncate abstract if very long
    abstract = paper.summary.replace("\n", " ")
    if len(abstract) > 500:
        abstract = abstract[:500] + "..."

    return (
        f"Title    : {paper.title}\n"
        f"Authors  : {author_str}\n"
        f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
        f"URL      : {paper.entry_id}\n"
        f"Abstract : {abstract}"
    )
