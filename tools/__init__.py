# tools/__init__.py
# Makes the tools directory a Python package.
# Exposes all tool functions from a single import point.

from tools.onco_tools import (
    oncology_rag_search,
    analyze_medical_image,
    classify_breast_ultrasound,
    classify_skin_lesion,
    classify_cancer_type,
    set_session_image,
    set_session_csv,
    clear_session_data,
)

from tools.external_tools import (
    search_clinical_trials,
    fetch_pubmed_abstracts,
    summarize_arxiv_paper,
)

__all__ = [
    "oncology_rag_search",
    "analyze_medical_image",
    "classify_breast_ultrasound",
    "classify_skin_lesion",
    "classify_cancer_type",
    "search_clinical_trials",
    "fetch_pubmed_abstracts",
    "summarize_arxiv_paper",
]
