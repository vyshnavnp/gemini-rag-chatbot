# tools/onco_tools.py — LangChain @tool functions for OncoBot.

import os
import base64

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATOR_MODEL = "gemini-3.1-flash-lite-preview"

# ---------------------------------------------------------------------------
# Session-level shared state for uploaded data (image / CSV).
# Tools read from these instead of receiving raw data through LLM tool args,
# because LLMs cannot reliably pass 100K+ char strings as arguments.
# ---------------------------------------------------------------------------
_session_image_b64: str | None = None
_session_genomic_csv: str | None = None


def set_session_image(b64: str | None) -> None:
    global _session_image_b64
    _session_image_b64 = b64


def set_session_csv(csv_str: str | None) -> None:
    global _session_genomic_csv
    _session_genomic_csv = csv_str


def clear_session_data() -> None:
    global _session_image_b64, _session_genomic_csv
    _session_image_b64 = None
    _session_genomic_csv = None

_embed_model = None
_COLLECTION_NAME = "langchain"


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embed_model


def _get_retriever():
    """Build a fresh ChromaDB retriever. Returns None if chroma_db missing."""
    if not os.path.exists(CHROMA_PATH):
        return None
    embeddings = _get_embed_model()
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )
    return vector_store.as_retriever(search_kwargs={"k": 6})


@tool
def oncology_rag_search(query: str) -> str:
    """
    Search the local oncology knowledge base (ChromaDB vector store) for
    information related to the given query.

    Use this tool whenever the user asks a factual question about:
    - cancer types, stages, or symptoms
    - treatment options (chemotherapy, immunotherapy, radiation, surgery)
    - drug names, mechanisms, or side effects
    - oncology research, clinical context, or patient support topics

    The knowledge base contains MedQuAD XML question-answer pairs and
    arXiv oncology research paper PDFs. Returns up to 6 relevant passages
    as a single combined string.

    Args:
        query: The search query in any language. The multilingual embedding
               model will match it against the English knowledge base.

    Returns:
        A string containing the concatenated relevant passages, each
        prefixed with its source filename.
    """
    retriever = _get_retriever()
    if retriever is None:
        return (
            "Knowledge base is not available. "
            "Run 'python updater.py' to build it first."
        )

    docs = retriever.invoke(query)

    if not docs:
        # Check whether the collection itself is empty (helps diagnose fresh deployments).
        try:
            import chromadb as _cdb
            _client = _cdb.PersistentClient(path=CHROMA_PATH)
            _count = _client.get_or_create_collection(_COLLECTION_NAME).count()
            if _count == 0:
                return (
                    "The local knowledge base is currently empty. "
                    "The background indexer runs every 30 minutes; "
                    "please wait a moment and try again, or run 'python updater.py' manually."
                )
        except Exception:
            pass
        return (
            f"No relevant oncology information found in the knowledge base for: '{query}'. "
            "Try rephrasing, or use fetch_pubmed_abstracts for a live literature search."
        )

    results = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        source_name = os.path.basename(source)
        results.append(f"[Source {i}: {source_name}]\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


@tool
def analyze_medical_image(question: str) -> str:
    """
    Analyze a medical image (scan, diagram, or pathology slide) using
    Google Gemini's vision capability.

    Use this tool when the user has uploaded an image and is asking a
    general question about it. The uploaded image is accessed automatically.

    Args:
        question: The user's question about the image.

    Returns:
        A string with Gemini's interpretation of the medical image in the
        context of the user's question.
    """
    if not _session_image_b64:
        return "No image uploaded. Please upload a medical image first."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY is not set. Cannot perform image analysis."

    vision_llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, api_key=api_key)

    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "You are an oncology image analysis assistant. "
                "Analyze this medical image in the context of cancer research. "
                f"Question: {question}"
            )
        },
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{_session_image_b64}"
        }
    ])

    try:
        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Image analysis failed: {str(e)}"


@tool
def generate_pathway_diagram(topic: str) -> str:
    """
    Generate a Graphviz DOT language diagram for a biological or clinical
    pathway related to oncology.

    Use this tool when the user asks to:
    - visualize a pathway (e.g., "show me the metastasis pathway")
    - draw a diagram (e.g., "diagram of T-cell activation")
    - map a process (e.g., "map chemotherapy side effects")
    - show a flowchart of any cancer-related biological process

    The output is a raw Graphviz DOT string. The Streamlit app renders
    this with st.graphviz_chart(). The diagram uses top-to-bottom layout
    (rankdir=TB) for readability.

    Args:
        topic: A plain English description of the pathway or process to
               visualize (e.g., "PD-1/PD-L1 checkpoint inhibition pathway").

    Returns:
        A Graphviz DOT format string, or an error message if generation
        fails. The string does NOT include the triple-backtick fences --
        just the raw DOT content starting with 'digraph'.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY is not set. Cannot generate diagram."

    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, api_key=api_key)

    diagram_prompt = f"""
You are a biomedical visualization expert.
Generate ONLY a valid Graphviz DOT language diagram for the following oncology topic.

Rules:
- Start with: digraph G {{ rankdir=TB;
- Use descriptive node labels in double quotes
- Use -> for directed edges
- End with }}
- Output ONLY the DOT code, no explanation, no markdown fences, no extra text

Topic: {topic}
"""

    try:
        response = llm.invoke(diagram_prompt)
        dot_content = response.content.strip()

        # Strip markdown code fences if the model added them anyway.
        if "```" in dot_content:
            lines = dot_content.split("\n")
            dot_content = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        return dot_content.strip()
    except Exception as e:
        return f"Diagram generation failed: {str(e)}"


_ONCOSCANBC_MODEL = None
_ONCOSCANBC_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "oncoscan_bc.pth")
_ONCOSCANBC_CLASSES = ["benign", "malignant", "normal"]


def _load_oncoscanbc_model():
    """Lazily load the OncoScanBC MobileNetV2 model."""
    global _ONCOSCANBC_MODEL
    if _ONCOSCANBC_MODEL is not None:
        return _ONCOSCANBC_MODEL

    if not os.path.exists(_ONCOSCANBC_MODEL_PATH):
        return None

    import torch
    from torchvision import models

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.last_channel, len(_ONCOSCANBC_CLASSES)
    )
    model.load_state_dict(
        torch.load(_ONCOSCANBC_MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    _ONCOSCANBC_MODEL = model
    return _ONCOSCANBC_MODEL


def _preprocess_image_b64(image_b64: str):
    """Decode a base64 image and apply standard ImageNet preprocessing."""
    import io
    from PIL import Image
    from torchvision import transforms

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


@tool
def classify_breast_ultrasound(clinical_context: str = "Breast ultrasound classification") -> str:
    """
    Classify a breast ultrasound image as benign, malignant, or normal
    using the OncoScanBC MobileNetV2 deep learning model.

    Use this tool ONLY when the user has uploaded a breast ultrasound
    image and wants a classification or diagnosis prediction.
    The uploaded image is accessed automatically — do NOT pass image data.

    This is a trained CNN classifier — it returns a predicted class and
    confidence score, NOT a free-text description. For general image
    analysis or non-ultrasound images, use analyze_medical_image instead.

    Args:
        clinical_context: Optional clinical context or notes about the scan.

    Returns:
        A string reporting the predicted class and confidence percentage.
    """
    if not _session_image_b64:
        return "No image uploaded. Please upload a breast ultrasound image first."

    model = _load_oncoscanbc_model()
    if model is None:
        return (
            "OncoScanBC model is not available. "
            "Place the trained weights at: models/oncoscan_bc.pth"
        )

    import torch

    tensor = _preprocess_image_b64(_session_image_b64)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    label = _ONCOSCANBC_CLASSES[predicted.item()]
    conf_pct = confidence.item() * 100

    return (
        f"OncoScanBC Prediction: {label.upper()}\n"
        f"Confidence: {conf_pct:.1f}%\n\n"
        f"Class probabilities:\n"
        + "\n".join(
            f"  - {cls}: {probabilities[i].item()*100:.1f}%"
            for i, cls in enumerate(_ONCOSCANBC_CLASSES)
        )
        + "\n\nNote: This is an AI prediction for research purposes only. "
        "Clinical diagnosis requires histopathological confirmation."
    )


_ONCOSCANSKIN_MODEL = None
_ONCOSCANSKIN_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "oncoscan_skin.pth")
_ONCOSCANSKIN_CLASSES = [
    "actinic keratoses", "basal cell carcinoma", "benign keratosis-like lesions",
    "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions",
]


def _load_oncoscanskin_model():
    """Lazily load the OncoScanSkin MobileNetV2 model."""
    global _ONCOSCANSKIN_MODEL
    if _ONCOSCANSKIN_MODEL is not None:
        return _ONCOSCANSKIN_MODEL

    if not os.path.exists(_ONCOSCANSKIN_MODEL_PATH):
        return None

    import torch
    from torchvision import models

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.last_channel, len(_ONCOSCANSKIN_CLASSES)
    )
    model.load_state_dict(
        torch.load(_ONCOSCANSKIN_MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    _ONCOSCANSKIN_MODEL = model
    return _ONCOSCANSKIN_MODEL


@tool
def classify_skin_lesion(clinical_context: str = "Skin lesion classification") -> str:
    """
    Classify a cutaneous (skin) lesion from a dermoscopy or microscopy
    image using the OncoScanSkin MobileNetV2 deep learning model.

    Use this tool ONLY when the user has uploaded a skin lesion image
    and wants a classification or melanoma screening prediction.
    The uploaded image is accessed automatically — do NOT pass image data.

    Supported classes: actinic keratoses, basal cell carcinoma,
    benign keratosis-like lesions, dermatofibroma, melanoma,
    melanocytic nevi, vascular lesions.

    Args:
        clinical_context: Optional clinical context or notes about the lesion.

    Returns:
        A string reporting the predicted class and confidence percentage.
    """
    if not _session_image_b64:
        return "No image uploaded. Please upload a skin lesion image first."

    model = _load_oncoscanskin_model()
    if model is None:
        return (
            "OncoScanSkin model is not available. "
            "Place the trained weights at: models/oncoscan_skin.pth"
        )

    import torch

    tensor = _preprocess_image_b64(_session_image_b64)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    label = _ONCOSCANSKIN_CLASSES[predicted.item()]
    conf_pct = confidence.item() * 100

    top3 = torch.topk(probabilities, k=min(3, len(_ONCOSCANSKIN_CLASSES)))
    top3_lines = [
        f"  - {_ONCOSCANSKIN_CLASSES[idx.item()]}: {prob.item()*100:.1f}%"
        for prob, idx in zip(top3.values, top3.indices)
    ]

    return (
        f"OncoScanSkin Prediction: {label.upper()}\n"
        f"Confidence: {conf_pct:.1f}%\n\n"
        f"Top 3 predictions:\n"
        + "\n".join(top3_lines)
        + "\n\nNote: This is an AI prediction for research purposes only. "
        "Clinical diagnosis requires histopathological confirmation."
    )


_ONCOTYPEBC_MODEL = None
_ONCOTYPEBC_SCALER = None
_ONCOTYPEBC_LABEL_ENCODER = None
_ONCOTYPEBC_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "oncotype_bc.pth")
_ONCOTYPEBC_SCALER_PATH = os.path.join(_PROJECT_ROOT, "models", "scaler.pkl")
_ONCOTYPEBC_ENCODER_PATH = os.path.join(_PROJECT_ROOT, "models", "label_ecoder.pkl")

# Readable names for TCGA cancer type codes.
_ONCOTYPEBC_CANCER_NAMES = {
    "BRCA": "Breast Invasive Carcinoma",
    "KIRC": "Kidney Renal Clear Cell Carcinoma",
    "LUAD": "Lung Adenocarcinoma",
    "PRAD": "Prostate Adenocarcinoma",
    "COAD": "Colon Adenocarcinoma",
}


def _load_oncotypebc_model():
    """Lazily load OncoTypeBC model, scaler, and label encoder."""
    global _ONCOTYPEBC_MODEL, _ONCOTYPEBC_SCALER, _ONCOTYPEBC_LABEL_ENCODER
    if _ONCOTYPEBC_MODEL is not None:
        return _ONCOTYPEBC_MODEL, _ONCOTYPEBC_SCALER, _ONCOTYPEBC_LABEL_ENCODER

    if not os.path.exists(_ONCOTYPEBC_MODEL_PATH):
        return None, None, None

    import sys
    import pickle
    import torch
    import torch.nn as nn

    # Model class must be available for torch.load (pickle expects __main__.OncoTypeBCModel).
    class OncoTypeBCModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 5),
            )

        def forward(self, x):
            return self.network(x)

    sys.modules["__main__"].OncoTypeBCModel = OncoTypeBCModel

    _ONCOTYPEBC_MODEL = torch.load(
        _ONCOTYPEBC_MODEL_PATH, map_location="cpu", weights_only=False
    )
    _ONCOTYPEBC_MODEL.eval()

    if os.path.exists(_ONCOTYPEBC_SCALER_PATH):
        with open(_ONCOTYPEBC_SCALER_PATH, "rb") as f:
            _ONCOTYPEBC_SCALER = pickle.load(f)

    if os.path.exists(_ONCOTYPEBC_ENCODER_PATH):
        with open(_ONCOTYPEBC_ENCODER_PATH, "rb") as f:
            _ONCOTYPEBC_LABEL_ENCODER = pickle.load(f)

    return _ONCOTYPEBC_MODEL, _ONCOTYPEBC_SCALER, _ONCOTYPEBC_LABEL_ENCODER


@tool
def classify_cancer_type(analysis_note: str = "Cancer type classification") -> str:
    """
    Classify cancer type from gene expression data using the OncoTypeBC
    deep learning model trained on TCGA RNA-Seq data.

    Use this tool ONLY when the user has uploaded a gene expression CSV
    file and wants a cancer type prediction.
    The uploaded CSV is accessed automatically — do NOT pass CSV data.

    The five predicted cancer types are:
    - BRCA (Breast Invasive Carcinoma)
    - KIRC (Kidney Renal Clear Cell Carcinoma)
    - LUAD (Lung Adenocarcinoma)
    - PRAD (Prostate Adenocarcinoma)
    - COAD (Colon Adenocarcinoma)

    Args:
        analysis_note: Optional note about the sample or analysis request.

    Returns:
        A string reporting the predicted cancer type and confidence.
    """
    if not _session_genomic_csv:
        return "No gene expression CSV uploaded. Please upload a CSV file first."

    model, scaler, label_encoder = _load_oncotypebc_model()
    if model is None:
        return (
            "OncoTypeBC model is not available. "
            "Place the trained model at: models/oncotype_bc.pth"
        )

    import io
    import csv
    import torch

    reader = csv.reader(io.StringIO(_session_genomic_csv))
    rows = list(reader)
    if len(rows) < 2:
        return "Invalid CSV: need at least a header row and one data row."

    headers = rows[0]
    try:
        features = [float(v) for v in rows[1]]
    except ValueError:
        return "Invalid CSV: all data values must be numeric."

    if scaler is not None:
        features_scaled = scaler.transform([features])
        tensor = torch.tensor(features_scaled, dtype=torch.float32)
    else:
        tensor = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    if label_encoder is not None:
        label_code = label_encoder.inverse_transform([predicted.item()])[0]
        label = _ONCOTYPEBC_CANCER_NAMES.get(label_code, label_code)
    else:
        label = f"Class {predicted.item()}"

    prob_lines = []
    for i in range(len(probabilities)):
        if label_encoder is not None:
            code = label_encoder.inverse_transform([i])[0]
            name = _ONCOTYPEBC_CANCER_NAMES.get(code, code)
        else:
            name = f"Class {i}"
        prob_lines.append(f"  - {name}: {probabilities[i].item()*100:.1f}%")

    conf_pct = confidence.item() * 100

    return (
        f"OncoTypeBC Prediction: {label}\n"
        f"Confidence: {conf_pct:.1f}%\n\n"
        f"Cancer type probabilities:\n"
        + "\n".join(prob_lines)
        + f"\n\nInput features: {len(headers)} columns detected."
        + "\n\nNote: This is an AI prediction for research purposes only. "
        "Clinical diagnosis requires histopathological and genomic confirmation."
    )

