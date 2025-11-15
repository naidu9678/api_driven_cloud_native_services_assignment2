# BITS Pilani Smart AI Assistant

Multimodal, API-driven assistant built for CC ZG506 – Assignment II. The project targets the **education domain** and unifies five AI sub-tasks across NLP and Computer Vision using Google Gemini, LangChain, FAISS, YOLOv8, Tesseract OCR, and a fine-tuned model fallback hosted on Hugging Face.

---

## 1. Solution Overview
- **Domain focus**: Academic enablement for faculty and students (syllabus Q&A, grading, study aids, diagram understanding).
- **Categories covered**:  
  - *Natural Language Processing*: Retrieval-Augmented Generation (RAG), summarization, adaptive question generation, answer grading, fallback chit-chat.  
  - *Computer Vision*: Script/diagram ingestion via OCR, YOLO-based diagram detection and explanation pipeline.
- **Unified objective**: Provide an interactive hub where the same PDFs fuel Q&A, grading, diagram explanation, summaries, and assessments—reducing manual effort for instructors.
- **APIs & Models**: Google Generative AI (Gemini 2.5 Flash + text-embedding-004), Hugging Face (fine-tuned Flan-T5), Ultralytics YOLOv8, LangChain ecosystem.

---

## 2. Architecture at a Glance
1. **Document ingestion & RAG core**  
   - PDFs uploaded via Streamlit sidebar.  
   - `PyPDFLoader → RecursiveCharacterTextSplitter → FAISS` with Google embeddings.  
   - Vector store reused across modes through session state.
2. **Conversational layer**  
   - `ChatGoogleGenerativeAI` answers queries using LangChain retrieval + prompt orchestration.  
   - Automatic fallback to fine-tuned Flan-T5 (`naidu9678/flan_t5_finetuned_education`) if context is weak or Gemini errors out.
3. **CV + OCR layer**  
   - `YOLOv8n` weights (`yolov8n.pt`) pre-bundled for detecting diagrams.  
   - `pytesseract` extracts text from PDFs/images for grading & explanation flows.  
   - CV outputs feed LLM prompts for rubric comparison or diagram narration.
4. **Educational tooling**  
   - Summarization & question generation reuse vector store chunks or ad-hoc uploads via shared `load_docs_from_file()` helper.  
   - Configurable difficulty, counts, and context routing.
5. **LLMOps signals**  
   - Extensive logging via Python `logging`.  
   - Training notebook captures `train_runtime`, `train_samples_per_second`, `train_steps_per_second`, `total_flos`, `train_loss` for compliance with assignment metrics requirement.  
   - Streamlit status messages surface processing state to end users.

---

### HD Architecture Diagram
- Mermaid source: `docs/architecture_diagram.mmd`  

---

## 3. Key Functionalities

| # | Capability | Description & File References |
|---|------------|--------------------------------|
| 1 | **Document Q&A (Text RAG)** | Upload/merge PDFs, build FAISS index, and chat via `Q&A (Text RAG)` mode in `chatbot.py`. Gemini 2.5 Flash grounds answers strictly in retrieved chunks with refusal policy if evidence missing. |
| 2 | **Automated Assignment Grading (CV + LLM)** | `process_image_for_grading()` OCRs student scripts & answer keys (PDF/image) and prompts Gemini to score out of 10 with rubric-style feedback. Handles missing OCR, empty text, and input validation. |
| 3 | **Diagram/Chart Explanation (CV + RAG)** | `process_image_for_explanation()` detects diagrams (YOLO) and extracts labels via OCR, then retrieves supplemental theory from PDFs to narrate context-aware explanations. |
| 4 | **Educational Tools – Summaries & Adaptive Question Generation** | `summarize_content()` and `generate_questions()` repurpose either uploaded documents or existing FAISS retriever. Difficulty & question count are user-configurable; output formatted for quick study. |
| 5 | **Fallback Conversational Model** | `load_flan_t5_chatbot()` loads fine-tuned Flan‑T5 from Hugging Face and auto-engages when Gemini lacks context or raises errors, ensuring uninterrupted answers. |
| 6 | **Vector-store CLI Utility** | `pdf_loader.py` can pre-build/refresh FAISS indices offline (e.g., `python pdf_loader.py`). Keeps app startup snappy for large syllabi. |
| 7 | **Fine-tuning Pipeline** | `training__model_assignment2_updated.ipynb` documents dataset prep, Flan-T5 training, Hub push, and inference sanity checks. |

---

## 4. Repository Layout
```
api_driven_cloud_native_services_assignment2/
├── chatbot.py                     # Streamlit app (all modes)
├── pdf_loader.py                  # CLI ingestion to FAISS
├── hug.py                         # Minimal script to test HF model
├── student_qa_dataset.json        # Custom QA dataset used for fine-tuning
├── training__model_assignment2_updated.ipynb
├── faiss_index/                   # Persisted FAISS index (binary + metadata)
├── pdfs/                          # Sample syllabus, rubrics, answer keys
├── yolov8n.pt                     # YOLO weights for diagram detection
├── requirements.txt
└── README.md
```

---

## 5. Prerequisites
- **Python** ≥ 3.10 (tested on macOS 15 / Linux with virtualenv).
- **System dependencies**  
  - [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html). Update `pytesseract.pytesseract.tesseract_cmd` in `chatbot.py` to your OS path (default Windows path is prefilled).  
  - `libGL` packages for OpenCV if running on Linux servers.
- **APIs / Credentials**  
  - `GOOGLE_API_KEY` with access to Gemini 2.5 Flash + `text-embedding-004`.  
  - Hugging Face account (only needed to re-train or pull private checkpoints).
- **GPU (optional)** for faster Flan-T5 inference/training; automatically falls back to CPU if unavailable.

---

## 6. Setup & Installation
```bash
# 1. Clone
git clone <repo_url>

# 2. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env   # if you keep a template
export GOOGLE_API_KEY="XXXXXXXX"  # macOS/Linux
set GOOGLE_API_KEY=XXXXXXXX       # Windows PowerShell

# 5. (Optional) Pre-build FAISS index from bundled PDFs
python pdf_loader.py

# 6. Launch Streamlit app
streamlit run chatbot.py
```

---

## 7. Using the Application

1. **Ingest PDFs**  
   - Use sidebar “Document Management” uploader.  
   - Click **Process & Update Vector Store** to merge into FAISS (`faiss_index/`).  
   - Status toasts report progress; logs saved to console.

2. **Select Operation Mode**
   - `Q&A (Text RAG)`: Chat about uploaded syllabus/policies. Streaming chat UI keeps history in `st.session_state.chat_history`.  
   - `Assignment Grading (CV+LLM)`: Upload student script + answer key (PDF/image) and supply original question. App OCRs both, compares, and responds with score + structured feedback.  
   - `Educational Tools (NLP)`:  
     - *Summarization*: Provide topic to query current vector store or upload fresh PDF/TXT for one-off summary.  
     - *Adaptive Question Generation*: Similar flow with difficulty + quantity controls.  
   - `Diagram/Chart Explanation (CV+RAG)`: Upload a JPEG/PNG diagram; YOLO detects regions, OCR extracts labels, and Gemini explains using retrieved syllabus context.  

3. **Fallback Behaviour**  
   - If Gemini cannot ground answers (“context missing”) or the API fails, the Flan‑T5 pipeline answers and the UI tags the response with `(Fallback)` so reviewers know the provenance.

4. **Session controls**  
   - App leverages Streamlit reruns to trigger heavy workflows (grading, summarization, QGen, diagram explanation) without freezing the chat pane.

---

## 8. Datasets, Fine-tuning, and Model Registry

| Asset | Purpose |
|-------|---------|
| `student_qa_dataset.json` | Synthetic BITS cloud curriculum QA pairs used to fine-tune Flan‑T5 for fallback knowledge. |
| `training__model_assignment2_updated.ipynb` | Colab-ready notebook covering installation, preprocessing, training, evaluation, and Hugging Face push. |
| `naidu9678/flan_t5_finetuned_education` | Public HF repository storing the fine-tuned weights referenced by `chatbot.py` and `hug.py`. |

**Reproducing fine-tuning**
1. Upload dataset to Colab along with `yolov8n.pt` if needed for tests.  
2. Execute cells sequentially (install deps → load dataset → tokenize → train).  
3. Notebook logs key metrics (`train_runtime`, `train_samples_per_second`, `train_steps_per_second`, `total_flos`, `train_loss`) satisfying the “≥5 metrics” requirement.  
4. Run final cells to push to Hugging Face Hub (requires `notebook_login()` write token).  
5. Update `fine_tuned_model_name` in `chatbot.py` if you publish under a different namespace.

---

## 9. LLMOps Considerations & Metrics
- **Observability**: Structured logging (INFO/WARNING/ERROR) throughout ingestion, triggers, and fallbacks. Check terminal logs during Streamlit execution for detailed traces.  
- **Metrics captured** (Training notebook):  
  - `train_runtime`  
  - `train_samples_per_second`  
  - `train_steps_per_second`  
  - `total_flos`  
  - `train_loss`  
- **Quality controls**:  
  - RAG context limit `k=5` to avoid hallucination.  
  - System prompts instruct Gemini to decline if evidence absent.  
  - Fallback responses clearly prefixed to distinguish from grounded answers.  
  - Temporary directories cleaned post-processing to prevent stale data or contamination.

---

## 10. Troubleshooting
- **“Tesseract not found”**: Install Tesseract and update the `pytesseract.pytesseract.tesseract_cmd` constant near the top of `chatbot.py` to `/usr/local/bin/tesseract` (macOS) or corresponding Linux path.  
- **FAISS load errors**: Delete `faiss_index/` and rebuild via sidebar upload or `python pdf_loader.py`. Ensure Google API quota allows embedding generation.  
- **Gemini quota/timeouts**: The app sets a 300 s timeout. If calls still fail, verify billing/project quota; fallback model will continue to answer but without document grounding.  
- **YOLO model missing**: Redownload `yolov8n.pt` and keep it at repo root; `ultralytics` expects local weights.  
- **Large PDFs**: Increase `chunk_size` / `chunk_overlap` in `chatbot.py` if needed, or pre-chunk content offline.

---

