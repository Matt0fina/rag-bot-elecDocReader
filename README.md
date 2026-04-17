# ⚡ ElecDocReader: Multimodal RAG for Electrical Datasheet Extraction

<img width="2513" height="1305" alt="image" src="https://github.com/user-attachments/assets/99d4aab2-d58b-4c0d-a06c-76917429771a" />
<video src="https://github.com/user-attachments/assets/cc7b0350-fe41-4044-baae-94096979ec80" width="100%" autoplay loop muted></video>

## 📖 Project Objective
In Electronic Design Automation (EDA) and Hardware/Test Engineering, extracting exact parameters from dense electrical component datasheets is a massive, time-consuming bottleneck. Engineers rely on highly specific data—such as dynamic AC characteristics, test conditions, and absolute maximum ratings—to build reliable simulation models (like SPICE). 

While standard Retrieval-Augmented Generation (RAG) pipelines hold promise, standard text-parsing algorithms algorithmically "flatten" complex multi-column tables. This destroys the spatial relationships between parameters and their conditions, leading to dangerous LLM hallucinations. 

**ElecDocReader** solves this spatial reasoning deficit. By implementing a **multimodal RAG architecture** featuring vision-based layout parsing, strict JSON-schema enforcement, and programmatic context isolation, this system preserves the visual hierarchy of datasheets to achieve highly deterministic parameter extraction.

---

## 🚀 Key Findings & Results
The system was rigorously evaluated against a baseline text-only RAG architecture using a "Golden Dataset" of 30 stratified queries targeting components for an **Optical Heart Rate Monitor (PPG)** circuit.

* **Baseline (PyMuPDF + Llama-3.1-8B):** Achieved **56.67%** accuracy, heavily struggling with layout fragmentation and semantic vector overlap.
* **Proposed Architecture (UnstructuredAPI + Llama-3.3-70B):** Achieved **86.67%** accuracy. 

A Two-Proportion Z-Test confirmed this **30.0% absolute improvement is statistically significant ($p < 0.01$)**.

### 📊 Tiered Complexity Analysis
1. **Tier 1 (Surface Extraction):** Solved the text fragmentation issue where simple parsers detach parameter names from their values.
2. **Tier 2 (Condition Mapping):** Maintained row-column integrity in dense tables, successfully linking parameters (e.g., $h_{FE}$) to nested test conditions (e.g., $I_C = 150mA$, $V_{CE} = 10V$).
3. **Tier 3 (Safety Thresholds):** Mitigated the **"Vector Overlap"** problem, ensuring the retriever successfully differentiates between "Nominal Operating Ranges" and critical "Absolute Maximum Ratings."

<img width="2454" height="1509" alt="image" src="https://github.com/user-attachments/assets/3738a73e-d800-49f2-a064-def131a9202b" />
---

## 🛠️ Architecture & Tech Stack

### Frontend
* **Streamlit:** A dual-column engineering dashboard featuring a conversational chat interface on the left and a live, structured data-extraction table (Pandas DataFrame) on the right.

### Backend
* **FastAPI:** High-performance asynchronous API handling document processing and LLM inference routing.
* **LangChain:** Orchestration framework connecting the document loaders, vector store, and LLMs.
* **ChromaDB:** Local SQLite-based vector database. *(Note: Engineered with custom programmatic directory flushing during document uploads to completely prevent "phantom context" and cross-component contamination).*

### Data Processing & AI
* **UnstructuredAPI (`hi_res` strategy):** Vision-based layout parser that identifies bounding boxes to preserve tabular topologies.
* **Embeddings:** Google Gemini (`models/gemini-embedding-001`) / BAAI (`bge-large-en-v1.5`).
* **LLM Inference:** Groq-hosted Llama-3.3-70B-Versatile for high-speed, high-reasoning JSON generation.

---

## ⚙️ Installation & Usage Instructions

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You will also need API keys for Groq, Google Gemini (for embeddings), and Unstructured (if using their cloud API).

### 2. Clone the Repository
```bash
git clone [https://github.com/YourUsername/rag-bot-elecDocReader.git](https://github.com/YourUsername/rag-bot-elecDocReader.git)
cd rag-bot-elecDocReader
```

### Environment Setup
Create a virtual environement and install the required dependencies:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**Install backend dependencies**
```bash
cd client
pip install -r requirements.txt
```
**Install frontend dependencies**
```bash
cd ../server
pip install -r requirements.txt
```

### Configuration with API Keys
```bash
Groq_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
UNSTRUCTURED_API_KEY=your_unstructured_api_key_here
```

## ▶️ Running the Application
**Start EDR backend:**
```bash
# Terminal 1
cd server
uvicorn main:app --reload
```

**Start EDR frontend:**
```bash
# Terminal 2
cd  client
streamlit run app.py
```

