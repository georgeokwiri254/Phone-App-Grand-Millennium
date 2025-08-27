Perfect 👍 Since you **already have a Streamlit + SQLite project**, I’ll rewrite the README to clearly explain that your new addition is **local LLM + RAG + Vector DB**.

Here’s a **refined README.md** plan:

---

# 🏨 Revenue Analytics Assistant (Local LLM + SQLite + Streamlit)

This project extends a **Streamlit + SQLite revenue analytics app** with a **local LLM (llama.cpp)** and **RAG (Retrieval-Augmented Generation)**, enabling natural language questions on hotel revenue data.

Everything runs **locally on CPU** — no external APIs, fully open-source, suitable for an **8GB RAM PC**.

---

## 🚀 Key Features

* 🔍 **Ask questions in natural language** (e.g., *“What was total revenue in July 2025?”*)
* 🗄️ **SQLite integration** for live querying of revenue data
* 📚 **RAG pipeline** with **ChromaDB** for contextual retrieval from historical data
* 🧠 **Local LLM (llama-cpp)** to generate SQL and natural answers
* 🎛️ **Streamlit UI** for interactive queries, results, and visualizations
* 🔒 **Privacy-first**: No internet or external APIs required

---

## 📦 Installation

```bash
git clone <this-repo>
cd revenue-assistant
pip install streamlit chromadb sentence-transformers llama-cpp-python pydantic==1.10.13
```

---

## 🧠 Models

Download a **quantized `.gguf` model** (2–4GB for CPU use):

* [Mistral-7B-Instruct Q4](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
* [Phi-3-mini Q4](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
* [Qwen-2.5 1.5B Q4](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)

Update your model path in the Streamlit sidebar.

---

## 📂 Database Setup

Place your **SQLite database** (e.g., `revenue.db`) in the project directory.

Example tables:

* `reservations` — guest-level bookings (dates, ADR, length of stay)
* `departments` — departmental revenue (Rooms, F\&B, Spa)
* `daily_revenue` — daily totals
* `monthly_summary` — monthly KPIs (ADR, RevPAR, occupancy, total revenue)

---

## 🛠️ Workflow

1. **Indexing**: Convert SQLite rows → text chunks → store embeddings in **ChromaDB**.
2. **Retrieval**: On each question, retrieve top-k relevant chunks.
3. **SQL Agent**: LLM generates a valid SQL query → executes on SQLite.
4. **Answer Generation**: LLM combines schema + retrieved context + SQL results → produces a narrative answer.

---

## 🖥️ Streamlit UI

* **Sidebar**:

  * Select SQLite DB & model path
  * Adjust LLM parameters (temperature, max tokens)
  * Build / refresh vector index
  * Toggle SQL Agent

* **Main Panel**:

  * Enter natural language questions
  * See generated SQL queries
  * View retrieved context chunks
  * Inspect SQL results (table + charts)
  * Read final assistant answer

---

## ⚡ Performance Notes

* Use **Q4 quantized models** for smooth performance on 8GB RAM.
* Limit table ingestion (e.g., 10k rows) when indexing to vector DB.
* Tune chunk size (20–50 rows) for better retrieval.
* Keep answer length (`max_tokens`) modest (256–512).

---

## 📊 Example Queries

* *“What was Rooms revenue in August 2025?”*
* *“Show monthly ADR trend for 2024.”*
* *“Which department had the strongest YoY growth?”*
* *“What was RevPAR in July vs June 2025?”*

---

## 🔒 Why Local?

* Runs **offline**, no API costs.
* Sensitive hotel data never leaves your machine.
* Perfect for hotels needing **data privacy + AI insights**.

---

👉 Would you like me to **add a "Future Roadmap" section** (dashboards, forecasting, multi-property support), or keep this README focused only on the current functionality?
