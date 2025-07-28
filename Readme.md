# 🧠 Adobe Hackathon Round 1B — Persona-Driven Document Intelligence

This solution addresses Round 1B of the Adobe India Hackathon by extracting and ranking the most relevant sections from multiple PDF documents based on a given **persona** and **job-to-be-done**.

It uses **semantic similarity**, **heading-based extraction**, and **dynamic topic modeling** to identify high-value information tailored to user intent.

---

## 🚀 Key Features

- 📄 **Intelligent PDF Section Extraction**  
  Uses `PyMuPDF` to extract clean, non-tabular content under bold, high-font headings.

- 🧠 **Semantic Relevance Scoring**  
  Embeds section text and queries using `SentenceTransformer` (locally hosted), and ranks based on cosine similarity.

- 🧵 **Parallel Processing**  
  Sections are extracted concurrently using `asyncio` and `ThreadPoolExecutor`, enabling efficient processing across large collections.

- 🎯 **Dynamic Topic Categorization**  
  Titles are auto-clustered into top categories based on keyword frequency, matched to persona-job keywords.

- 📁 **Structured Output**  
  Generates an easy-to-parse `challenge1b_output.json` file with ranked sections, summaries, and metadata.

---

## 🖥️ How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
Then run:
```bash
python main.py
This processes Collection 1, Collection 2, and Collection 3, generating challenge1b_output.json in each folder.

🐳 Docker Instructions

Make sure you're in the project folder containing the `Dockerfile`, `main.py`, and `model/` directory.

```bash
docker build --platform linux/amd64 -t mysolution:round1b .

docker run --rm --network none -v "${PWD}:/app" mysolution:round1b

📤 Output Format
Each collection folder will contain:

{
  "metadata": {
    "input_documents": [...],
    "persona": "...",
    "job_to_be_done": "...",
    "processing_timestamp": "..."
  },
  "extracted_sections": [
    {
      "document": "...",
      "section_title": "...",
      "importance_rank": 1,
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "...",
      "refined_text": "...",
      "page_number": 5
    }
  ]
}





