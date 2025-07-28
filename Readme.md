
# Adobe India Hackathon 2025: Round 1B — Persona-Driven Document Intelligence

## 🧠 Overview

This project is the solution for **Round 1B** of the Adobe India Hackathon 2025, under the theme _"Connect What Matters — For the User Who Matters"_. The goal is to build a system that analyzes a set of PDF documents and identifies the most relevant sections based on a specific **persona** and their **job-to-be-done**.

The output includes:
- The most relevant **sections** with ranking and page numbers.
- A refined **sub-section analysis** summarizing the extracted content.

---

## 🚀 Features

- **Semantic Search**: Uses a locally loaded SentenceTransformer model to match section content with the user's intent.
- **Persona Relevance**: Dynamically generates an "intent" query by combining persona and job descriptions.
- **Diverse Selection**: Ensures section diversity across documents and topics.
- **Dynamic Topic Discovery**: Identifies categories using frequent keywords in section titles.
- **Fast, Parallel Extraction**: Utilizes `asyncio` with thread pool to process multiple PDFs efficiently.
- **Offline & Lightweight**: No internet usage, model is local, and fully CPU-compliant.

---

## 📦 Technologies Used

- **Python 3.8+**
- **Libraries**:
  - `PyMuPDF` (`fitz`) — PDF reading and text layout
  - `sentence-transformers` — For semantic similarity (runs locally)
  - `torch` — Backend for sentence-transformers
  - `asyncio`, `concurrent.futures`, `re`, `logging` — Async and parsing logic

---

## 📂 Input Format

Each `Collection X` folder contains:

- `PDFs/`: Folder with 3–10 related PDF files.
- `challenge1b_input.json`:
```json
{
  "persona": { "role": "Investment Analyst" },
  "job_to_be_done": { "task": "Analyze financial trends" },
  "documents": [
    { "filename": "report1.pdf" },
    { "filename": "report2.pdf" }
  ]
}
```

---

## ✅ Output Format

The script generates `challenge1b_output.json` in the same folder, structured like:

```json
{
  "metadata": {
    "input_documents": ["report1.pdf", "report2.pdf"],
    "persona": "Investment Analyst",
    "job_to_be_done": "Analyze financial trends",
    "processing_timestamp": "2025-07-28T10:00:00"
  },
  "extracted_sections": [
    {
      "document": "report1.pdf",
      "section_title": "Revenue Growth",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "report1.pdf",
      "refined_text": "Revenue increased by 20%...",
      "page_number": 3
    }
  ]
}
```

---

## 🐳 Docker Recommendations

To meet the hackathon constraints:

- CPU-only
- Model size ≤ 1GB (your SentenceTransformer model must be within this limit)
- No internet access allowed during execution
- Processing time ≤ 60 seconds for 3–5 documents


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AshNicolus/challenge1b.git
cd challenge1b
```

### 2. Build the Docker image

```bash
docker build --platform linux/amd64 -t mysolution:round1b .
```

---

## Usage

### 3. Run the Docker container

```bash
docker run --rm --network none -v "${PWD}:/app" mysolution:round1b
```

---

## 🏃‍♂️ How to Run(on local Environment)

The script will automatically process each folder named `Collection 1`, `Collection 2`, etc.

```bash
python main.py
```

Make sure each folder includes a valid `challenge1b_input.json` and a `PDFs/` subdirectory.

---

## ⚙️ Configuration

Key parameters in `main.py`:

| Name              | Description                                      |
|-------------------|--------------------------------------------------|
| `EMBEDDING_MODEL` | Local path to sentence-transformer model         |
| `TOP_K`           | Number of top sections to select (default: 12)   |
| `MAX_PER_DOC`     | Max sections to include per document (default: 2)|
| `NUM_THREADS`     | Threads for parallel PDF processing (default: 2) |

---

## 👨‍💻 Contributors

- **Yash Nema** — Lead Developer
- **Saloni Kumari** — Developer & Testing
- **Jennessa** — Researcher

---

## 📄 License

This solution is proprietary and submitted exclusively for the Adobe India Hackathon 2025. 

