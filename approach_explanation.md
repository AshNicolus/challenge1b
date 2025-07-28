# 🧠 Approach Explanation – Challenge 1B: Persona-Driven Document Intelligence

## 👤 Problem Overview

The goal of this challenge is to build an intelligent system that can scan through a set of PDF documents and extract the most relevant sections based on a given **persona** (such as a traveler, researcher, or food lover) and a specific **job-to-be-done**.

We wanted to create a solution that works across various document formats and languages, captures the structure of content, and ranks it in a way that truly reflects what the persona is looking for.

---

## 🛠️ How We Approached It

### 1. Understanding the Document Structure

Our initial step involves structural analysis of the PDF, grounded in Document Layout Analysis (DLA) principles. Inspired by O'Gorman (1993), we use visual cues to interpret document geometry. Using **PyMuPDF**, we extract:

- Font sizes  
- Bold/italic styles  
- Text position on the page  

This helps reliably detect titles and headings — even when the PDFs lack a formal structure. Each heading marks the start of a new section, and we group paragraphs that follow under it.

---

### 2. Speeding Things Up with Parallel Processing

Given that multiple PDFs may be processed at once, we applied parallelization using Python’s `asyncio` and `ThreadPoolExecutor`.

This allows us to:
- Extract structures from multiple documents in parallel  
- Drastically reduce runtime without compromising accuracy  

---

### 3. Understanding Relevance Using AI

After extracting all sections, we determine which are relevant to the persona’s job using semantic similarity.

We utilized **Sentence-BERT (Reimers & Gurevych, 2019)** — specifically the `all-MiniLM-L6-v2` model — to embed both:
- Section content  
- Persona’s job-to-be-done  

These are transformed into vector representations, and **cosine similarity** is computed. This ensures we're not just matching keywords, but actual semantic meaning — allowing relevance to be determined even if wording differs.

---

### 4. Organizing with Dynamic Categories

Instead of predefined topics, we:
- Extract common keywords from section headings  
- Cross-reference with keywords from the persona’s description  

This creates **dynamic categories** which are ranked based on overlap with the user's intent. It makes the system flexible and adaptive to various document types and personas.

---

### 5. Picking the Best Content

Finally, we select the **top-ranked** sections based on:
- Semantic similarity scores  
- Document diversity (avoiding over-representation of a single file)  
- Thematic variety (using the dynamic categories)

Each selected section is also **summarized** to aid fast comprehension.

---

## ✅ Summary

Our system integrates:
- Document structure detection  
- Semantic understanding  
- Performance optimization  

It mimics how a human would skim and search for the most relevant parts of a document collection — but delivers results in seconds.

---

## 📚 References

- Reimers, N., & Gurevych, I. (2019). _Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks_. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP).

- O'Gorman, L. (1993). _The document spectrum for page layout analysis_. IEEE Transactions on Pattern Analysis and Machine Intelligence.
