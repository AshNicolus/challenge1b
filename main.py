import os
import json
import fitz  # PyMuPDF for PDF reading
import re
import string
import torch
import logging
import time
import asyncio
import concurrent.futures
from datetime import datetime
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any, Set, Tuple

# ----------- Configuration Constants -----------
# These settings control the core behavior of the analysis pipeline.
EMBEDDING_MODEL = "./model"          # Local path to the SentenceTransformer model.
TOP_K = 12                           # Total number of top sections to select for the final output.
MAX_PER_DOC = 2                      # Maximum number of sections to pick from any single document to ensure diversity.
INTENTS_TOP_N = 1                    # Number of primary "intents" to derive from the persona and job description.
CATEGORIES_TOP_N = 4                 # Number of dynamic topic categories to generate from the document content.
NUM_THREADS = 2                      # Number of threads for parallel processing of PDFs.

# Limit PyTorch threads to avoid resource contention in a CPU-only environment.
# This is crucial for performance when running multiple processes in parallel.
torch.set_num_threads(1)

# ----------- Logging Setup -----------
# Configure a clear and consistent logging format for monitoring and debugging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler()]
)

# ----------- Load SentenceTransformer Model -----------
# The AI model is loaded once at the start to avoid the overhead of reloading it for each operation.
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
logging.info(f"🧠 Using model: {EMBEDDING_MODEL} on CPU")

# ----------- Common Stopwords -----------
# A predefined set of common English words to be filtered out during keyword extraction.
STOPWORDS = set([
    "the", "and", "for", "a", "an", "to", "of", "in", "on", "with", "at", "by", "from", "or", "that",
    "this", "is", "are", "was", "were", "be", "have", "has", "it", "as", "but", "if", "then", "so",
    "your", "you", "i", "we", "our", "they", "them", "their", "not", "all", "can", "will", "may", "more"
])

# ----------- Utility: Clean Text -----------
def clean_text(text: str) -> str:
    """
    Removes unwanted characters and normalizes whitespace in a string.

    Args:
        text: The input string to clean.

    Returns:
        A cleaned version of the string.
    """
    # Remove bullet points and similar list markers.
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219•\-]+", "", text)
    # Collapse multiple whitespace characters into a single space.
    text = re.sub(r"\s+", " ", text)
    # Remove any leading or trailing punctuation and whitespace.
    return text.strip(string.punctuation + " \n\t\r")

# ----------- Utility: Extract Keywords -----------
def extract_keywords(text: str) -> Set[str]:
    """
    Extracts a set of meaningful keywords from a given text.

    Args:
        text: The text to extract keywords from.

    Returns:
        A set of unique, potentially important keywords.
    """
    # Find all words that are at least 3 characters long.
    words = re.findall(r"\b\w{3,}\b", text.lower())
    # Filter out any common stopwords.
    return set(w for w in words if w not in STOPWORDS)

# ----------- PDF Section Extraction using Heading Heuristics -----------
def extract_sections_by_heading(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts structured sections from a PDF based on visual heading cues.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A list of dictionaries, each representing a section.
    """
    doc = fitz.open(pdf_path)
    sections = []

    for page_num, page in enumerate(doc):
        # Extract text with detailed style information (font, size, etc.).
        blocks = page.get_text("dict")["blocks"]
        spans = []

        # Flatten the nested structure into a simple list of text spans.
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    spans.append({
                        "text": span["text"].strip(),
                        "font_size": span["size"],
                        "is_bold": "Bold" in span["font"]
                    })

        font_sizes = [s["font_size"] for s in spans if s["text"]]
        if not font_sizes:
            continue # Skip empty pages.

        # Identify headings based on font size, boldness, and length.
        max_font_size = max(font_sizes)
        heading_indices = []
        for idx, span in enumerate(spans):
            if (
                span["text"] and
                span["is_bold"] and
                span["font_size"] >= 0.9 * max_font_size and
                len(span["text"].split()) <= 20
            ):
                heading_indices.append((idx, span["text"]))

        # Group content under each identified heading.
        for h_idx, (span_idx, heading_text) in enumerate(heading_indices):
            start = span_idx + 1
            # A section ends where the next heading begins.
            end = heading_indices[h_idx + 1][0] if h_idx + 1 < len(heading_indices) else len(spans)
            content = [s["text"] for s in spans[start:end] if s["text"]]
            full_text = " ".join(content).strip()
            
            if heading_text and full_text:
                sections.append({
                    "section_title": heading_text,
                    "text": full_text,
                    "page_number": page_num + 1
                })

    doc.close()
    return sections

# ----------- Async Section Extraction from PDFs -----------
async def extract_sections_async(base_dir: str, doc_list: List[Dict[str, str]], max_workers: int = 2) -> List[Dict[str, Any]]:
    """
    Extracts sections from multiple PDFs in parallel using a thread pool.

    Args:
        base_dir: The root directory of the collection.
        doc_list: A list of document metadata dictionaries.
        max_workers: The number of parallel threads to use.

    Returns:
        A single flat list of all sections from all documents.
    """
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    # This helper function wraps the synchronous extraction logic for async execution.
    async def extract_from_file(doc: Dict[str, str]) -> List[Dict[str, Any]]:
        filename = doc["filename"]
        pdf_path = os.path.join(base_dir, "PDFs", filename)

        # The synchronous (blocking) part of the code.
        def extract_sync() -> List[Dict[str, Any]]:
            secs = extract_sections_by_heading(pdf_path)
            # Tag each section with its source document filename.
            for s in secs:
                s["document"] = filename
            return secs

        # Schedule the blocking function to run in the thread pool.
        return await loop.run_in_executor(executor, extract_sync)

    # Launch all extraction tasks concurrently and wait for them to complete.
    tasks = [extract_from_file(doc) for doc in doc_list]
    results = await asyncio.gather(*tasks)
    # Flatten the list of lists into a single list of sections.
    return [s for group in results for s in group]

# ----------- Compute Semantic Similarity Scores -----------
def compute_similarity(texts: List[str], query: str) -> List[float]:
    """
    Computes cosine similarity between a list of texts and a query using the AI model.

    Args:
        texts: A list of text strings (e.g., section contents).
        query: The query string to compare against.

    Returns:
        A list of similarity scores, one for each input text.
    """
    all_texts = texts + [query]
    # Convert all texts into vector embeddings.
    embeddings = embedding_model.encode(all_texts, convert_to_tensor=True, batch_size=8)
    # Compare each text embedding to the query embedding.
    similarities = util.cos_sim(embeddings[:-1], embeddings[-1])
    return similarities.squeeze(1).tolist()

# ----------- Quick Summarization (Top 3 Sentences) -----------
def fast_summarize(text: str, max_sentences: int = 3) -> str:
    """
    Creates a quick, extractive summary by taking the first few sentences of a text.

    Args:
        text: The text to summarize.
        max_sentences: The maximum number of sentences to include.

    Returns:
        A summarized string.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:max_sentences])

# ----------- Extract Intents from Persona & Job -----------
def extract_intents(persona: str, job: str, top_n: int = 1) -> List[str]:
    """
    Creates a focused "intent" query by combining the persona and their job.

    Args:
        persona: The role of the user (e.g., "Investment Analyst").
        job: The task the user wants to accomplish.

    Returns:
        A list containing the combined intent string.
    """
    return [f"{persona}: {job}"]

# ----------- Get Most Common Section Topics Dynamically -----------
def get_dynamic_categories(sections: List[Dict[str, Any]], top_n: int = 4) -> List[str]:
    """
    Discovers the main topics by finding the most common keywords in section titles.

    Args:
        sections: A list of all extracted sections.
        top_n: The number of top categories to return.

    Returns:
        A list of the most common topic keywords.
    """
    titles = ' '.join([clean_text(sec['section_title']) for sec in sections])
    words = [w for w in re.findall(r"\b\w{4,}\b", titles.lower()) if w not in STOPWORDS]
    most_common = [w for w, _ in Counter(words).most_common(top_n)]
    return most_common

# ----------- Tag Section to Category Based on Text Match -----------
def tag_category(title: str, body: str, categories: List[str]) -> str:
    """
    Assigns a section to a category if a category keyword appears in its text.

    Args:
        title: The section's title.
        body: The section's body text.
        categories: The list of dynamic categories to check against.

    Returns:
        The name of the first matching category, or "other".
    """
    text = f"{title} {body}".lower()
    for cat in categories:
        if cat in text:
            return cat
    return "other"

# ----------- Prioritize Categories by Persona-Relevance -----------
def prioritize_categories(dynamic_categories: List[str], persona: str, job: str) -> List[str]:
    """
    Ranks dynamic categories based on relevance to the user's query.

    Args:
        dynamic_categories: The list of discovered categories.
        persona: The user's role.
        job: The user's task.

    Returns:
        A new list of categories, sorted from most to least relevant.
    """
    persona_job_keywords = extract_keywords(persona + " " + job)
    cat_scores = []
    for cat in dynamic_categories:
        overlap = len({cat} & persona_job_keywords)
        cat_scores.append((cat, overlap))
    # Sort by score (descending), then by original order for stability.
    return [cat for cat, _ in sorted(cat_scores, key=lambda x: (-x[1], dynamic_categories.index(x[0])))]

# ----------- Assign Similarity Score to Sections -----------
def score_sections(sections: List[Dict[str, Any]], persona: str, job: str, intents: List[str]) -> List[Dict[str, Any]]:
    """
    Assigns a semantic similarity score to each section based on the user's intent.

    Args:
        sections: The list of sections to score.
        persona: The user's role.
        job: The user's task.
        intents: The generated intent query list.

    Returns:
        The list of sections, with a 'score' key added to each.
    """
    query = f"query: {intents[0]}"
    # Truncate text for faster embedding calculation.
    texts = [sec["text"][:300] for sec in sections]
    scores = compute_similarity(texts, query)
    for i, sec in enumerate(sections):
        sec['score'] = scores[i]
    return sections

# ----------- Select Final Sections By Category and Relevance -----------
def select_top_by_category(sections: List[Dict[str, Any]], categories: List[str], top_k: int = 10, max_per_doc: int = 2) -> List[Dict[str, Any]]:
    """
    Selects the final list of sections, balancing relevance and diversity.

    Args:
        sections: The list of all scored and categorized sections.
        categories: The prioritized list of dynamic categories.
        top_k: The total number of sections to return.
        max_per_doc: Max sections allowed from any single document.

    Returns:
        A final, curated list of the most relevant sections.
    """
    final = []
    used_docs = {} # Tracks how many sections are taken from each document.

    # First pass: try to pick one top-scoring section from each category.
    for cat in categories:
        cat_sections = [s for s in sections if s['category'] == cat]
        for s in cat_sections:
            if used_docs.get(s['document'], 0) < max_per_doc:
                final.append(s)
                used_docs[s['document']] = used_docs.get(s['document'], 0) + 1
                if len(final) == top_k:
                    return final
                break # Move to the next category.

    # Second pass: fill remaining slots with the highest-scoring sections available.
    for s in sections:
        if len(final) == top_k:
            break
        if used_docs.get(s['document'], 0) >= max_per_doc or s in final:
            continue
        final.append(s)
        used_docs[s['document']] = used_docs.get(s['document'], 0) + 1
    return final

# ----------- Main Pipeline to Analyze a Collection Folder -----------
def analyze_collection(base_dir: str):
    """
    Orchestrates the entire analysis pipeline for a single document collection.

    Args:
        base_dir: The path to the collection folder (e.g., "Collection 1").
    """
    start_time = time.time()
    logging.info(f"📁 Starting analysis for: {base_dir}")

    # Load input JSON (persona, job, and documents).
    input_path = os.path.join(base_dir, 'challenge1b_input.json')
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona = input_data['persona']['role']
    task = input_data['job_to_be_done']['task']

    # Step 1: Extract sections from all PDFs in parallel.
    all_sections = asyncio.run(
        extract_sections_async(base_dir, input_data['documents'], max_workers=NUM_THREADS)
    )

    # Step 2: Filter to keep only the top 5 longest sections per document.
    # This pre-filters out minor sections to speed up the semantic analysis.
    filtered = []
    for doc in input_data['documents']:
        fname = doc['filename']
        doc_secs = [s for s in all_sections if s['document'] == fname]
        top_secs = sorted(doc_secs, key=lambda s: len(s['text']), reverse=True)[:5]
        filtered.extend(top_secs)
    all_sections = filtered

    # Step 3: Score all sections based on semantic similarity to the user's intent.
    intents = extract_intents(persona, task, top_n=INTENTS_TOP_N)
    all_sections = score_sections(all_sections, persona, task, intents)

    # Step 4: Dynamically discover topics and tag each section.
    dynamic_categories = get_dynamic_categories(all_sections, top_n=CATEGORIES_TOP_N)
    prioritized_categories = prioritize_categories(dynamic_categories, persona, task)
    for sec in all_sections:
        sec['category'] = tag_category(sec['section_title'], sec['text'], dynamic_categories)

    # Step 5: Select the best sections, ensuring diversity across topics and documents.
    sorted_sections = sorted(all_sections, key=lambda x: x['score'], reverse=True)
    top_sections = select_top_by_category(sorted_sections, prioritized_categories, top_k=TOP_K, max_per_doc=MAX_PER_DOC)

    # Step 6: Build the final output JSON structure.
    extracted_sections = []
    subsection_analysis = []
    for rank, sec in enumerate(top_sections, start=1):
        extracted_sections.append({
            "document": sec["document"],
            "section_title": sec["section_title"],
            "importance_rank": rank,
            "page_number": sec["page_number"]
        })
        subsection_analysis.append({
            "document": sec["document"],
            "refined_text": fast_summarize(sec["text"]),
            "page_number": sec["page_number"]
        })

    # Step 7: Write the final output JSON file.
    output = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in input_data['documents']],
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": str(datetime.now())
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    output_path = os.path.join(base_dir, 'challenge1b_output.json')
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    duration = time.time() - start_time
    logging.info(f"✅ Finished {base_dir} in {duration:.2f} seconds.")
    print(f"✅ Output written to {output_path}")

# ----------- Execute Across All Collections -----------
if __name__ == "__main__":
    # This loop will find and process each "Collection X" folder.
    for i in range(1, 4):
        folder = f"Collection {i}"
        analyze_collection(folder)
