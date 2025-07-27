import os
import json
import fitz  # PyMuPDF
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
from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# model.save("model")


# ---------- CONFIG ----------
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = "./model"

TOP_K = 12
MAX_PER_DOC = 2
INTENTS_TOP_N = 1
CATEGORIES_TOP_N = 4
NUM_THREADS = 2
# ----------------------------

torch.set_num_threads(1)

# ---------- Setup Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------- Load Model ----------
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
logging.info(f"🧠 Using model: {EMBEDDING_MODEL} on CPU")

# ----------------------------------

STOPWORDS = set([
    "the", "and", "for", "a", "an", "to", "of", "in", "on", "with", "at", "by", "from", "or", "that",
    "this", "is", "are", "was", "were", "be", "have", "has", "it", "as", "but", "if", "then", "so",
    "your", "you", "i", "we", "our", "they", "them", "their", "not", "all", "can", "will", "may", "more"
])

def clean_text(text):
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219•\-]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(string.punctuation + " \n\t\r")

def extract_keywords(text):
    words = re.findall(r"\b\w{3,}\b", text.lower())
    return set(w for w in words if w not in STOPWORDS)

def extract_sections_by_heading(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        spans = []
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
            continue
        max_font_size = max(font_sizes)
        heading_indices = []
        for idx, span in enumerate(spans):
            if (
                span["text"]
                and span["is_bold"]
                and span["font_size"] >= 0.9 * max_font_size
                and len(span["text"].split()) <= 20
            ):
                heading_indices.append((idx, span["text"]))
        for h_idx, (span_idx, heading_text) in enumerate(heading_indices):
            start = span_idx + 1
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

async def extract_sections_async(base_dir, doc_list, max_workers=2):
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def extract_from_file(doc):
        filename = doc["filename"]
        pdf_path = os.path.join(base_dir, "PDFs", filename)

        def extract_sync():
            secs = extract_sections_by_heading(pdf_path)
            for s in secs:
                s["document"] = filename
            return secs

        return await loop.run_in_executor(executor, extract_sync)

    tasks = [extract_from_file(doc) for doc in doc_list]
    results = await asyncio.gather(*tasks)
    return [s for group in results for s in group]

def compute_similarity(texts, query):
    all_texts = texts + [query]
    embeddings = embedding_model.encode(all_texts, convert_to_tensor=True, batch_size=8)
    similarities = util.cos_sim(embeddings[:-1], embeddings[-1])
    return similarities.squeeze(1).tolist()

def fast_summarize(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:max_sentences])

def extract_intents(persona, job, top_n=1):
    return [f"{persona}: {job}"]

def get_dynamic_categories(sections, top_n=4):
    titles = ' '.join([clean_text(sec['section_title']) for sec in sections])
    words = [w for w in re.findall(r"\b\w{4,}\b", titles.lower()) if w not in STOPWORDS]
    most_common = [w for w, _ in Counter(words).most_common(top_n)]
    return most_common

def tag_category(title, body, categories):
    text = f"{title} {body}".lower()
    for cat in categories:
        if cat in text:
            return cat
    return "other"

def prioritize_categories(dynamic_categories, persona, job):
    persona_job_keywords = extract_keywords(persona + " " + job)
    cat_scores = []
    for cat in dynamic_categories:
        overlap = len({cat} & persona_job_keywords)
        cat_scores.append((cat, overlap))
    return [cat for cat, _ in sorted(cat_scores, key=lambda x: (-x[1], dynamic_categories.index(x[0])))]

def score_sections(sections, persona, job, intents):
    query = f"query: {intents[0]}"
    texts = [sec["text"][:300] for sec in sections]  # Truncate to speed up
    scores = compute_similarity(texts, query)
    for i, sec in enumerate(sections):
        sec['score'] = scores[i]
    return sections

def select_top_by_category(sections, categories, top_k=10, max_per_doc=2):
    final = []
    used_docs = {}
    for cat in categories:
        cat_sections = [s for s in sections if s['category'] == cat]
        for s in cat_sections:
            if used_docs.get(s['document'], 0) < max_per_doc:
                final.append(s)
                used_docs[s['document']] = used_docs.get(s['document'], 0) + 1
                if len(final) == top_k:
                    return final
                break
    for s in sections:
        if len(final) == top_k:
            break
        if used_docs.get(s['document'], 0) >= max_per_doc:
            continue
        if s in final:
            continue
        final.append(s)
        used_docs[s['document']] = used_docs.get(s['document'], 0) + 1
    return final

# ---------------- MAIN ----------------
def analyze_collection(base_dir):
    start_time = time.time()
    logging.info(f"📁 Starting analysis for: {base_dir}")

    input_path = os.path.join(base_dir, 'challenge1b_input.json')
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona = input_data['persona']['role']
    task = input_data['job_to_be_done']['task']

    all_sections = asyncio.run(
        extract_sections_async(base_dir, input_data['documents'], max_workers=NUM_THREADS)
    )

    # Pre-filter: Keep top 5 longest sections per document
    filtered = []
    for doc in input_data['documents']:
        fname = doc['filename']
        doc_secs = [s for s in all_sections if s['document'] == fname]
        top_secs = sorted(doc_secs, key=lambda s: len(s['text']), reverse=True)[:5]
        filtered.extend(top_secs)
    all_sections = filtered

    intents = extract_intents(persona, task, top_n=INTENTS_TOP_N)
    all_sections = score_sections(all_sections, persona, task, intents)

    dynamic_categories = get_dynamic_categories(all_sections, top_n=CATEGORIES_TOP_N)
    prioritized_categories = prioritize_categories(dynamic_categories, persona, task)
    for sec in all_sections:
        sec['category'] = tag_category(sec['section_title'], sec['text'], dynamic_categories)

    sorted_sections = sorted(all_sections, key=lambda x: x['score'], reverse=True)
    top_sections = select_top_by_category(sorted_sections, prioritized_categories, top_k=TOP_K, max_per_doc=MAX_PER_DOC)

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

if __name__ == "__main__":
    for i in range(1, 4):
        folder = f"Collection {i}"
        analyze_collection(folder)
