import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8") 

HERE = Path(__file__).parent.parent
PDF_PATH = HERE / "materials" / "IACS_47_Steel_Shipbuilding_Quality_2025.pdf"
QUESTIONS_PATH = HERE/"examples"/ "questions.json"
OUT_DIR = HERE / "outputs"

FIRST_PAGE, LAST_PAGE = 4, 8 # PDF pages, counted from 1
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
MAX_WORDS = 110 
MISSING = "The supplied pages do not specify an answer."

EXTRA_DEV_UNANSWERABLE = [
    "What is the allowable misalignment for butt welds?",
    "What leg length is required for fillet welds?",
    "What is the capital of France?",
]

STOP = set(
    "a an and are as at be been by can do does for from has have how in is it its of on or "
    "shall should that the their there these this to under used use was what when where which "
    "who whom why will with within without must may not".split()
)


def stem(w):
    """Very light stemmer so 'measured', 'measure', 'measuring' match."""
    for suf in ("ing", "ed", "s"):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def tokens(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def sentences(text):
    parts = re.split(r"(?<=[.;:])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]



# Extraction + chunking

HEADER_FOOTER = [re.compile(p) for p in (r"^No\.?\s*47$", r"^Page \d+ of \d+", r"^IACS Rec")]
MARGIN = re.compile(r"^(?:No\.|\(cont\)|47)\s+(?=\S)") 
CLAUSE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z]") 


def clean_lines(page_text):
    for raw in page_text.splitlines():
        line = raw.strip()
        while True:
            new = MARGIN.sub("", line).strip()
            if new == line:
                break
            line = new
        if not line or line in {"No.", "47", "(cont)"}:
            continue
        if any(p.match(line) for p in HEADER_FOOTER):
            continue
        yield line


def split_long(text):
    if len(text.split()) <= MAX_WORDS:
        return [text]
    pieces, cur = [], []
    for s in sentences(text):
        if cur and len(" ".join(cur + [s]).split()) > MAX_WORDS:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def build_chunks():
    reader = PdfReader(str(PDF_PATH))
    chunks = []
    for page in range(FIRST_PAGE, LAST_PAGE + 1):
        text = reader.pages[page - 1].extract_text() or ""
        buf = []

        def flush():
            if buf:
                for piece in split_long(" ".join(buf)):
                    chunks.append({"id": len(chunks), "page": page, "text": piece})
                buf.clear()

        for line in clean_lines(text):
            if CLAUSE.match(line):
                flush() # a new numbered clause starts a new chunk
            buf.append(line)
        flush() # chunks never cross a page boundary
    return chunks


 # Retrievers (both return (chunk_index, score) for the top K)

class KeywordRetriever:
    name = "Keyword (BM25)"

    def __init__(self, chunks):
        self.bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])

    def search(self, question, k=TOP_K):
        qt = tokens(question)
        scores = self.bm25.get_scores(qt)
        order = np.argsort(-scores)[:k]
        norm = max(len(qt), 1) # per-query-term score so the threshold is comparable across questions
        return [(int(i), float(scores[i]) / norm) for i in order]


class EmbeddingRetriever:
    name = f"Embedding ({EMBED_MODEL})"

    def __init__(self, chunks):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.vecs = self.model.encode([c["text"] for c in chunks], normalize_embeddings=True)

    def search(self, question, k=TOP_K):
        qv = self.model.encode([question], normalize_embeddings=True)[0]
        sims = self.vecs @ qv # cosine similarity (vectors are normalised)
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


 # Answering (identical for both methods)

def make_answer(question, hits, chunks, threshold):
    """Best-matching sentence from the top-3 chunks, with a page citation."""
    if not hits or hits[0][1] < threshold:
        return MISSING
    q = set(tokens(question))
    best = None
    for rank, (ci, _) in enumerate(hits):
        for s in sentences(chunks[ci]["text"]):
            key = (len(q & set(tokens(s))), -rank) # more overlap wins; earlier chunk breaks ties
            if best is None or key > best[0]:
                best = (key, s, chunks[ci]["page"])
    if best is None or best[0][0] == 0:
        return MISSING
    return f"{best[1]} (p. {best[2]})"


 # Threshold calibration (dev only)

def calibrate(retriever, chunks, dev_questions):
    items = [] 
    for q in dev_questions:
        hits = retriever.search(q["question"])
        pages = {chunks[i]["page"] for i, _ in hits}
        if q["relevant_pdf_pages"] and pages & set(q["relevant_pdf_pages"]):
            items.append((hits[0][1], True))
    for text in EXTRA_DEV_UNANSWERABLE:
        items.append((retriever.search(text)[0][1], False))

    scores = sorted({s for s, _ in items})

    def correct(t):
        return sum((s >= t) if want else (s < t) for s, want in items)

    best_n = max(correct(t) for t in scores)
    t = min(x for x in scores if correct(x) == best_n)
    lower = [s for s in scores if s < t]
    return (t + max(lower)) / 2 if lower else t * 0.9 # midpoint to the next lower score


 # Evaluation

def run(questions, retriever, chunks, threshold):
    rows = []
    for q in questions:
        hits = retriever.search(q["question"])
        relevant = q["relevant_pdf_pages"]
        pages = [chunks[i]["page"] for i, _ in hits]
        ans = make_answer(q["question"], hits, chunks, threshold)
        rows.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "relevant_pdf_pages": relevant,
            "answerable": bool(relevant),
            "top1_score": round(hits[0][1], 4),
            "retrieval_hit": bool(set(pages) & set(relevant)) if relevant else None,
            "top3": [{"page": chunks[i]["page"], "score": round(s, 4), "text": chunks[i]["text"]}
                     for i, s in hits],
            "answer": ans,
            "said_no_answer": ans == MISSING,
        })
    return rows


def metrics(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {
        "hits": sum(r["retrieval_hit"] for r in ans),
        "answerable": len(ans),
        "correct_no_answer": sum(r["said_no_answer"] for r in unans),
        "unanswerable": len(unans),
        "wrong_no_answer": sum(r["said_no_answer"] for r in ans),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", action="store_true", help="only dump chunks and exit")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    chunks = build_chunks()
    (OUT_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(chunks)} chunks from PDF pages {FIRST_PAGE}-{LAST_PAGE} -> outputs/chunks.json")
    if args.chunks:
        return

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]

    all_results, table = {}, []
    for retriever in (KeywordRetriever(chunks), EmbeddingRetriever(chunks)):
        threshold = calibrate(retriever, chunks, dev) # dev only, then frozen
        dev_rows = run(dev, retriever, chunks, threshold)
        test_rows = run(test, retriever, chunks, threshold) # test is evaluated once, never tuned on
        dm, tm = metrics(dev_rows), metrics(test_rows)
        all_results[retriever.name] = {"threshold": threshold, "dev": dev_rows, "test": test_rows}
        table.append((retriever.name, threshold, dm, tm))

        print(f"\n=== {retriever.name} | no-answer threshold {threshold:.3f} ===")
        for r in dev_rows + test_rows:
            hit = "n/a" if r["retrieval_hit"] is None else ("HIT " if r["retrieval_hit"] else "MISS")
            print(f"[{r['split']:4}] {r['id']:4} {hit} top1={r['top1_score']:.3f} {r['answer'][:90]}")

    lines = [
        "| Method | Dev hits | Test hits | Test correct 'no answer' | Wrong 'no answer' on answerable (dev/test) | Threshold |",
        "|---|---|---|---|---|---|",
    ]
    for name, thr, dm, tm in table:
        lines.append(
            f"| {name} | {dm['hits']}/{dm['answerable']} | {tm['hits']}/{tm['answerable']} | "
            f"{tm['correct_no_answer']}/{tm['unanswerable']} | {dm['wrong_no_answer']}/{tm['wrong_no_answer']} | {thr:.3f} |"
        )
    md = "\n".join(lines)
    print("\n" + md)

    (OUT_DIR / "comparison.md").write_text(md + "\n", encoding="utf-8")
    (OUT_DIR / "results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved outputs/comparison.md and outputs/results.json")


if __name__ == "__main__":
    main()
