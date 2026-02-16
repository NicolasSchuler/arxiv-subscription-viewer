"""Paper similarity — TF-IDF index, cosine + Jaccard similarity."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING

from arxiv_browser.models import STOPWORDS, Paper, PaperMetadata

if TYPE_CHECKING:
    from arxiv_browser.parsing import parse_arxiv_date
else:
    # Deferred import to avoid circular deps at runtime
    def parse_arxiv_date(date_str: str) -> datetime:  # type: ignore[assignment]
        from arxiv_browser.parsing import parse_arxiv_date as _parse

        return _parse(date_str)


# ============================================================================
# Similarity Constants
# ============================================================================

SIMILARITY_TOP_N = 10  # Number of similar papers to show
SIMILARITY_RECENCY_WEIGHT = 0.08
SIMILARITY_STARRED_BOOST = 0.05
SIMILARITY_UNREAD_BOOST = 0.03
SIMILARITY_READ_PENALTY = 0.02
SIMILARITY_RECENCY_DAYS = 365
SIMILARITY_WEIGHT_CATEGORY = 0.30
SIMILARITY_WEIGHT_AUTHOR = 0.20
SIMILARITY_WEIGHT_TEXT = 0.50

# ============================================================================
# TF-IDF Index
# ============================================================================

_TFIDF_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")


def _tokenize_for_tfidf(text: str | None) -> list[str]:
    """Tokenize text for TF-IDF, preserving term frequency."""
    if not text:
        return []
    return [tok for tok in _TFIDF_TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute sublinear term frequency: 1 + log(count)."""
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    return {term: 1.0 + math.log(count) for term, count in counts.items()}


class TfidfIndex:
    """Sparse TF-IDF index for cosine similarity over a paper corpus."""

    __slots__ = ("_idf", "_norms", "_tfidf_vectors")

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._tfidf_vectors: dict[str, dict[str, float]] = {}
        self._norms: dict[str, float] = {}

    @staticmethod
    def build(papers: list[Paper], text_fn: Callable[[Paper], str]) -> TfidfIndex:
        """Build index from papers. text_fn extracts text per paper."""
        index = TfidfIndex()
        doc_tfs: dict[str, dict[str, float]] = {}
        df: dict[str, int] = {}
        for paper in papers:
            tokens = _tokenize_for_tfidf(text_fn(paper))
            if not tokens:
                continue
            tf = _compute_tf(tokens)
            doc_tfs[paper.arxiv_id] = tf
            for term in tf:
                df[term] = df.get(term, 0) + 1
        n = len(doc_tfs)
        if n < 2:
            return index
        index._idf = {term: math.log(1 + n / (1 + freq)) for term, freq in df.items()}
        for arxiv_id, tf in doc_tfs.items():
            vec: dict[str, float] = {}
            norm_sq = 0.0
            for term, tf_val in tf.items():
                tfidf = tf_val * index._idf.get(term, 0.0)
                if tfidf > 0.0:
                    vec[term] = tfidf
                    norm_sq += tfidf * tfidf
            index._tfidf_vectors[arxiv_id] = vec
            index._norms[arxiv_id] = math.sqrt(norm_sq) if norm_sq > 0.0 else 0.0
        return index

    def cosine_similarity(self, id_a: str, id_b: str) -> float:
        """Cosine similarity between two papers by arxiv_id."""
        vec_a = self._tfidf_vectors.get(id_a)
        vec_b = self._tfidf_vectors.get(id_b)
        if not vec_a or not vec_b:
            return 0.0
        norm_a = self._norms.get(id_a, 0.0)
        norm_b = self._norms.get(id_b, 0.0)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        # Iterate over smaller vector for efficiency
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        dot = sum(w * vec_b.get(t, 0.0) for t, w in vec_a.items())
        return dot / (norm_a * norm_b)

    def __contains__(self, arxiv_id: str) -> bool:
        return arxiv_id in self._tfidf_vectors

    def __len__(self) -> int:
        return len(self._tfidf_vectors)


# ============================================================================
# Keyword & Author Extraction
# ============================================================================


def _extract_keywords(text: str | None, min_length: int = 4) -> set[str]:
    """Extract significant keywords from text, filtering stopwords."""
    if not text:
        return set()
    words = set()
    for word in text.lower().split():
        # Remove non-alphanumeric characters
        clean = "".join(c for c in word if c.isalnum())
        if len(clean) >= min_length and clean not in STOPWORDS:
            words.add(clean)
    return words


_AUTHOR_SPLIT_RE = re.compile(r",|(?:\s+and\s+)")


def _extract_author_lastnames(authors: str) -> set[str]:
    """Extract last names from author string."""
    lastnames = set()
    # Split by common separators
    for author in _AUTHOR_SPLIT_RE.split(authors):
        parts = author.strip().split()
        if parts:
            # Last word is typically the last name
            lastname = parts[-1].lower()
            # Remove non-alphanumeric
            lastname = "".join(c for c in lastname if c.isalnum())
            if lastname:
                lastnames.add(lastname)
    return lastnames


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ============================================================================
# Paper Similarity Scoring
# ============================================================================


def compute_paper_similarity(
    paper_a: Paper,
    paper_b: Paper,
    abstract_a: str | None = None,
    abstract_b: str | None = None,
    tfidf_index: TfidfIndex | None = None,
) -> float:
    """Compute weighted similarity score between two papers.

    When tfidf_index is provided, uses TF-IDF cosine similarity for text (50%)
    with category Jaccard (30%) and author Jaccard (20%).

    Without tfidf_index, falls back to legacy Jaccard weights:
    categories 40%, authors 30%, title keywords 20%, abstract keywords 10%.

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if paper_a.arxiv_id == paper_b.arxiv_id:
        return 1.0

    # Category similarity
    cats_a = set(paper_a.categories.split())
    cats_b = set(paper_b.categories.split())
    cat_sim = _jaccard_similarity(cats_a, cats_b)

    # Author similarity
    authors_a = _extract_author_lastnames(paper_a.authors)
    authors_b = _extract_author_lastnames(paper_b.authors)
    author_sim = _jaccard_similarity(authors_a, authors_b)

    if tfidf_index is not None:
        # TF-IDF branch: text similarity from pre-built index
        text_sim = tfidf_index.cosine_similarity(paper_a.arxiv_id, paper_b.arxiv_id)
        return (
            SIMILARITY_WEIGHT_CATEGORY * cat_sim
            + SIMILARITY_WEIGHT_AUTHOR * author_sim
            + SIMILARITY_WEIGHT_TEXT * text_sim
        )

    # Legacy Jaccard branch
    title_kw_a = _extract_keywords(paper_a.title)
    title_kw_b = _extract_keywords(paper_b.title)
    title_sim = _jaccard_similarity(title_kw_a, title_kw_b)

    if abstract_a is None:
        abstract_a = paper_a.abstract or ""
    if abstract_b is None:
        abstract_b = paper_b.abstract or ""
    abstract_kw_a = _extract_keywords(abstract_a)
    abstract_kw_b = _extract_keywords(abstract_b)
    abstract_sim = _jaccard_similarity(abstract_kw_a, abstract_kw_b)

    return 0.4 * cat_sim + 0.3 * author_sim + 0.2 * title_sim + 0.1 * abstract_sim


def build_similarity_corpus_key(papers: list[Paper]) -> str:
    """Build a deterministic fingerprint for the similarity corpus."""
    h = sha256()
    for paper in papers:
        abstract_text = paper.abstract if paper.abstract is not None else paper.abstract_raw
        h.update(paper.arxiv_id.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(paper.title.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update((abstract_text or "").encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def find_similar_papers(
    target: Paper,
    all_papers: list[Paper],
    top_n: int = SIMILARITY_TOP_N,
    metadata: dict[str, PaperMetadata] | None = None,
    abstract_lookup: Callable[[Paper], str] | None = None,
    tfidf_index: TfidfIndex | None = None,
) -> list[tuple[Paper, float]]:
    """Find the top N most similar papers to the target.

    Args:
        target: The paper to find similarities for
        all_papers: List of all papers to search
        top_n: Number of similar papers to return

    Returns:
        List of (paper, score) tuples, sorted by score descending
    """
    scored = []
    if abstract_lookup is None:

        def _default_abstract_lookup(paper: Paper) -> str:
            return paper.abstract or ""

        abstract_lookup = _default_abstract_lookup

    use_tfidf = tfidf_index is not None
    target_abstract = abstract_lookup(target) if not use_tfidf else None

    newest_date = datetime.min
    for paper in all_papers:
        paper_date = parse_arxiv_date(paper.date)
        if paper_date > newest_date:
            newest_date = paper_date

    def metadata_boost(arxiv_id: str) -> float:
        if not metadata:
            return 0.0
        entry = metadata.get(arxiv_id)
        if not entry:
            return 0.0
        boost = 0.0
        if entry.starred:
            boost += SIMILARITY_STARRED_BOOST
        if entry.is_read:
            boost -= SIMILARITY_READ_PENALTY
        else:
            boost += SIMILARITY_UNREAD_BOOST
        return boost

    def recency_score(paper: Paper) -> float:
        if newest_date == datetime.min:
            return 0.0
        paper_date = parse_arxiv_date(paper.date)
        if paper_date == datetime.min:
            return 0.0
        age_days = max(0, (newest_date - paper_date).days)
        return max(0.0, 1.0 - (age_days / SIMILARITY_RECENCY_DAYS))

    for paper in all_papers:
        if paper.arxiv_id == target.arxiv_id:
            continue
        paper_abstract = abstract_lookup(paper) if not use_tfidf else None
        score = compute_paper_similarity(
            target,
            paper,
            target_abstract,
            paper_abstract,
            tfidf_index=tfidf_index,
        )
        score += SIMILARITY_RECENCY_WEIGHT * recency_score(paper)
        score += metadata_boost(paper.arxiv_id)
        score = max(0.0, min(1.0, score))
        if score > 0:
            scored.append((paper, score))

    # Sort by score descending and take top N
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


__all__ = [
    "SIMILARITY_READ_PENALTY",
    "SIMILARITY_RECENCY_DAYS",
    "SIMILARITY_RECENCY_WEIGHT",
    "SIMILARITY_STARRED_BOOST",
    "SIMILARITY_TOP_N",
    "SIMILARITY_UNREAD_BOOST",
    "SIMILARITY_WEIGHT_AUTHOR",
    "SIMILARITY_WEIGHT_CATEGORY",
    "SIMILARITY_WEIGHT_TEXT",
    "TfidfIndex",
    "build_similarity_corpus_key",
    "compute_paper_similarity",
    "find_similar_papers",
]
