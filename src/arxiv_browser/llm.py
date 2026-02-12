"""LLM summary, relevance scoring, auto-tagging — prompts, DB persistence, parsing."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from arxiv_browser.models import CONFIG_APP_NAME, Paper, UserConfig

logger = logging.getLogger(__name__)
SUMMARY_DB_FILENAME = "summaries.db"
RELEVANCE_DB_FILENAME = "relevance.db"

# ============================================================================
# LLM Prompt Templates & Presets
# ============================================================================

DEFAULT_LLM_PROMPT = (
    "You are an expert science communicator who makes complex research accessible. "
    "Summarize the following arXiv paper for a Computer Science university student "
    "who may not be an expert in this specific subfield. "
    "Avoid jargon where possible; when technical terms are necessary, briefly "
    "explain them in parentheses. Keep the TOTAL response under 400 words.\n\n"
    "## Problem\n"
    "What real-world or theoretical problem does this paper address, and why does "
    "it matter? (2-3 sentences)\n\n"
    "## Approach\n"
    "Explain the key idea and methodology at a high level. Focus on the intuition — "
    "what makes the approach work? (3-5 sentences)\n\n"
    "## Results\n"
    "What did the authors demonstrate? Mention key experiments or findings and how "
    "they compare to previous work. (2-3 sentences)\n\n"
    "## Limitations\n"
    "What are the main limitations or open questions? (2-3 sentences)\n\n"
    "## Key Takeaway\n"
    "In one sentence, what should the reader remember about this paper?\n\n"
    "---\n"
    "Paper: {title}\n"
    "Authors: {authors}\n"
    "Categories: {categories}\n\n"
    "{paper_content}"
)

LLM_PRESETS: dict[str, str] = {
    "claude": "claude -p {prompt}",
    "codex": "codex exec {prompt}",
    "llm": "llm {prompt}",
    "copilot": "copilot --model gpt-5-mini -p {prompt}",
    "opencode": "opencode run -m zai-coding-plan/glm-4.7 -- {prompt}",
}

# Structured summary mode templates
SUMMARY_MODES: dict[str, tuple[str, str]] = {
    "default": (
        "Full summary (Problem / Approach / Results)",
        DEFAULT_LLM_PROMPT,
    ),
    "quick": (
        "Quick abstract-only summary",
        "Provide a concise 3-5 sentence summary focused on key contribution, method, "
        "and main result. Keep it direct and easy to scan.\n\n"
        "Paper: {title}\nAuthors: {authors}\nCategories: {categories}\n\n"
        "{paper_content}",
    ),
    "tldr": (
        "1-2 sentence TLDR",
        "Provide a 1-2 sentence TLDR summary of this paper. Be concise and capture "
        "the key contribution.\n\n"
        "Paper: {title}\nAuthors: {authors}\nCategories: {categories}\n\n"
        "{paper_content}",
    ),
    "methods": (
        "Technical methodology deep-dive",
        "Analyze the technical methodology of this paper in detail (~500 words). "
        "Focus on:\n"
        "1. The core algorithm or technique\n"
        "2. Key mathematical formulations or architectural choices\n"
        "3. Training/optimization details\n"
        "4. How it differs from prior approaches\n\n"
        "Paper: {title}\nAuthors: {authors}\nCategories: {categories}\n\n"
        "{paper_content}",
    ),
    "results": (
        "Key experimental results with numbers",
        "Summarize the key experimental results of this paper. Focus on:\n"
        "1. Main benchmarks and datasets used\n"
        "2. Quantitative results (accuracy, speedup, etc.) with specific numbers\n"
        "3. Comparisons with baselines and state-of-the-art\n"
        "4. Ablation study findings\n\n"
        "Paper: {title}\nAuthors: {authors}\nCategories: {categories}\n\n"
        "{paper_content}",
    ),
    "comparison": (
        "Comparison with related work",
        "Compare this paper with related work in the field. Focus on:\n"
        "1. What prior approaches exist for this problem\n"
        "2. How this paper's method differs from each\n"
        "3. Advantages and disadvantages vs alternatives\n"
        "4. Where this work fits in the broader research landscape\n\n"
        "Paper: {title}\nAuthors: {authors}\nCategories: {categories}\n\n"
        "{paper_content}",
    ),
}

RELEVANCE_PROMPT_TEMPLATE = (
    "Rate this paper's relevance to the research interests below.\n"
    'Return ONLY valid JSON: {{"score": N, "reason": "..."}}\n'
    "- score: integer 1-10 (10 = highly relevant)\n"
    "- reason: 1 sentence explaining the rating\n\n"
    "Research interests: {interests}\n\n"
    "Title: {title}\n"
    "Authors: {authors}\n"
    "Categories: {categories}\n"
    "Abstract: {abstract}\n"
)

AUTO_TAG_PROMPT_TEMPLATE = (
    "Suggest tags for this academic paper based on the taxonomy below.\n"
    'Return ONLY valid JSON: {{"tags": ["tag1", "tag2", ...]}}\n'
    "- Use the namespace:value format (e.g. topic:llm, method:quantization)\n"
    "- Prefer existing tags from the taxonomy when they fit\n"
    "- Suggest 2-5 tags total\n"
    "- Tags should be lowercase, concise (1-3 words), and use hyphens for multi-word values\n\n"
    "Existing taxonomy: {taxonomy}\n\n"
    "Title: {title}\n"
    "Authors: {authors}\n"
    "Categories: {categories}\n"
    "Abstract: {abstract}\n"
)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer questions about this paper.\n"
    "Be concise and specific. Reference paper details when relevant.\n\n"
    "Paper: {title}\nAuthors: {authors}\nCategories: {categories}\n\n"
    "{paper_content}"
)

# ============================================================================
# LLM Summary Persistence (SQLite)
# ============================================================================


def get_summary_db_path() -> Path:
    """Get the path to the summary SQLite database."""
    config_dir = Path(user_config_dir(CONFIG_APP_NAME))
    return config_dir / SUMMARY_DB_FILENAME


def _init_summary_db(db_path: Path) -> None:
    """Create the summaries table with composite PK if it doesn't exist.

    Migrates from old single-PK schema (arxiv_id only) to composite PK
    (arxiv_id, command_hash) to support multiple summary modes per paper.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        # Check if table exists with old schema (single PK on arxiv_id)
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='summaries'"
        ).fetchone()
        if row and "PRIMARY KEY (arxiv_id, command_hash)" not in row[0]:
            conn.execute("DROP TABLE summaries")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS summaries ("
            "  arxiv_id TEXT NOT NULL,"
            "  command_hash TEXT NOT NULL,"
            "  summary TEXT NOT NULL,"
            "  created_at TEXT NOT NULL,"
            "  PRIMARY KEY (arxiv_id, command_hash)"
            ")"
        )


def _load_summary(db_path: Path, arxiv_id: str, command_hash: str) -> str | None:
    """Load a cached summary if it exists and the command hash matches."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT summary FROM summaries WHERE arxiv_id = ? AND command_hash = ?",
                (arxiv_id, command_hash),
            ).fetchone()
            return row[0] if row else None
    except sqlite3.Error:
        logger.warning("Failed to load summary for %s", arxiv_id, exc_info=True)
        return None


def _save_summary(db_path: Path, arxiv_id: str, summary: str, command_hash: str) -> None:
    """Persist a summary to the SQLite database."""
    try:
        _init_summary_db(db_path)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO summaries (arxiv_id, summary, command_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (arxiv_id, summary, command_hash, datetime.now().isoformat()),
            )
    except sqlite3.Error:
        logger.warning("Failed to save summary for %s", arxiv_id, exc_info=True)


def _compute_command_hash(command_template: str, prompt_template: str) -> str:
    """Hash the command + prompt templates to detect config changes."""
    key = f"{command_template}|{prompt_template}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ============================================================================
# Relevance Scoring Persistence (SQLite)
# ============================================================================


def get_relevance_db_path() -> Path:
    """Get the path to the relevance scoring SQLite database."""
    config_dir = Path(user_config_dir(CONFIG_APP_NAME))
    return config_dir / RELEVANCE_DB_FILENAME


def _init_relevance_db(db_path: Path) -> None:
    """Create the relevance_scores table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS relevance_scores ("
            "  arxiv_id TEXT NOT NULL,"
            "  interests_hash TEXT NOT NULL,"
            "  score INTEGER NOT NULL,"
            "  reason TEXT NOT NULL,"
            "  created_at TEXT NOT NULL,"
            "  PRIMARY KEY (arxiv_id, interests_hash)"
            ")"
        )


def _load_relevance_score(
    db_path: Path, arxiv_id: str, interests_hash: str
) -> tuple[int, str] | None:
    """Load a cached relevance score if it exists."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT score, reason FROM relevance_scores "
                "WHERE arxiv_id = ? AND interests_hash = ?",
                (arxiv_id, interests_hash),
            ).fetchone()
            return (row[0], row[1]) if row else None
    except sqlite3.Error:
        logger.warning("Failed to load relevance score for %s", arxiv_id, exc_info=True)
        return None


def _save_relevance_score(
    db_path: Path, arxiv_id: str, interests_hash: str, score: int, reason: str
) -> None:
    """Persist a relevance score to the SQLite database."""
    try:
        _init_relevance_db(db_path)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO relevance_scores "
                "(arxiv_id, interests_hash, score, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (arxiv_id, interests_hash, score, reason, datetime.now().isoformat()),
            )
    except sqlite3.Error:
        logger.warning("Failed to save relevance score for %s", arxiv_id, exc_info=True)


def _load_all_relevance_scores(db_path: Path, interests_hash: str) -> dict[str, tuple[int, str]]:
    """Bulk-load all relevance scores for a given interests hash."""
    if not db_path.exists():
        return {}
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT arxiv_id, score, reason FROM relevance_scores WHERE interests_hash = ?",
                (interests_hash,),
            ).fetchall()
            return {row[0]: (row[1], row[2]) for row in rows}
    except sqlite3.Error:
        logger.warning("Failed to bulk-load relevance scores", exc_info=True)
        return {}


# ============================================================================
# Relevance Scoring Prompt & Response Parsing
# ============================================================================

# Pre-compiled patterns for parsing LLM relevance JSON responses
_RELEVANCE_SCORE_RE = re.compile(r'"score"\s*:\s*(\d+)')
_RELEVANCE_REASON_RE = re.compile(r'"reason"\s*:\s*"([^"]+)"')
_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def build_relevance_prompt(paper: Paper, interests: str) -> str:
    """Build a relevance scoring prompt for a paper.

    Uses the RELEVANCE_PROMPT_TEMPLATE, substituting paper fields and interests.
    """
    abstract = paper.abstract or paper.abstract_raw or "(no abstract)"
    return RELEVANCE_PROMPT_TEMPLATE.format(
        title=paper.title,
        authors=paper.authors,
        categories=paper.categories,
        abstract=abstract,
        interests=interests,
    )


def _parse_relevance_response(text: str) -> tuple[int, str] | None:
    """Parse the LLM's relevance scoring response.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Strip markdown fences then JSON parse
    3. Regex fallback for score and reason fields

    Returns (score, reason) tuple or None if parsing fails.
    Score is clamped to 1-10 range.
    """
    stripped = text.strip()

    # Strategy 1: direct JSON parse
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "score" in data:
            score = max(1, min(10, int(data["score"])))
            reason = str(data.get("reason", ""))
            return (score, reason)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Strategy 2: strip markdown fences
    fence_match = _MARKDOWN_FENCE_RE.search(stripped)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1))
            if isinstance(data, dict) and "score" in data:
                score = max(1, min(10, int(data["score"])))
                reason = str(data.get("reason", ""))
                return (score, reason)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 3: regex fallback
    score_match = _RELEVANCE_SCORE_RE.search(stripped)
    reason_match = _RELEVANCE_REASON_RE.search(stripped)
    if score_match:
        score = max(1, min(10, int(score_match.group(1))))
        reason = reason_match.group(1) if reason_match else ""
        return (score, reason)

    return None


# ============================================================================
# Auto-Tagging Prompt & Response Parsing
# ============================================================================

_AUTO_TAG_TAGS_RE = re.compile(r'"tags"\s*:\s*\[([^\]]*)\]')


def build_auto_tag_prompt(paper: Paper, existing_tags: list[str]) -> str:
    """Build an auto-tagging prompt for a paper.

    Uses AUTO_TAG_PROMPT_TEMPLATE, substituting paper fields and existing taxonomy.
    """
    abstract = paper.abstract or paper.abstract_raw or "(no abstract)"
    taxonomy = ", ".join(existing_tags) if existing_tags else "(no existing tags — create new ones)"
    return AUTO_TAG_PROMPT_TEMPLATE.format(
        title=paper.title,
        authors=paper.authors,
        categories=paper.categories,
        abstract=abstract,
        taxonomy=taxonomy,
    )


def _extract_tags_from_json(data: Any) -> list[str] | None:
    """Extract and normalize tags from a parsed JSON object.

    Returns lowercased, stripped tag strings if data contains a valid "tags" list,
    or None otherwise.
    """
    if isinstance(data, dict) and "tags" in data:
        tags = data["tags"]
        if isinstance(tags, list):
            return [str(t).strip().lower() for t in tags if str(t).strip()]
    return None


def _parse_auto_tag_response(text: str) -> list[str] | None:
    """Parse the LLM's auto-tag response.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Strip markdown fences then JSON parse
    3. Regex fallback for tags array

    Returns list of tag strings or None if parsing fails.
    """
    stripped = text.strip()

    # Strategy 1: direct JSON parse
    try:
        result = _extract_tags_from_json(json.loads(stripped))
        if result is not None:
            return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Strategy 2: strip markdown fences
    fence_match = _MARKDOWN_FENCE_RE.search(stripped)
    if fence_match:
        try:
            result = _extract_tags_from_json(json.loads(fence_match.group(1)))
            if result is not None:
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 3: regex fallback
    tags_match = _AUTO_TAG_TAGS_RE.search(stripped)
    if tags_match:
        raw_items = tags_match.group(1)
        items = re.findall(r'"([^"]*)"', raw_items)
        if items:
            return [t.strip().lower() for t in items if t.strip()]

    return None


# ============================================================================
# LLM Prompt Building & Command Resolution
# ============================================================================

_LLM_PROMPT_FIELDS = frozenset(
    {
        "title",
        "authors",
        "categories",
        "abstract",
        "arxiv_id",
        "paper_content",
    }
)


def build_llm_prompt(paper: Paper, prompt_template: str = "", paper_content: str = "") -> str:
    """Build the full prompt by substituting paper data into the template.

    If paper_content is provided, it is included via the {paper_content}
    placeholder.  When the template does not contain that placeholder the
    content is appended automatically so that every prompt benefits from
    full-paper context when available.
    """
    template = prompt_template or DEFAULT_LLM_PROMPT
    abstract = paper.abstract or paper.abstract_raw or "(no abstract)"
    content = paper_content or f"Abstract:\n{abstract}"
    values = {
        "title": paper.title,
        "authors": paper.authors,
        "categories": paper.categories,
        "abstract": abstract,
        "arxiv_id": paper.arxiv_id,
        "paper_content": content,
    }
    try:
        result = template.format(**values)
    except (KeyError, ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid prompt template: {e}. "
            f"Valid placeholders: {', '.join(f'{{{k}}}' for k in sorted(_LLM_PROMPT_FIELDS))}"
        ) from e
    # Auto-append paper content if the template didn't include {paper_content}
    if "{paper_content}" not in template and content:
        result = result + "\n\n" + content
    return result


def _resolve_llm_command(config: UserConfig) -> str:
    """Resolve the LLM command template from config (custom or preset).

    Returns the command template string, or "" if not configured.
    Logs a warning if the preset name is unrecognized.
    """
    if config.llm_command:
        return config.llm_command
    if config.llm_preset:
        if config.llm_preset in LLM_PRESETS:
            return LLM_PRESETS[config.llm_preset]
        valid = ", ".join(sorted(LLM_PRESETS))
        logger.warning("Unknown llm_preset %r. Valid presets: %s", config.llm_preset, valid)
    return ""


def _build_llm_shell_command(command_template: str, prompt: str) -> str:
    """Build the final shell command by substituting the prompt.

    Raises ValueError if the template does not contain {prompt}.
    """
    if "{prompt}" not in command_template:
        raise ValueError(
            f"LLM command template must contain {{prompt}} placeholder, got: {command_template!r}"
        )
    escaped_prompt = shlex.quote(prompt)
    return command_template.replace("{prompt}", escaped_prompt)


__all__ = [
    "AUTO_TAG_PROMPT_TEMPLATE",
    "CHAT_SYSTEM_PROMPT",
    "DEFAULT_LLM_PROMPT",
    "LLM_PRESETS",
    "RELEVANCE_PROMPT_TEMPLATE",
    "SUMMARY_MODES",
    "build_auto_tag_prompt",
    "build_llm_prompt",
    "build_relevance_prompt",
    "get_relevance_db_path",
    "get_summary_db_path",
]
