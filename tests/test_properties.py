"""Property-based tests using Hypothesis.

Verifies mathematical invariants across parsing, export, similarity, query,
and config modules. Each test runs 50 examples in CI, 200 in dev.

Run:
    uv run pytest tests/test_properties.py -v
    uv run pytest tests/test_properties.py -v --hypothesis-seed=0  # reproducible
"""

from __future__ import annotations

import csv
import io

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings

from arxiv_browser.config import _config_to_dict, _dict_to_config
from arxiv_browser.export import (
    escape_bibtex,
    format_paper_as_bibtex,
    format_paper_as_ris,
    format_papers_as_csv,
)
from arxiv_browser.models import (
    SORT_OPTIONS,
    Paper,
    SessionState,
    UserConfig,
)
from arxiv_browser.parsing import clean_latex, normalize_arxiv_id, parse_arxiv_date
from arxiv_browser.query import (
    insert_implicit_and,
    reconstruct_query,
    sort_papers,
    tokenize_query,
)
from arxiv_browser.similarity import (
    TfidfIndex,
    _compute_tf,
    _jaccard_similarity,
    find_similar_papers,
)

# ── Hypothesis profiles ─────────────────────────────────────────────
settings.register_profile("ci", max_examples=50, deadline=None)
settings.register_profile("dev", max_examples=200, deadline=None)
settings.load_profile("ci")

# ── Custom strategies ────────────────────────────────────────────────

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.composite
def arxiv_ids(draw: st.DrawFn) -> str:
    """Generate valid new-style arXiv IDs like '2401.12345'."""
    yy = draw(st.integers(min_value=7, max_value=99))
    mm = draw(st.integers(min_value=1, max_value=12))
    seq = draw(st.integers(min_value=0, max_value=99999))
    return f"{yy:02d}{mm:02d}.{seq:05d}"


@st.composite
def arxiv_dates(draw: st.DrawFn) -> str:
    """Generate dates in arXiv email format: 'Mon, 15 Jan 2024'."""
    day_name = draw(st.sampled_from(_DAYS))
    day_num = draw(st.integers(min_value=1, max_value=28))
    month = draw(st.sampled_from(_MONTHS))
    year = draw(st.integers(min_value=2000, max_value=2030))
    return f"{day_name}, {day_num:02d} {month} {year}"


@st.composite
def papers(draw: st.DrawFn) -> Paper:
    """Generate a Paper with valid structure (abstract_raw prevents HTTP fetches)."""
    aid = draw(arxiv_ids())
    abstract = draw(st.text(min_size=0, max_size=200))
    title = draw(st.text(min_size=1, max_size=100).filter(lambda s: s.strip()))
    authors = draw(st.text(min_size=1, max_size=100).filter(lambda s: s.strip()))
    categories = draw(
        st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz.ABCDEFGHIJKLMNOPQRSTUVWXYZ- "),
            min_size=1,
            max_size=50,
        ).filter(lambda s: s.strip())
    )
    date = draw(arxiv_dates())
    return Paper(
        arxiv_id=aid,
        date=date,
        title=title,
        authors=authors,
        categories=categories,
        comments=draw(st.one_of(st.none(), st.text(max_size=50))),
        abstract=abstract,
        url=f"https://arxiv.org/abs/{aid}",
        abstract_raw=abstract,
    )


_DEFAULT_ELEMENTS = st.text(max_size=10)


def finite_sets(elements: st.SearchStrategy = _DEFAULT_ELEMENTS) -> st.SearchStrategy[set]:
    """Strategy that generates a finite set for Jaccard tests."""
    return st.frozensets(elements, max_size=20).map(set)


# ============================================================================
# Tier 1 — High-value, easy properties
# ============================================================================


class TestCleanLatexProperties:
    """Properties of the LaTeX cleaning function."""

    @given(text=st.text(max_size=300))
    def test_idempotency(self, text: str) -> None:
        """Applying clean_latex twice yields the same result as once."""
        once = clean_latex(text)
        twice = clean_latex(once)
        assert once == twice, f"Not idempotent: {text!r} → {once!r} → {twice!r}"

    @given(text=st.text(max_size=300))
    def test_no_consecutive_spaces(self, text: str) -> None:
        """Output never has consecutive spaces."""
        result = clean_latex(text)
        assert "  " not in result

    @given(text=st.text(max_size=300))
    def test_no_leading_trailing_whitespace(self, text: str) -> None:
        """Output is always stripped."""
        result = clean_latex(text)
        assert result == result.strip()

    @given(text=st.text(max_size=300))
    def test_never_crashes(self, text: str) -> None:
        """clean_latex never raises on arbitrary input."""
        result = clean_latex(text)
        assert isinstance(result, str)


class TestEscapeBibtexProperties:
    """Properties of BibTeX character escaping."""

    @given(text=st.text(alphabet=st.characters(exclude_categories=("Cs",)), max_size=100))
    def test_completeness_after_one_pass(self, text: str) -> None:
        """After escaping, no unescaped special chars remain."""
        result = escape_bibtex(text)
        # Check that bare specials (not preceded by backslash) are gone
        for char in "&%_#":
            # Find occurrences not preceded by backslash
            for i, c in enumerate(result):
                if c == char and (i == 0 or result[i - 1] != "\\"):
                    raise AssertionError(f"Unescaped {char!r} at position {i} in {result!r}")

    @given(text=st.from_regex(r"[a-zA-Z0-9 .,;:!?]+", fullmatch=True))
    def test_plain_text_unchanged(self, text: str) -> None:
        """Text without special chars passes through unmodified."""
        assert escape_bibtex(text) == text


class TestJaccardProperties:
    """Mathematical properties of Jaccard similarity."""

    @given(a=finite_sets(), b=finite_sets())
    def test_range(self, a: set, b: set) -> None:
        """Jaccard similarity is always in [0, 1]."""
        score = _jaccard_similarity(a, b)
        assert 0.0 <= score <= 1.0

    @given(a=finite_sets(), b=finite_sets())
    def test_symmetry(self, a: set, b: set) -> None:
        """J(A, B) == J(B, A)."""
        assert _jaccard_similarity(a, b) == _jaccard_similarity(b, a)

    @given(a=finite_sets())
    def test_identity(self, a: set) -> None:
        """J(A, A) == 1.0 when A is non-empty."""
        if a:
            assert _jaccard_similarity(a, a) == 1.0

    @given(n=st.integers(min_value=1, max_value=50))
    def test_disjoint_is_zero(self, n: int) -> None:
        """Guaranteed-disjoint sets have Jaccard = 0."""
        a = set(range(0, n))
        b = set(range(n, 2 * n))
        assert _jaccard_similarity(a, b) == 0.0


class TestComputeTfProperties:
    """Properties of sublinear term frequency computation."""

    @given(tokens=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=50))
    def test_all_values_at_least_one(self, tokens: list[str]) -> None:
        """TF values are always >= 1.0 (formula: 1 + log(count))."""
        tf = _compute_tf(tokens)
        for v in tf.values():
            assert v >= 1.0

    @given(tokens=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=50))
    def test_keys_equal_unique_tokens(self, tokens: list[str]) -> None:
        """TF dict keys are exactly the unique tokens."""
        tf = _compute_tf(tokens)
        assert set(tf.keys()) == set(tokens)

    def test_empty_returns_empty(self) -> None:
        """Empty token list → empty dict."""
        assert _compute_tf([]) == {}


class TestCosineSimProperties:
    """Properties of TF-IDF cosine similarity."""

    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_symmetry(self, data: st.DataObject) -> None:
        """cos(A, B) == cos(B, A)."""
        p1 = data.draw(papers())
        p2 = data.draw(papers())
        idx = TfidfIndex.build([p1, p2], lambda p: p.abstract or "")
        assert idx.cosine_similarity(p1.arxiv_id, p2.arxiv_id) == idx.cosine_similarity(
            p2.arxiv_id, p1.arxiv_id
        )

    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_range(self, data: st.DataObject) -> None:
        """Cosine similarity is in [0, 1]."""
        p1 = data.draw(papers())
        p2 = data.draw(papers())
        idx = TfidfIndex.build([p1, p2], lambda p: p.abstract or "")
        score = idx.cosine_similarity(p1.arxiv_id, p2.arxiv_id)
        assert 0.0 <= score <= 1.0


# ============================================================================
# Tier 2 — Round-trip properties
# ============================================================================


class TestTokenizeQueryRoundTrip:
    """Round-trip and structural properties of query tokenization."""

    @given(query=st.text(alphabet=st.characters(categories=("L", "N", "Zs")), max_size=80))
    def test_all_tokens_are_term_or_op(self, query: str) -> None:
        """Every token has kind 'term' or 'op'."""
        tokens = tokenize_query(query)
        for tok in tokens:
            assert tok.kind in ("term", "op"), f"Unexpected kind: {tok.kind}"

    @given(query=st.from_regex(r"[a-zA-Z]{1,10}( (AND|OR) [a-zA-Z]{1,10}){0,3}", fullmatch=True))
    def test_reconstruct_preserves_terms(self, query: str) -> None:
        """Re-tokenizing a reconstructed query preserves term values."""
        tokens = tokenize_query(query)
        if len(tokens) < 2:
            return
        # Reconstruct by excluding a non-existent index (keeps all tokens)
        rebuilt = reconstruct_query(tokens, exclude_index=len(tokens) + 100)
        re_tokens = tokenize_query(rebuilt)
        original_terms = [t.value for t in tokens if t.kind == "term"]
        rebuilt_terms = [t.value for t in re_tokens if t.kind == "term"]
        assert original_terms == rebuilt_terms

    @given(query=st.from_regex(r"[a-zA-Z]{1,10}( (AND|OR) [a-zA-Z]{1,10}){1,3}", fullmatch=True))
    def test_removing_term_reduces_count(self, query: str) -> None:
        """Removing a term via reconstruct_query reduces token count."""
        tokens = tokenize_query(query)
        term_indices = [i for i, t in enumerate(tokens) if t.kind == "term"]
        if not term_indices:
            return
        idx = term_indices[0]
        rebuilt = reconstruct_query(tokens, exclude_index=idx)
        new_tokens = tokenize_query(rebuilt)
        new_terms = [t for t in new_tokens if t.kind == "term"]
        old_terms = [t for t in tokens if t.kind == "term"]
        assert len(new_terms) < len(old_terms)


class TestInsertImplicitAndProperties:
    """Properties of implicit AND insertion."""

    @given(query=st.text(alphabet=st.characters(categories=("L", "N", "Zs")), max_size=80))
    def test_idempotency(self, query: str) -> None:
        """Inserting implicit ANDs twice is the same as once."""
        tokens = tokenize_query(query)
        once = insert_implicit_and(tokens)
        twice = insert_implicit_and(once)
        assert [(t.kind, t.value) for t in once] == [(t.kind, t.value) for t in twice]


class TestNormalizeArxivIdProperties:
    """Properties of arXiv ID normalization."""

    @given(aid=arxiv_ids())
    def test_idempotency_on_valid_ids(self, aid: str) -> None:
        """Normalizing a valid ID twice yields the same result."""
        once = normalize_arxiv_id(aid)
        twice = normalize_arxiv_id(once)
        assert once == twice

    @given(aid=arxiv_ids(), version=st.integers(min_value=1, max_value=99))
    def test_version_stripped(self, aid: str, version: int) -> None:
        """Versioned ID normalizes to base ID."""
        versioned = f"{aid}v{version}"
        assert normalize_arxiv_id(versioned) == aid

    @given(aid=arxiv_ids())
    def test_abs_url_normalized(self, aid: str) -> None:
        """abs URL normalizes to bare ID."""
        url = f"https://arxiv.org/abs/{aid}"
        assert normalize_arxiv_id(url) == aid

    @given(aid=arxiv_ids(), version=st.integers(min_value=1, max_value=99))
    def test_pdf_url_normalized(self, aid: str, version: int) -> None:
        """PDF URL with version normalizes to bare ID."""
        url = f"https://arxiv.org/pdf/{aid}v{version}.pdf"
        assert normalize_arxiv_id(url) == aid


# ============================================================================
# Tier 3 — Format / structural properties
# ============================================================================


class TestBibtexFormatProperties:
    """Structural properties of BibTeX output."""

    @given(paper=papers())
    def test_starts_with_misc(self, paper: Paper) -> None:
        """BibTeX entry starts with @misc{."""
        bib = format_paper_as_bibtex(paper)
        assert bib.startswith("@misc{")

    @given(paper=papers())
    def test_balanced_braces(self, paper: Paper) -> None:
        """BibTeX entry has balanced unescaped curly braces."""
        bib = format_paper_as_bibtex(paper)
        # Remove escaped braces before counting (escape_bibtex produces \{ and \})
        unescaped = bib.replace(r"\{", "").replace(r"\}", "")
        assert unescaped.count("{") == unescaped.count("}")

    @given(paper=papers())
    def test_required_fields_present(self, paper: Paper) -> None:
        """BibTeX entry contains title, author, year, eprint fields."""
        bib = format_paper_as_bibtex(paper)
        for field in ("title", "author", "year", "eprint"):
            assert f"  {field} = " in bib, f"Missing field: {field}"


class TestRisFormatProperties:
    """Structural properties of RIS output."""

    @given(paper=papers())
    def test_starts_with_type(self, paper: Paper) -> None:
        """RIS entry starts with TY  - ELEC."""
        ris = format_paper_as_ris(paper)
        assert ris.startswith("TY  - ELEC")

    @given(paper=papers())
    def test_ends_with_terminator(self, paper: Paper) -> None:
        """RIS entry ends with ER  -."""
        ris = format_paper_as_ris(paper)
        assert ris.strip().endswith("ER  -")


class TestCsvFormatProperties:
    """Structural properties of CSV output."""

    @given(paper_list=st.lists(papers(), min_size=1, max_size=10))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_parseable_by_csv_reader(self, paper_list: list[Paper]) -> None:
        """CSV output is parseable by Python's csv.reader."""
        csv_text = format_papers_as_csv(paper_list)
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) >= 2  # header + at least one data row

    @given(paper_list=st.lists(papers(), min_size=1, max_size=10, unique_by=lambda p: p.arxiv_id))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_row_count(self, paper_list: list[Paper]) -> None:
        """Row count equals unique papers + 1 (header)."""
        csv_text = format_papers_as_csv(paper_list)
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == len(paper_list) + 1


class TestSortPapersProperties:
    """Properties of the paper sorting function."""

    @given(paper_list=st.lists(papers(), min_size=0, max_size=15))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_length_preserved(self, paper_list: list[Paper]) -> None:
        """Sorting preserves list length."""
        sorted_list = sort_papers(paper_list, "title")
        assert len(sorted_list) == len(paper_list)

    @given(paper_list=st.lists(papers(), min_size=0, max_size=15))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_element_set_preserved(self, paper_list: list[Paper]) -> None:
        """Sorting preserves the set of arxiv_ids."""
        sorted_list = sort_papers(paper_list, "title")
        assert {p.arxiv_id for p in sorted_list} == {p.arxiv_id for p in paper_list}

    @given(paper_list=st.lists(papers(), min_size=2, max_size=15))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_title_sort_is_ordered(self, paper_list: list[Paper]) -> None:
        """Title sort produces alphabetical order (case-insensitive)."""
        sorted_list = sort_papers(paper_list, "title")
        titles = [p.title.lower() for p in sorted_list]
        assert titles == sorted(titles)


class TestFindSimilarProperties:
    """Properties of the similarity recommendation engine."""

    @given(data=st.data())
    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=30,
    )
    def test_no_self_inclusion(self, data: st.DataObject) -> None:
        """Target paper is never in its own recommendations."""
        target = data.draw(papers())
        others = data.draw(st.lists(papers(), min_size=1, max_size=5))
        all_papers = [target, *others]
        results = find_similar_papers(target, all_papers)
        result_ids = {p.arxiv_id for p, _ in results}
        assert target.arxiv_id not in result_ids

    @given(data=st.data())
    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=30,
    )
    def test_scores_in_range(self, data: st.DataObject) -> None:
        """All similarity scores are in [0, 1]."""
        target = data.draw(papers())
        others = data.draw(st.lists(papers(), min_size=1, max_size=5))
        all_papers = [target, *others]
        results = find_similar_papers(target, all_papers)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    @given(data=st.data())
    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=30,
    )
    def test_sorted_descending(self, data: st.DataObject) -> None:
        """Results are sorted by score in descending order."""
        target = data.draw(papers())
        others = data.draw(st.lists(papers(), min_size=2, max_size=8))
        all_papers = [target, *others]
        results = find_similar_papers(target, all_papers)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    @given(data=st.data(), top_n=st.integers(min_value=1, max_value=5))
    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=30,
    )
    def test_length_bounded(self, data: st.DataObject, top_n: int) -> None:
        """Result length never exceeds top_n."""
        target = data.draw(papers())
        others = data.draw(st.lists(papers(), min_size=1, max_size=10))
        all_papers = [target, *others]
        results = find_similar_papers(target, all_papers, top_n=top_n)
        assert len(results) <= top_n


# ============================================================================
# Tier 4 — No-crash (robustness) properties
# ============================================================================


class TestNoCrashProperties:
    """Verify functions never crash on arbitrary input."""

    @given(text=st.text(max_size=500))
    def test_clean_latex_never_crashes(self, text: str) -> None:
        result = clean_latex(text)
        assert isinstance(result, str)

    @given(date_str=st.text(max_size=100))
    def test_parse_arxiv_date_never_crashes(self, date_str: str) -> None:
        result = parse_arxiv_date(date_str)
        from datetime import datetime

        assert isinstance(result, datetime)

    @given(raw=st.text(max_size=200))
    def test_normalize_arxiv_id_never_crashes(self, raw: str) -> None:
        result = normalize_arxiv_id(raw)
        assert isinstance(result, str)

    @given(query=st.text(max_size=200))
    def test_tokenize_query_never_crashes(self, query: str) -> None:
        result = tokenize_query(query)
        assert isinstance(result, list)

    @given(text=st.text(max_size=200))
    def test_escape_bibtex_never_crashes(self, text: str) -> None:
        result = escape_bibtex(text)
        assert isinstance(result, str)


# ============================================================================
# Config round-trip properties (Item 9)
# ============================================================================


class TestConfigRoundTripProperties:
    """Verify the config serialization contract: any input → valid output."""

    def test_scalar_round_trip(self) -> None:
        """A default UserConfig survives config_to_dict → dict_to_config."""
        config = UserConfig()
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert restored.session.sort_index == config.session.sort_index
        assert restored.theme_name == config.theme_name
        assert restored.arxiv_api_max_results == config.arxiv_api_max_results

    @given(max_results=st.integers(min_value=-1000, max_value=1000))
    def test_max_results_clamped(self, max_results: int) -> None:
        """Any integer max_results is clamped to [1, 200] after deserialization."""
        data = {"arxiv_api_max_results": max_results}
        config = _dict_to_config(data)
        assert 1 <= config.arxiv_api_max_results <= 200

    @given(sort_index=st.integers(min_value=-100, max_value=100))
    def test_sort_index_clamped(self, sort_index: int) -> None:
        """Any integer sort_index is clamped to valid SORT_OPTIONS range."""
        data = {"session": {"sort_index": sort_index}}
        config = _dict_to_config(data)
        assert 0 <= config.session.sort_index < len(SORT_OPTIONS)

    @given(
        data=st.dictionaries(
            keys=st.text(max_size=30),
            values=st.one_of(
                st.none(),
                st.booleans(),
                st.integers(min_value=-100, max_value=100),
                st.text(max_size=50),
                st.lists(st.text(max_size=20), max_size=5),
            ),
            max_size=20,
        )
    )
    def test_never_crashes(self, data: dict) -> None:
        """Arbitrary dict input to _dict_to_config never raises."""
        config = _dict_to_config(data)
        assert isinstance(config, UserConfig)
        # Verify the critical invariant: sort_index is always valid
        assert 0 <= config.session.sort_index < len(SORT_OPTIONS)


class TestSessionStatePostInit:
    """Verify SessionState.__post_init__ defense-in-depth."""

    @given(sort_index=st.integers(min_value=-100, max_value=100))
    def test_direct_construction_clamps(self, sort_index: int) -> None:
        """SessionState clamps sort_index on direct construction."""
        state = SessionState(sort_index=sort_index)
        assert 0 <= state.sort_index < len(SORT_OPTIONS)

    def test_valid_indices_preserved(self) -> None:
        """Valid sort indices are not modified."""
        for i in range(len(SORT_OPTIONS)):
            state = SessionState(sort_index=i)
            assert state.sort_index == i
