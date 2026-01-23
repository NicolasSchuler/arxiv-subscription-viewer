# PDF Download Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add async background PDF downloading with progress tracking to the arXiv browser TUI.

**Architecture:** Add `httpx` for async HTTP, implement download queue with concurrency limit (3), track progress in status bar, store PDFs in configurable directory (`~/arxiv-pdfs/` by default).

**Tech Stack:** Python 3.13, Textual, httpx (new dependency)

---

### Task 1: Add httpx Dependency

**Files:**
- Modify: `pyproject.toml:12-15`

**Step 1: Add httpx to dependencies**

Edit `pyproject.toml` dependencies list:

```toml
dependencies = [
    "textual>=7.3.0",
    "rapidfuzz>=3.0.0",
    "httpx>=0.27.0",
]
```

**Step 2: Install dependencies**

Run: `uv sync`
Expected: Dependencies install successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add httpx for async PDF downloads"
```

---

### Task 2: Add Constants and Config Field

**Files:**
- Modify: `arxiv_browser.py:160-168` (constants section)
- Modify: `arxiv_browser.py:421` (UserConfig)
- Test: `test_arxiv_browser.py`

**Step 1: Write test for config serialization roundtrip**

Add to `test_arxiv_browser.py` after other config tests:

```python
class TestPdfDownloadConfig:
    """Tests for PDF download configuration."""

    def test_pdf_download_dir_default_empty(self):
        """Default pdf_download_dir should be empty string."""
        from arxiv_browser import UserConfig
        config = UserConfig()
        assert config.pdf_download_dir == ""

    def test_pdf_download_dir_serialization_roundtrip(self):
        """pdf_download_dir should survive config serialization."""
        from arxiv_browser import UserConfig, _config_to_dict, _dict_to_config
        config = UserConfig(pdf_download_dir="/custom/path")
        data = _config_to_dict(config)
        restored = _dict_to_config(data)
        assert restored.pdf_download_dir == "/custom/path"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test_arxiv_browser.py::TestPdfDownloadConfig -v`
Expected: FAIL - `pdf_download_dir` not defined

**Step 3: Add constants and config field**

Add after line 165 (after `MAX_HISTORY_FILES`):

```python
# PDF download settings
DEFAULT_PDF_DOWNLOAD_DIR = "arxiv-pdfs"  # Relative to home directory
PDF_DOWNLOAD_TIMEOUT = 60  # Seconds per download
MAX_CONCURRENT_DOWNLOADS = 3  # Limit parallel downloads
```

Add to `UserConfig` dataclass after `bibtex_export_dir` (around line 421):

```python
    pdf_download_dir: str = ""  # Empty = use ~/arxiv-pdfs/
```

Add to `_config_to_dict()` after `bibtex_export_dir` (around line 453):

```python
        "pdf_download_dir": config.pdf_download_dir,
```

Add to `_dict_to_config()` after `bibtex_export_dir` (around line 594):

```python
        pdf_download_dir=_safe_get(data, "pdf_download_dir", "", str),
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest test_arxiv_browser.py::TestPdfDownloadConfig -v`
Expected: PASS

**Step 5: Commit**

```bash
git add arxiv_browser.py test_arxiv_browser.py
git commit -m "feat: add PDF download constants and config field"
```

---

### Task 3: Add Download Path Helper

**Files:**
- Modify: `arxiv_browser.py` (add helper method to ArxivBrowser)
- Test: `test_arxiv_browser.py`

**Step 1: Write test for download path helper**

Add to `TestPdfDownloadConfig` class:

```python
    def test_get_pdf_download_path_default(self, tmp_path, monkeypatch):
        """Default path should be ~/arxiv-pdfs/{arxiv_id}.pdf."""
        from arxiv_browser import Paper, UserConfig, DEFAULT_PDF_DOWNLOAD_DIR
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        paper = Paper(
            arxiv_id="2301.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="Test Author",
            categories="cs.AI",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2301.12345",
        )
        config = UserConfig()

        # Import the function we'll create
        from arxiv_browser import get_pdf_download_path
        path = get_pdf_download_path(paper, config)

        assert path == tmp_path / DEFAULT_PDF_DOWNLOAD_DIR / "2301.12345.pdf"

    def test_get_pdf_download_path_custom_dir(self, tmp_path):
        """Custom dir should be used when configured."""
        from arxiv_browser import Paper, UserConfig, get_pdf_download_path

        paper = Paper(
            arxiv_id="2301.12345",
            date="Mon, 15 Jan 2024",
            title="Test Paper",
            authors="Test Author",
            categories="cs.AI",
            comments=None,
            abstract=None,
            url="https://arxiv.org/abs/2301.12345",
        )
        config = UserConfig(pdf_download_dir=str(tmp_path / "my-pdfs"))

        path = get_pdf_download_path(paper, config)
        assert path == tmp_path / "my-pdfs" / "2301.12345.pdf"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test_arxiv_browser.py::TestPdfDownloadConfig::test_get_pdf_download_path_default -v`
Expected: FAIL - `get_pdf_download_path` not defined

**Step 3: Implement helper function**

Add after `discover_history_files()` function (around line 846), as a module-level function:

```python
def get_pdf_download_path(paper: Paper, config: UserConfig) -> Path:
    """Get the local file path for a downloaded PDF.

    Args:
        paper: The paper to get the download path for.
        config: User configuration with optional custom download directory.

    Returns:
        Path where the PDF should be saved.
    """
    if config.pdf_download_dir:
        base_dir = Path(config.pdf_download_dir)
    else:
        base_dir = Path.home() / DEFAULT_PDF_DOWNLOAD_DIR
    return base_dir / f"{paper.arxiv_id}.pdf"
```

Add to `__all__` list (around line 116):

```python
    "get_pdf_download_path",
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest test_arxiv_browser.py::TestPdfDownloadConfig -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add arxiv_browser.py test_arxiv_browser.py
git commit -m "feat: add get_pdf_download_path helper function"
```

---

### Task 4: Add httpx Import and Download State

**Files:**
- Modify: `arxiv_browser.py` (imports and ArxivBrowser.__init__)

**Step 1: Add httpx import**

Add after `from rapidfuzz import fuzz` (around line 69):

```python
import httpx
```

**Step 2: Add download state to ArxivBrowser.__init__**

Add after `self._pending_mark_action` (around line 2215):

```python
        # PDF download state
        self._download_queue: deque[Paper] = deque()
        self._downloading: set[str] = set()  # arxiv_ids currently downloading
        self._download_results: dict[str, bool] = {}  # arxiv_id -> success
        self._download_total: int = 0  # Total papers in current batch
```

**Step 3: Verify app still starts**

Run: `uv run python arxiv_browser.py --help`
Expected: Help text displays without import errors

**Step 4: Commit**

```bash
git add arxiv_browser.py
git commit -m "feat: add httpx import and download state"
```

---

### Task 5: Implement Core Download Method

**Files:**
- Modify: `arxiv_browser.py` (add _download_pdf_async method)

**Step 1: Add async download method**

Add after `_get_pdf_url` method (around line 3559):

```python
    async def _download_pdf_async(self, paper: Paper) -> bool:
        """Download a single PDF asynchronously.

        Args:
            paper: The paper to download.

        Returns:
            True if download succeeded, False otherwise.
        """
        url = self._get_pdf_url(paper)
        path = get_pdf_download_path(paper, self._config)

        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    timeout=PDF_DOWNLOAD_TIMEOUT,
                    follow_redirects=True,
                )
                response.raise_for_status()
                path.write_bytes(response.content)
            logger.debug(f"Downloaded PDF for {paper.arxiv_id} to {path}")
            return True
        except (httpx.HTTPError, OSError) as e:
            logger.debug(f"Download failed for {paper.arxiv_id}: {e}")
            return False
```

**Step 2: Verify no syntax errors**

Run: `uv run python -c "import arxiv_browser"`
Expected: No errors

**Step 3: Commit**

```bash
git add arxiv_browser.py
git commit -m "feat: add _download_pdf_async method"
```

---

### Task 6: Implement Download Queue Processing

**Files:**
- Modify: `arxiv_browser.py` (add queue processing methods)

**Step 1: Add queue processing methods**

Add after `_download_pdf_async` method:

```python
    def _start_downloads(self) -> None:
        """Start download tasks up to the concurrency limit."""
        while self._download_queue and len(self._downloading) < MAX_CONCURRENT_DOWNLOADS:
            paper = self._download_queue.popleft()
            if paper.arxiv_id in self._downloading:
                continue
            self._downloading.add(paper.arxiv_id)
            asyncio.create_task(self._process_single_download(paper))

    async def _process_single_download(self, paper: Paper) -> None:
        """Process a single download and update state."""
        success = await self._download_pdf_async(paper)
        self._download_results[paper.arxiv_id] = success
        self._downloading.discard(paper.arxiv_id)

        # Update progress
        completed = len(self._download_results)
        total = self._download_total
        self._update_download_progress(completed, total)

        # Start more downloads if queue has items
        self._start_downloads()

        # Check if batch is complete
        if completed == total:
            self._finish_download_batch()

    def _update_download_progress(self, completed: int, total: int) -> None:
        """Update status bar with download progress."""
        try:
            status_bar = self.query_one("#status-bar", Label)
            status_bar.update(f"Downloading: {completed}/{total} complete")
        except NoMatches:
            pass

    def _finish_download_batch(self) -> None:
        """Handle completion of a download batch."""
        successes = sum(1 for v in self._download_results.values() if v)
        failures = len(self._download_results) - successes

        # Get download directory for notification
        if self._config.pdf_download_dir:
            download_dir = self._config.pdf_download_dir
        else:
            download_dir = f"~/{DEFAULT_PDF_DOWNLOAD_DIR}"

        if failures == 0:
            self.notify(
                f"Downloaded {successes} PDF{'s' if successes != 1 else ''} to {download_dir}",
                title="Download Complete",
            )
        else:
            self.notify(
                f"Downloaded {successes}/{self._download_total} PDFs ({failures} failed)",
                title="Download Complete",
                severity="warning",
            )

        # Reset state
        self._download_results.clear()
        self._download_total = 0
        self._update_status_bar()
```

**Step 2: Verify no syntax errors**

Run: `uv run python -c "import arxiv_browser"`
Expected: No errors

**Step 3: Commit**

```bash
git add arxiv_browser.py
git commit -m "feat: add download queue processing methods"
```

---

### Task 7: Add Key Binding and Action

**Files:**
- Modify: `arxiv_browser.py` (BINDINGS and action method)

**Step 1: Add key binding**

Add to `BINDINGS` list after the BibTeX bindings (around line 2148):

```python
        Binding("d", "download_pdf", "Download"),
```

**Step 2: Add action method**

Add after `action_open_pdf` method (around line 3594):

```python
    def action_download_pdf(self) -> None:
        """Download PDFs for selected papers (or current paper)."""
        # Collect papers to download
        papers_to_download: list[Paper] = []

        if self.selected_ids:
            for arxiv_id in self.selected_ids:
                paper = self._get_paper_by_id(arxiv_id)
                if paper:
                    papers_to_download.append(paper)
        else:
            details = self.query_one(PaperDetails)
            if details.paper:
                papers_to_download.append(details.paper)

        if not papers_to_download:
            self.notify("No papers to download", title="Download", severity="warning")
            return

        # Filter out already downloaded
        to_download: list[Paper] = []
        for paper in papers_to_download:
            path = get_pdf_download_path(paper, self._config)
            if path.exists() and path.stat().st_size > 0:
                logger.debug(f"Skipping {paper.arxiv_id}: already downloaded")
            else:
                to_download.append(paper)

        if not to_download:
            self.notify("All PDFs already downloaded", title="Download")
            return

        # Initialize download batch
        self._download_queue.extend(to_download)
        self._download_total = len(to_download)
        self._download_results.clear()

        # Notify and start downloads
        self.notify(
            f"Downloading {len(to_download)} PDF{'s' if len(to_download) != 1 else ''}...",
            title="Download",
        )
        self._start_downloads()
```

**Step 3: Verify no syntax errors**

Run: `uv run python -c "import arxiv_browser"`
Expected: No errors

**Step 4: Commit**

```bash
git add arxiv_browser.py
git commit -m "feat: add download_pdf action and key binding"
```

---

### Task 8: Update Documentation

**Files:**
- Modify: `arxiv_browser.py` (docstring at top)
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update module docstring**

Add to key bindings list in module docstring (around line 15, after `B` binding):

```python
    d       - Download PDF(s) to local folder
```

**Step 2: Update CLAUDE.md key bindings**

Add to key bindings reference section after `B` line:

```
d       - Download PDF(s) to local folder
```

**Step 3: Update README.md**

Add to features section and key bindings. Add a "PDF Downloads" section:

```markdown
### PDF Downloads

Press `d` to download PDFs for selected papers (or current paper) to your local machine:

- Default location: `~/arxiv-pdfs/`
- Configure custom directory in `config.json` with `pdf_download_dir`
- Already-downloaded files are skipped
- Progress shown in status bar
- Supports batch downloads with multi-select
```

**Step 4: Commit**

```bash
git add arxiv_browser.py CLAUDE.md README.md
git commit -m "docs: document PDF download feature"
```

---

### Task 9: Run Full Test Suite

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass (should be 97+ tests)

**Step 2: Manual smoke test**

Run: `uv run python arxiv_browser.py`

Test:
1. Navigate to a paper
2. Press `d` - should see "Downloading 1 PDF..."
3. Check `~/arxiv-pdfs/` for the downloaded file
4. Press `d` again - should see "All PDFs already downloaded"
5. Select multiple papers with `space`
6. Press `d` - should see batch download with progress

**Step 3: Final commit if any adjustments needed**

```bash
git status
# If clean, no action needed
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add httpx dependency | pyproject.toml |
| 2 | Add constants and config | arxiv_browser.py, tests |
| 3 | Add download path helper | arxiv_browser.py, tests |
| 4 | Add imports and state | arxiv_browser.py |
| 5 | Implement download method | arxiv_browser.py |
| 6 | Implement queue processing | arxiv_browser.py |
| 7 | Add key binding and action | arxiv_browser.py |
| 8 | Update documentation | arxiv_browser.py, CLAUDE.md, README.md |
| 9 | Full test suite | - |

Total: 9 tasks, ~9 commits
