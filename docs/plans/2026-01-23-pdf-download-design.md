# PDF Download Feature Design

**Date:** 2026-01-23
**Status:** Approved

## Overview

Add async background PDF downloading to the arXiv browser TUI. Users can download PDFs for selected papers (or the current paper) to a local directory with progress tracking.

## User Interaction

**Key binding:** `d` (download)

**Behavior:**
- If papers are selected (multi-select): download all selected PDFs
- If no selection: download the currently highlighted paper's PDF
- Shows immediate notification: "Downloading N PDFs..."
- Status bar updates with progress: "Downloading: 2/3 complete"
- Final notification: "Downloaded 3 PDFs to ~/arxiv-pdfs/"

**Download directory:**
- Default: `~/arxiv-pdfs/` (configurable via `UserConfig.pdf_download_dir`)
- Creates directory if it doesn't exist
- Filename format: `{arxiv_id}.pdf` (e.g., `2301.12345.pdf`)
- Skips already-downloaded files (checks if file exists and has non-zero size)

**Error handling:**
- Network failures: notification with error message, continues with remaining downloads
- Partial success: "Downloaded 2/3 PDFs (1 failed)"

## Technical Architecture

### New Dependency

Add to `pyproject.toml`:
```toml
httpx>=0.27.0
```

### Constants

```python
DEFAULT_PDF_DOWNLOAD_DIR = "arxiv-pdfs"  # Relative to home directory
PDF_DOWNLOAD_TIMEOUT = 60  # Seconds per download
MAX_CONCURRENT_DOWNLOADS = 3  # Limit parallel downloads
```

### UserConfig Extension

```python
@dataclass(slots=True)
class UserConfig:
    # ... existing fields ...
    pdf_download_dir: str = ""  # Empty = use ~/arxiv-pdfs/
```

### ArxivBrowser State

```python
self._download_queue: deque[Paper] = deque()
self._downloading: set[str] = set()  # arxiv_ids currently downloading
self._download_results: dict[str, bool] = {}  # arxiv_id -> success
self._download_total: int = 0  # Total papers in current batch
```

### Helper Functions

- `_get_pdf_download_path(paper: Paper) -> Path` - returns full path for downloaded PDF
- `_download_pdf_async(paper: Paper) -> bool` - async download single PDF
- `_process_download_queue()` - manages concurrent downloads
- `_start_downloads()` - spawns up to MAX_CONCURRENT_DOWNLOADS tasks

## Implementation Details

### Download Flow

1. `action_download_pdf()` is called (user presses `d`)
2. Collect papers to download (selected or current)
3. Filter out already-downloaded papers (file exists check)
4. Add to `_download_queue`, set `_download_total`
5. Show notification: "Downloading N PDFs..."
6. Call `_start_downloads()` which spawns up to `MAX_CONCURRENT_DOWNLOADS` tasks

### Core Async Method

```python
async def _download_pdf_async(self, paper: Paper) -> bool:
    """Download a single PDF. Returns True on success."""
    url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
    path = self._get_pdf_download_path(paper)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=PDF_DOWNLOAD_TIMEOUT, follow_redirects=True)
            response.raise_for_status()
            path.write_bytes(response.content)
        return True
    except (httpx.HTTPError, OSError) as e:
        logger.debug(f"Download failed for {paper.arxiv_id}: {e}")
        return False
```

### Progress Tracking

- After each download completes, update status bar: "Downloading: 2/5 complete"
- When all complete, show final notification with success/failure counts
- Clear download state for next batch

## Testing

### Unit Tests (`test_arxiv_browser.py`)

`TestPdfDownload` class:
- `test_get_pdf_download_path_default` - default ~/arxiv-pdfs/ location
- `test_get_pdf_download_path_custom` - custom config directory
- `test_pdf_url_construction` - verify URL format
- `test_download_skips_existing` - doesn't re-download existing files

### Integration Tests

- Mock `httpx.AsyncClient` to test download flow without network

## Documentation Updates

1. **CLAUDE.md** - Add `d` to key bindings reference
2. **README.md** - Document PDF download feature
3. **arxiv_browser.py** docstring - Add `d` to key bindings list

## Config Serialization

Add `pdf_download_dir` to:
- `_config_to_dict()`
- `_dict_to_config()`

Follow existing pattern used for `bibtex_export_dir`.
