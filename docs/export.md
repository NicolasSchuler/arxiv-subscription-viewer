# 📥 Export, PDF & Figure Preview

## Export Menu (`E`)

Press `E` to open the export menu. Supported formats:

| Format | Clipboard | File |
|--------|-----------|------|
| BibTeX | ✅ (`b`) | ✅ (`B`, Zotero-compatible) |
| RIS | ✅ (`r`) | ✅ (`R`) |
| CSV | ✅ (`v`) | ✅ (`C`) |
| Markdown | ✅ (`m`) | — |
| Markdown table | ✅ (`t`) | — |
| Plain text | ✅ (`c`) | — |

Markdown, Markdown-table, and plain-text exports are clipboard-only. File exports (BibTeX, RIS, CSV) are written to disk.

File exports default to `~/arxiv-exports/`. Configure a custom directory:

```json
{
  "bibtex_export_dir": "/path/to/exports"
}
```

## PDF Downloads (`d`)

Press `d` to download PDFs for selected papers (or current paper):

- Default location: `~/arxiv-pdfs/`
- Already-downloaded files are skipped
- Progress shown in status bar
- Supports batch downloads with multi-select
- Currently supports arXiv papers only; non-arXiv provider records are skipped with a warning
- Verifies downloaded bytes look like a PDF before replacing an existing file

```json
{
  "pdf_download_dir": "/path/to/pdfs"
}
```

## PDF Preview (`F`)

Press `F` to render a terminal preview of the current paper's PDF. If the PDF is not already downloaded, the app downloads it first when an HTTP client is active.

- Renders with default MIT-compatible/permissive dependencies (`pypdfium2` and `Pillow`)
- Displays with an in-app half-block renderer, with ASCII fallback under `--ascii` / no-color modes
- Caches preview PNGs next to the configured PDF directory in `.preview-cache/`
- Limits preview length with `pdf_preview_max_pages` (default `3`)
- Currently supports arXiv papers only

```json
{
  "pdf_preview_max_pages": 5
}
```

## HTML Figure Preview (`I`)

Press `I` to preview the first figure from the current paper's arXiv HTML page.

- Fetches `https://arxiv.org/html/{arxiv_id}` and selects the first LaTeXML paper figure (`<figure class="ltx_figure"><img ...>`)
- Ignores header/logo images outside paper figures and resolves relative image URLs against arXiv
- Caches the normalized figure image next to the paper download/cache area in `.figure-cache/`
- Uses the same terminal-safe renderer as PDF preview, with half-block Unicode output by default and ASCII-safe output under `--ascii` / no-color modes
- Currently supports arXiv papers only
- Shows a warning, without changing paper state, when arXiv HTML is unavailable, no figure is present, image bytes are invalid, the image type is unsupported, or the network request fails

## PDF Viewer (`P`)

By default, `P` opens PDFs in your browser. Configure a custom viewer:

```json
{
  "pdf_viewer": "zathura {url}"
}
```

`{url}` and `{path}` are interchangeable and both receive the paper's PDF URL — `P` opens the remote PDF in your viewer, not a downloaded file. If neither placeholder is present, the URL is appended as the final argument. On first use, the app asks for trust confirmation. (To open an already-downloaded local file, use `d` to download, then your own tools.)

When `prefer_pdf_url` is enabled, non-arXiv provider records still open their canonical provider URL instead of constructing an arXiv PDF URL.

## Metadata Export & Import

Use the command palette (`Ctrl+p`) to back up or migrate your annotations:

- **Export Metadata** — writes a timestamped JSON snapshot to `~/arxiv-exports/`
- **Import Metadata** — restores from a JSON snapshot (notes, tags, read/star state, review schedule, line annotations, watch list, tracked authors, bookmarks, collections, research interests)

## Collections (`Ctrl+k`)

Organize papers into reading lists:

- Create, rename, delete collections and edit descriptions
- Add papers from command palette (`Ctrl+p`)
- Export individual collections as Markdown

---

**See also:** [Config Reference](config-reference.md) · [Search & Filters](search-filters.md) · [Troubleshooting](troubleshooting.md) · [All Docs](README.md)
