# 📥 Export & PDF Configuration

## Export Menu (`E`)

Press `E` to open the export menu. Supported formats:

| Format | Clipboard | File |
|--------|-----------|------|
| BibTeX | ✅ | ✅ (Zotero-compatible) |
| Markdown | ✅ | ✅ |
| RIS | ✅ | ✅ |
| CSV | — | ✅ |

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

```json
{
  "pdf_preview_max_pages": 5
}
```

## PDF Viewer (`P`)

By default, `P` opens PDFs in your browser. Configure a custom viewer:

```json
{
  "pdf_viewer": "zathura {url}"
}
```

Placeholders: `{url}` or `{path}`. If omitted, the URL is appended as the final argument. On first use, the app asks for trust confirmation.

## Metadata Export & Import

Use the command palette (`Ctrl+p`) to back up or migrate your annotations:

- **Export Metadata** — writes a timestamped JSON snapshot to `~/arxiv-exports/`
- **Import Metadata** — restores from a JSON snapshot (notes, tags, read/star state, watch list, bookmarks, marks, collections)

## Collections (`Ctrl+k`)

Organize papers into reading lists:

- Create, rename, delete collections and edit descriptions
- Add papers from command palette (`Ctrl+p`)
- Export individual collections as Markdown

---

**See also:** [Config Reference](config-reference.md) · [Search & Filters](search-filters.md) · [Troubleshooting](troubleshooting.md) · [All Docs](README.md)
