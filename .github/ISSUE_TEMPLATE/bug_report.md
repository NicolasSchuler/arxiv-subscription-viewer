---
name: Bug report
about: Report something that isn't working correctly
labels: bug
---

## Environment

- **arxiv-subscription-viewer version**: (run `arxiv-viewer --version`)
- **Python version**: (run `python --version`)
- **OS**: (e.g. macOS 14, Ubuntu 22.04, Windows 11)
- **Terminal / shell**: (e.g. iTerm2, GNOME Terminal, fish)
- **Diagnostics** (optional but helpful): output of `arxiv-viewer doctor`

## Description

A clear description of the bug.

## Steps to Reproduce

1. ...
2. ...
3. See error

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened. Include the full error message / traceback if available.

<details>
<summary>Debug log (if relevant)</summary>

Run with `arxiv-viewer --debug` and paste the relevant section of the debug log. Run
`arxiv-viewer config-path` to find your config/log directory, or use the per-OS paths:
`~/.config/arxiv-browser/debug.log` (Linux), `~/Library/Application Support/arxiv-browser/debug.log`
(macOS), `%APPDATA%\arxiv-browser\debug.log` (Windows).

```
paste log here
```
</details>

## Additional Context

Any other context, screenshots, or config snippets.
