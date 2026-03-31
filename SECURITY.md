# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| Latest (`main`) | ✅ |
| `0.1.x` | ✅ |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/NicolasSchuler/arxiv-subscription-viewer/security/advisories/new) of this repository.
2. Click **"Report a vulnerability"**.
3. Fill in the details: affected versions, reproduction steps, and potential impact.

You can also reach the maintainer directly via the email listed on the [GitHub profile](https://github.com/NicolasSchuler).

## What to Expect

- **Acknowledgement** within 7 days of your report.
- **Status update** (confirmed, rejected, or need more info) within 14 days.
- **Patch and coordinated disclosure** within 30 days of confirmation for critical issues (or longer for complex cases — you'll be kept informed).

## Scope

This project is a local terminal application. Relevant security concerns include:

- **Subprocess injection** via LLM command templates or user-controlled input.
- **XML/HTML parsing vulnerabilities** (the project uses `defusedxml`).
- **Insecure handling of config files or API keys** stored in the user's config directory.
- **Path traversal** in file-loading or export paths.

Out of scope: denial-of-service against the user's own machine, issues in upstream dependencies (report those to the relevant project).

## Disclosure Policy

Once a fix is ready, we will:

1. Publish a patched release to PyPI.
2. Open a GitHub Security Advisory crediting the reporter (unless you prefer to remain anonymous).
3. Add a note to `CHANGELOG.md`.
