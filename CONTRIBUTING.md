# Contributing

This repository keeps contributor guidance lightweight: prefer small, task-focused changes over adding new planning or process documents.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. To report a security issue, see [SECURITY.md](SECURITY.md).

Start with [docs/architecture.md](docs/architecture.md). It is the primary human-oriented reference for package boundaries, canonical imports, and test patch rules.

## Docs map

- `README.md` — top-level install, usage, and quick-start material
- `docs/README.md` — landing page for user-facing guides
- `docs/*.md` — shipped feature guides and reference docs for end users
- `docs/architecture.md` — primary contributor architecture and import-boundary guide
- `AGENTS.md`, `CLAUDE.md`, and `docs/tui-style-guide.md` — supplemental maintainer/developer guidance

## Public vs. internal docs

- **Public docs** should describe shipped behavior, supported workflows, and user-visible commands/options.
- **Internal docs** can cover implementation details, maintainer workflows, and quality gates.
- Treat `docs/plans/` and `docs/*-prompt.md` as internal hygiene territory. Do not add to or update those files for normal user-facing doc work unless a task explicitly asks for maintainer-only material.
- Avoid creating new planning, status, or tracking docs in the repo unless the task explicitly requires them.

## Checks for docs changes

- If you change docs that mention CLI flags, subcommands, keybindings, presets, or completions-sensitive text, run:
  - `just docs-check`
- If your docs change accompanies Python code changes, run the normal baseline too:
  - `just check`

For doc-only edits like this guide or `AGENTS.md`, keep validation targeted instead of running the full suite unless the change depends on code behavior updates.
