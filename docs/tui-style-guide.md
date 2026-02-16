# TUI/CLI Style Guide

This style guide defines copy, layout, and interaction conventions for arXiv Subscription Viewer.

## 1. Purpose And Audiences

- Purpose: Keep the app discoverable for newcomers while preserving high-speed keyboard workflows for frequent users.
- Primary audience: Intermediate and power users who live in terminal workflows.
- Secondary audience: First-time users who need clear guidance without reading full docs first.

## 2. Voice And Copy Rules

- Prefer clear verbs: `Open`, `Search`, `Export`, `Download`, `Toggle`, `Manage`.
- Keep footer labels short and scannable (1 word when possible).
- Avoid blame language in errors; describe what happened and what to do next.
- Use sentence case for longer help descriptions.
- Keep abbreviations that are common in context (`API`, `S2`, `HF`), but expand in help text where useful.

## 3. Terminology Canon

- Use `Search` for query input interactions.
- Use `Selection` for multi-paper operations.
- Use `Command palette` for `Ctrl+p`.
- Use `Help overlay` for `?`.
- Use `Semantic Scholar` in help text, `S2` in compact status/footer hints.
- Use `History` for date navigation context.

## 4. Layout Hierarchy Rules

- Preserve the three-zone structure:
- Header: context and active dataset/date.
- Content: list + details with selection/focus state.
- Footer: immediate action hints and mode badges.
- Narrow-width behavior:
- Prefer compact status tokens and shorter labels before removing high-value hints.
- Keep `? help` visible in all contexts.

## 5. Color And Icon Accessibility Rules

- Never rely on color alone for meaning; pair with text or symbol.
- Ensure all key states remain readable with `--color never`.
- Ensure status/list indicators remain readable with `--ascii`.
- Keep color use semantic:
- Accent for interactive controls and key hints.
- Green/yellow/orange/pink for status meaning, not decoration.

## 6. Message Templates

- Error template:
- `Could not <action>.`
- `Why: <short reason if known>.`
- `Next step: <specific recovery action>.`
- Success template:
- `<Action complete>.`
- Optional detail: `<count/location/context>.`
- Progress template:
- `<Action> <progress-bar> <done>/<total>`
- Empty-state template:
- `No <entity> found.`
- `Try: <next command or filter adjustment>.`

## 7. Tables, Lists, Truncation, Wrapping

- Keep list rows stable: title, authors, metadata badges, optional abstract preview.
- Favor truncation with ellipsis over hard wrapping for dense list rows.
- Preserve key metadata visibility under width pressure:
- arXiv ID, category, and high-value badges (for example relevance/version).
- In compact status mode, drop lowest-priority tokens first and keep core context.

## 8. Help And Footer Discoverability

- Help overlay must include a top `Getting Started` section with the core flow:
- Search, move, select, open, command palette, full help.
- Footer should prioritize immediate next actions for current mode.
- Use consistent labels across footer/help/notifications:
- `palette`, `history`, `help`, `search`, `open`, `export`.
- Keep close instructions concise in modals (for example `Close: ? / Esc / q`).

## 9. PR Checklist

- [ ] New or changed UI copy uses the terminology canon.
- [ ] At least one explicit next-step hint exists in each interaction context.
- [ ] Error text includes actionable recovery guidance.
- [ ] Footer hints remain short and consistent with help overlay wording.
- [ ] Non-color/ASCII compatibility remains intact.
- [ ] Tests cover changed help/footer strings.
- [ ] No keybinding behavior changed unless explicitly intended.

## 10. UI Direction Options

### Minimalist Direction

- Keep status/footer ultra-compact.
- Emphasize list/detail reading flow.
- Show only core actions by default; keep advanced actions in palette/help.
- Tradeoff: Lowest cognitive load, but advanced capabilities are less visible.

### Feature-Rich Direction

- Surface more contextual hints and enrichment states inline.
- Keep advanced features visible in footer/status with short labels.
- Tradeoff: Higher discoverability for power features, but denser visual load.
