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
- Use `Semantic Scholar` in help text and `S2` in compact status/footer hints.
- Use `History` for date navigation context.
- Compact-vs-long naming policy:
- Footer stays compact (`o open`, `Ctrl+p palette`) for 80-col scan speed.
- Help/modals/binding descriptions use long-form labels (`Open in Browser`, `Command palette`).
- Canonical `Ctrl+e` wording: `Toggle S2 (browse) / Exit API (API mode)`.

## 4. Layout Hierarchy Rules

- Preserve the three-zone structure:
- Header: context and active dataset/date.
- Content: list + details with selection/focus state.
- Footer: immediate action hints and mode badges.
- Narrow-width behavior:
- Prefer compact status tokens and shorter labels before removing high-value hints.
- Keep `? help` visible in all contexts.

## 5. Footer Hierarchy Rules

- Default browse footer is capped at 10 hints.
- Always include these core hints in order:
- `/ search`, `o open`, `s sort`, `r read`, `x star`, `E export`, `Ctrl+p palette`, `? help`.
- Add exactly two context slots in order:
- Slot A: `[/] history` if history navigation is available, else `n notes`.
- Slot B: `e S2` if S2 is active, else `V versions` if starred papers exist, else `L relevance` if LLM is configured, else `t tags`.
- Search footer should emphasize immediate flow:
- `type to search`, `Enter apply`, `Esc clear`, `↑↓ move`, `? help`.
- API footer should emphasize mode exits and paging:
- `[/] page`, `Esc/Ctrl+e exit`, `A new query`, `o open`, `? help`.
- API empty states must mention both recovery paths:
- Pagination keys (`[` and `]`) and exit keys (`Esc` or `Ctrl+e`).

## 6. Color And Icon Accessibility Rules

- Never rely on color alone for meaning; pair with text or symbol.
- Ensure all key states remain readable with `--color never`.
- Ensure status/list indicators remain readable with `--ascii`.
- Keep color use semantic:
- Accent for interactive controls and key hints.
- Green/yellow/orange/pink for status meaning, not decoration.

## 7. Message Templates

- Error template:
- `Could not <action>.`
- Optional: `Why: <short reason if known>.`
- `Next step: <specific recovery action>.`
- Warning template:
- `<Short warning statement>.`
- Optional: `Why: <short reason if known>.`
- `Next step: <specific recovery action>.`
- Success template:
- `<Action complete>.`
- Optional detail: `<count/location/context>.`
- Optional: `Next step: <optional follow-up action>.`
- Progress template:
- `<Action> <progress-bar> <done>/<total>`
- Confirmation template:
- Body explains impact only; do not embed key legends in the message body.
- Confirm/cancel hints belong in modal chrome (for example `Confirm: y  Cancel: n / Esc`).
- Empty-state template:
- `No <entity> found.`
- `Try: <next command or filter adjustment>.`
- `Next: <follow-up command for discovery or recovery>.`
- Edit/manage modals must include a `Try:` next step when empty.

## 8. Tables, Lists, Truncation, Wrapping

- Keep list rows stable: title, authors, metadata badges, optional abstract preview.
- Favor truncation with ellipsis over hard wrapping for dense list rows.
- Preserve key metadata visibility under width pressure:
- arXiv ID, category, and high-value badges (for example relevance/version).
- In compact status mode, keep only immediate context:
- primary count/query, API page/loading, selection count, sort, S2, HF.
- Compact token casing rules:
- Use `API p<N> loading` for active API fetches.
- Omit lower-priority compact tokens such as preview/version details.

## 9. Help And Discoverability

- Help overlay must include a top `Getting Started` section with the core flow:
- Search, move, select, open, command palette, full help.
- Footer should prioritize immediate next actions for the current mode.
- Modals should use consistent close/cancel hints:
- `Close: Esc` for read-only views and `Cancel: Esc` for edit/confirm flows.
- Use consistent labels across footer/help/notifications:
- `palette`, `history`, `help`, `search`, `open`, `export`.
- Keep label density intentional:
- Footer uses compact tokens while help/modal copy uses expanded phrasing.
- Keep close instructions concise in modals (for example `Close: ? / Esc / q`).
- Command palette must provide a clear empty-state message with next-step guidance.

## 10. PR Checklist

- [ ] New or changed UI copy uses the terminology canon.
- [ ] `Ctrl+e` copy uses canonical browse/API wording.
- [ ] API footer copy uses `Esc/Ctrl+e exit`.
- [ ] Footer preserves the capped hierarchy and context-slot policy.
- [ ] `Ctrl+p palette` and `? help` remain visible in browse contexts.
- [ ] Every empty state includes a concrete `Try:` next step.
- [ ] Empty states also include a concise `Next:` follow-up hint.
- [ ] Confirm modals keep impact text in body and key hints in modal chrome.
- [ ] Error text includes actionable recovery guidance.
- [ ] Non-color/ASCII compatibility remains intact.
- [ ] Tests cover changed help/footer/status/empty/error strings.
- [ ] No keybinding behavior changed unless explicitly intended.

## 11. UI Direction Options

### Minimalist Direction (default)

- Keep status/footer compact.
- Emphasize list/detail reading flow.
- Show only core actions by default; keep advanced actions in palette/help.
- Tradeoff: Lowest cognitive load, but advanced capabilities are less visible.

### Feature-Rich Direction

- Surface more contextual hints and enrichment states inline.
- Keep advanced features visible in footer/status with short labels.
- Tradeoff: Higher discoverability for power features, but denser visual load.
