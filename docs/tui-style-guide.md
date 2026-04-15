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
- Use `Commands` for compact `Ctrl+p` prompts and `Command palette` for the modal title.
- Use `Help overlay` for `?`.
- Use `Semantic Scholar` in help text and `S2` in compact status/footer hints.
- Use `History` for date navigation context.
- Compact-vs-long naming policy:
- Footer stays compact (`o open`, `Ctrl+p commands`) for 80-col scan speed.
- Help/modals/binding descriptions use long-form labels (`Open in Browser`, `Command palette`).
- `Ctrl+e` is context-sensitive: `Toggle S2` in browse mode and `Exit Search Results` in arXiv search-results mode.

## 4. Layout Hierarchy Rules

### Modal Sizes

All modals use one of three standard widths. Pick the smallest size that fits the content.

| Size | Width | Height | Use for |
|------|-------|--------|---------|
| **Small** | `52` (fixed cols) | `auto` | Quick pickers, confirmations, single-choice dialogs |
| **Medium** | `70` (fixed cols) | keep existing | Form dialogs, editors, search, list management |
| **Large** | `80%` | `85%` | Full content views, help, recommendations, citations |

Large modals also keep `min-width: 60; min-height: 20;` for small-terminal safety.

**Current assignments:**

- **Small**: ConfirmModal, ExportMenuModal, SummaryModeModal, SectionToggleModal, MetadataSnapshotPickerModal
- **Medium**: PaperEditModal, ArxivSearchModal, CommandPaletteModal, WatchListModal, CollectionsModal, ResearchInterestsModal
- **Large**: HelpScreen, RecommendationsScreen, CitationGraphScreen, PaperChatScreen

- Preserve the three-zone structure:
  - Header: context and active dataset/date.
  - Content: list + details with selection/focus state.
  - Footer: immediate action hints and mode badges.
- Narrow-width behavior:
  - Prefer compact status tokens and shorter labels before removing high-value hints.
  - Keep `? help` visible in all contexts.

## 5. Footer Hierarchy Rules

- Default browse footer is capped at 9 hints.
- Always include these core hints in order:
  - `/ search`, `Space select`, `o open`, `s sort`, `r read`, context slot, `E export`, `Ctrl+p commands`, `? help`.
- The context slot is:
  - `[/] dates` if history navigation is available, else `x star`.
- Search footer should emphasize immediate flow:
  - `type to search`, `Enter apply`, `Esc close`, `↑↓ move`, `? help`.
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

### ASCII Glyph Pattern

When adding any non-ASCII character to the UI, follow the established glyph-set pattern:

1. **Centralized flag**: `src/arxiv_browser/_ascii.py` exposes `is_ascii_mode()` and `set_ascii_mode()`. Any module can import it (zero internal dependencies).
2. **Widget glyph sets**: `widgets/listing.py`, `widgets/details.py`, and `widgets/chrome.py` each maintain `_*_GLYPH_SETS` dicts with `"unicode"` and `"ascii"` variants, toggled via `set_ascii_*()` functions called from `App.__init__`.
3. **Other modules**: Import `is_ascii_mode()` and choose the appropriate character inline:
   ```python
   from arxiv_browser._ascii import is_ascii_mode
   sep = " - " if is_ascii_mode() else " \u00b7 "
   ```
4. **Common substitutions**:
   | Unicode | Codepoint | ASCII fallback |
   |---------|-----------|----------------|
   | `●`     | U+25CF    | `[x]`          |
   | `⭐`    | U+2B50    | `*`            |
   | `✓`     | U+2713    | `v`            |
   | `👁`    | U+1F441   | `[w]`          |
   | `▸` / `▾` | U+25B8/U+25BE | `>` / `v` |
   | `·`     | U+00B7    | `-` or `\|`    |
   | `→`     | U+2192    | `->`           |
   | `↑` / `↓` | U+2191/U+2193 | `^` / `v` |
   | `…`     | U+2026    | `...`          |
   | `—`     | U+2014    | `--`           |
   | `×`     | U+00D7    | `x`            |
   | `│`     | U+2502    | `\|`           |
   | `█` / `░` | U+2588/U+2591 | `#` / `-` |
   | `•`     | U+2022    | `-`            |
   | `🤖` / `⏳` | Emoji  | (omitted)      |

5. **Test**: The `TestAsciiModeNoUnicodeLeaks` class in `tests/test_widgets_listing.py` verifies no Unicode leaks across key widgets using `re.search(r'[^\x00-\x7f]', text)`.
6. **Rule**: Never add a non-ASCII character to UI-visible output without a corresponding ASCII fallback.

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
- Use a deterministic metadata-line budget of 78 visible characters in list rows.
- When metadata overflows, drop lowest-priority trailing badges and append `+N` to show hidden items.
- Preserve key metadata visibility under width pressure:
- arXiv ID, category, and high-value badges (for example relevance/version).
- In compact status mode, keep only immediate context:
- primary count/query, API page/loading, selection count, sort, S2, HF.
- Compact token casing rules:
- Use `API p<N> loading` for active API fetches.
- Omit lower-priority compact tokens such as preview/version details.

## 9. Help And Discoverability

### Keybinding Tiers

All keybindings are categorised into three progressive tiers so that new users see only the essentials while power users can discover advanced shortcuts on demand.

| Tier | Size | Visibility | Purpose |
|------|------|-----------|---------|
| **Core** | ~12 keys | Default footer | Navigation, search, open, read/star, export, sort, help, quit |
| **Standard** | ~15-20 keys | Prominent in help overlay | Selection, notes, tags, copy, download, PDF, watch, bookmarks, API search |
| **Power** | remaining | Command palette (`Ctrl+P`) | Marks, similarity, citations, LLM features, themes, collections, enrichment toggles |

Rules:
- Core keys always appear in the footer (capped at 9 visible hints per §5).
- Standard keys appear under "Standard · _group_" headers in the help overlay.
- Power keys appear under "Power · _group_" headers in the help overlay and are always accessible via the command palette.
- Moving a key between tiers requires updating the tier table in `ui_constants.py` and the corresponding help sections in `help_ui.py`.

### General discoverability rules

- Help overlay must include a top `Getting Started` section with the core flow:
  - Search, move, select, open, command palette, full help.
- Footer should prioritize immediate next actions for the current mode.
- Modals should use consistent close/cancel hints:
  - `Close: Esc/q` for read-only views and `Cancel: Esc` for edit/confirm flows.
- Use consistent labels across footer/help/notifications:
  - `commands`, `dates`, `help`, `search`, `open`, `export`.
- Keep label density intentional:
  - Footer uses compact tokens while help/modal copy uses expanded phrasing.
- Keep close instructions concise in modals (for example `Close: ? / Esc / q`).
- Command palette must provide a clear empty-state message with next-step guidance.

## 10. Keybinding Conventions

### Shift-Key Mnemonic Pattern

An observed convention in the existing bindings: **lowercase keys tend to perform quick actions on the current paper or toggle local state**, while **uppercase (Shift) keys tend to open management screens, advanced features, or broader-scope operations**.

This is a convention, not a strict rule — some exceptions exist for historical reasons and mnemonic clarity.

### Case-Sensitive Pairs

**✅ Fits the pattern:**

| Lower | Action | Upper | Action | Why it fits |
|-------|--------|-------|--------|-------------|
| `e` | Fetch S2 data for current paper | `E` | Open export menu | Single-paper enrichment vs multi-format management screen |
| `w` | Toggle watch-list filter | `W` | Manage watch list | Quick filter toggle vs configuration modal |

**⚠️ Partial fit:**

| Lower | Action | Upper | Action | Notes |
|-------|--------|-------|--------|-------|
| `r` | Toggle read status | `R` | Show similar papers | Both act on the current paper, but `R` opens an advanced discovery screen |
| `p` | Toggle abstract preview | `P` | Open PDF in viewer | Both are current-paper actions; `P` for PDF is a strong mnemonic |

**❌ Exceptions (mnemonic wins over pattern):**

| Lower | Action | Upper | Action | Rationale |
|-------|--------|-------|--------|-----------|
| `c` | Copy to clipboard | `C` | Chat with paper (LLM) | `C` for Chat is more discoverable than any pattern-consistent alternative |
| `v` | Toggle detail mode | `V` | Check version updates | `V` for Versions is a natural mnemonic; both could be considered quick actions |

### Standalone Uppercase Keys

These have no lowercase counterpart and all represent advanced or broad-scope features:

| Key | Action | Category |
|-----|--------|----------|
| `A` | Search all of arXiv (API mode) | Broad scope — switches mode entirely |
| `G` | Citation graph (S2-powered) | Advanced feature — opens drill-down screen |
| `L` | Score papers by relevance (LLM) | Multi-paper operation |

### Guidance for Future Keybindings

When adding a new keybinding:

1. **Prefer lowercase for quick single-paper actions** (toggles, fetches, in-place updates).
2. **Prefer uppercase for management screens or multi-paper operations** (modals, configuration, batch actions).
3. **If the mnemonic letter matters more than the convention**, prioritize discoverability over consistency. A binding that users can guess is better than one that fits a pattern nobody remembers.
4. **All keybindings must be discoverable** via the help overlay (`?`) and the command palette (`Ctrl+P`).
5. **Check for conflicts** before assigning — the `APP_BINDINGS` list in `ui_constants.py` is the single source of truth.

## 11. PR Checklist

- [ ] New or changed UI copy uses the terminology canon.
- [ ] `Ctrl+e` copy uses canonical browse/API wording.
- [ ] Read-only overlays support `Esc/q` close while edit/input overlays keep `Esc` cancel.
- [ ] API footer copy uses `Esc/Ctrl+e exit`.
- [ ] Footer preserves the capped hierarchy and context-slot policy.
- [ ] `Ctrl+p commands` and `? help` remain visible in browse contexts.
- [ ] Every empty state includes a concrete `Try:` next step.
- [ ] Empty states also include a concise `Next:` follow-up hint.
- [ ] Confirm modals keep impact text in body and key hints in modal chrome.
- [ ] Error text includes actionable recovery guidance.
- [ ] Non-color/ASCII compatibility remains intact.
- [ ] Tests cover changed help/footer/status/empty/error strings.
- [ ] No keybinding behavior changed unless explicitly intended.

## 12. UI Direction Options

### Minimalist Direction (default)

- Keep status/footer compact.
- Emphasize list/detail reading flow.
- Show only core actions by default; keep advanced actions in palette/help.
- Tradeoff: Lowest cognitive load, but advanced capabilities are less visible.

### Feature-Rich Direction

- Surface more contextual hints and enrichment states inline.
- Keep advanced features visible in footer/status with short labels.
- Tradeoff: Higher discoverability for power features, but denser visual load.
