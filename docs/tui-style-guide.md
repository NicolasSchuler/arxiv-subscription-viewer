# TUI/CLI Style Guide

This style guide defines copy, layout, and interaction conventions for arXiv Subscription Viewer.

## 1. Purpose And Audiences

- Purpose: Keep the app discoverable for newcomers while preserving high-speed keyboard workflows for frequent users.
- Primary audience: Intermediate and power users who live in terminal workflows.
- Secondary audience: First-time users who need clear guidance without reading full docs first.
- Core user path: install, scan papers, enrich the shortlist, organize a durable queue, export selected work, then configure defaults.

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

Most modals use one of three standard widths. Pick the smallest size that fits the content.

| Size | Width | Height | Use for |
|------|-------|--------|---------|
| **Small** | `52` (fixed cols) | `auto` | Quick pickers, confirmations, single-choice dialogs |
| **Medium** | `70` (fixed cols) | keep existing | Form dialogs, editors, search, list management |
| **Large** | `80%` | `85%` | Full content views, help, recommendations, citations |

Small/Medium modals also cap at `max-width: 90%` for narrow-terminal safety. Large modals keep `min-width: 60; min-height: 20;` for small-terminal safety.

Full-screen reading and diagnostic surfaces (e.g. Trend Radar, Author Profile, Quick Triage, Settings, Paper Comparison) intentionally use bespoke widths outside this set.

**Example assignments** (not exhaustive — see each modal's `CSS`):

- **Small**: ConfirmModal, ExportMenuModal, SummaryModeModal, SectionToggleModal, MetadataSnapshotPickerModal
- **Medium**: PaperEditModal, ArxivSearchModal, CommandPaletteModal, WatchListModal, CollectionsModal, ResearchInterestsModal
- **Large**: HelpScreen, RecommendationsScreen, CitationGraphScreen, PaperChatScreen

- Preserve the three-zone structure:
  - Header: context and active dataset/date.
  - Content: list + details with selection/focus state.
  - Footer: immediate action hints and mode badges.

### Responsive Pane Layout

- The list/detail content area is responsive to terminal width via Textual horizontal breakpoints (`HORIZONTAL_BREAKPOINTS`).
- At **96 cols and wider** the panes keep the side-by-side split (`#left-pane` / `#right-pane`).
- **Below 96 cols** the screen gains the `-narrow` class and the panes stack vertically, list first, so neither pane is squeezed below its readable minimum.
- Narrow-width behavior:
  - Prefer compact status tokens and shorter labels before removing high-value hints.
  - Keep `? help` visible in all contexts.

## 5. Footer Hierarchy Rules

- Default browse footer is capped at 9 hints.
- Footer hints are mouse-clickable: each hint with a bound action uses Textual `@click` action-link markup (single `Static`, no per-hint child widgets) so a click invokes the same app action as the key. Hints without a bound action (e.g. `[/] dates`) render as plain text.
- Always include these core hints in order:
  - `/ search`, `Space select`, `o open`, `s sort`, `r read`, context slot, `E export`, `Ctrl+p commands`, `? help`.
- The context slot is:
  - `[/] dates` if history navigation is available.
  - Otherwise `e S2` when Semantic Scholar is active.
  - Otherwise `L relevance` when LLM scoring is configured.
  - Otherwise `V versions` when starred papers exist.
  - Otherwise `x star`.
- Search footer should emphasize immediate flow:
  - `type to search`, `Enter apply`, `Esc close`, `↑↓ move`, `? help`.
- API footer should emphasize mode exits and paging:
  - `[/] page`, `Esc/Ctrl+e exit`, `A new query`, `o open`, `? help`.
- API empty states must mention both recovery paths:
  - Pagination keys (`[` and `]`) and exit keys (`Esc` or `Ctrl+e`).
- Detail-focus footer should show line-annotation flow:
  - `Tab list`, `j/k scroll`, `a annotate`, `v density`, `Ctrl+d sections`, `? help`.
- Context-sensitive keys must be documented by focus:
  - In list focus, `a` remains select-all.
  - In detail focus, `a` opens quick annotation input for the visible detail-line cursor.

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
2. **Widget glyph sets**: `widgets/listing.py` (`_META_GLYPH_SETS`, toggled by `set_ascii_icons()`), `widgets/details.py` (`_DETAIL_GLYPH_SETS`, `set_ascii_glyphs()`), and `widgets/footer_status.py` (`_CHROME_GLYPH_SETS`, `set_ascii_glyphs()`, re-exported through `widgets/chrome.py`) each keep `"unicode"`/`"ascii"` glyph variants, wired from `ArxivBrowser._configure_ascii_mode()` during init.
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
- Use a width-derived metadata-line budget (default 78, clamped to 36–120 via `meta_line_budget_for()`) so dense rows stay scannable at every pane width.
- When metadata overflows, drop lowest-priority trailing badges and append `+N` to show hidden items.
- Preserve key metadata visibility under width pressure:
  - arXiv ID, category, and high-value badges (for example relevance/version).
- In compact status mode, keep only immediate context:
  - primary count/query, API page/loading, selection count, sort, S2, HF.
- Compact token casing rules:
  - Use `API p<N> loading` for active API fetches.
  - Omit lower-priority compact tokens such as preview/version details.
- Visual status tokens:
  - Render enrichment progress, reading velocity, and category histogram only when the status bar has enough width.
  - Drop visual tokens before dropping core paper count, query/watch context, selection count, API page/loading, and sort context.
- Do not duplicate active progress text; when a visual token shows `Versions`, suppress the separate `Checking versions...` text.
- Use Unicode sparklines/histograms by default and ASCII-safe ramps (`#`, `-`, punctuation) when ASCII mode is active.

## 9. TUI Clarity Checklist

Use this before changing visible TUI behavior:

- **Primary task**: the current screen should make the next useful action obvious without showing the whole command surface.
- **Footer**: keep browse mode to the fixed core hints plus one contextual slot; do not add a second row of shortcuts.
- **Help overlay**: explain features with long-form names and keep the top `Getting Started` list aligned with README and the footer.
- **Rows**: preserve title, authors, and highest-value badges before adding optional enrichment signals.
- **State**: empty, loading, disabled, error, selected, detail-focus, and ASCII/no-color modes need explicit copy or non-color indicators.
- **Verification**: cover changed footer/help/status/empty-state strings with tests and run constrained-width layout checks when the change affects scanning.
- **Snapshots**: keep browse, 96-col breakpoint, narrow browse, detail focus, command palette, light theme, and ASCII high-contrast baselines visually inspected before accepting snapshot updates.

## 10. Help And Discoverability

### Keybinding Tiers

All keybindings are categorised into three progressive tiers so that new users see only the essentials while power users can discover advanced shortcuts on demand.

| Tier | Size | Visibility | Purpose |
|------|------|-----------|---------|
| **Core** | ~12 keys | Default footer | Navigation, search, open, read/star, export, sort, help, quit |
| **Standard** | ~15-20 keys | Prominent in help overlay | Selection, notes, tags, copy, download, PDF, watch, bookmarks, API search |
| **Power** | remaining | Command palette (`Ctrl+p`) | Marks, similarity, citations, LLM features, themes, collections, enrichment toggles |

Rules:
- Core keys always appear in the footer (capped at 9 visible hints per §5).
- Standard keys appear under "Standard · _group_" headers in the help overlay.
- Power keys appear under "Power · _group_" headers in the help overlay and are always accessible via the command palette.
- Moving a key between tiers means updating `HELP_SECTION_ACTIONS` in `help_ui.py` (which section a binding falls under) and, if a section name changes, `TIER_SECTION_NAMES` in `cli_keybindings.py`. Keep the explanatory tier comment in `ui_constants.py` in sync.

### General discoverability rules

- Help overlay must include a top `Getting Started` section with the core flow:
  - Scan, move, select, open/enrich, organize/export, command palette, full help.
- Footer should prioritize immediate next actions for the current mode.
- Modals should use consistent close/cancel hints:
  - `Close: Esc/q` for read-only views and `Cancel: Esc` for edit/confirm flows.
- Use consistent labels across footer/help/notifications:
  - `commands`, `dates`, `help`, `search`, `open`, `export`.
- Keep label density intentional:
  - Footer uses compact tokens while help/modal copy uses expanded phrasing.
- Keep close instructions concise in modals (for example `Close: ? / Esc / q`).
- Command palette must provide a clear empty-state message with next-step guidance.
- Command palette group headers are scan aids only; arrow/Enter selection must skip headers and disabled commands.

## 11. Keybinding Conventions

### Shift-Key Mnemonic Pattern

An observed convention in the existing bindings: **lowercase keys tend to perform quick actions on the current paper or toggle local state**, while **uppercase (Shift) keys tend to open management screens, advanced features, or broader-scope operations**.

This is a convention, not a strict rule — some exceptions exist for historical reasons and mnemonic clarity.

### Case-Sensitive Pairs

**✅ Fits the pattern:**

| Lower | Action | Upper | Action | Why it fits |
|-------|--------|-------|--------|-------------|
| `a` | Select all papers | `A` | Search all of arXiv (API mode) | Quick in-place select vs broad mode switch |
| `e` | Fetch S2 data for current paper | `E` | Open export menu | Single-paper enrichment vs multi-format management screen |
| `t` | Edit tags on current paper | `T` | Open quick-triage screen | Quick in-place tag edit vs review screen |
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
| `F` | Preview the PDF inline | Advanced current-paper preview |
| `G` | Citation graph (S2-powered) | Advanced feature — opens drill-down screen |
| `I` | Preview first HTML figure | Advanced current-paper preview using arXiv HTML |
| `L` | Score papers by relevance (LLM) | Multi-paper operation |

### Guidance for Future Keybindings

When adding a new keybinding:

1. **Prefer lowercase for quick single-paper actions** (toggles, fetches, in-place updates).
2. **Prefer uppercase for management screens or multi-paper operations** (modals, configuration, batch actions).
3. **If the mnemonic letter matters more than the convention**, prioritize discoverability over consistency. A binding that users can guess is better than one that fits a pattern nobody remembers.
4. **All keybindings must be discoverable** via the help overlay (`?`) and the command palette (`Ctrl+p`).
5. **Check for conflicts** before assigning — the `APP_BINDINGS` list in `ui_constants.py` is the single source of truth.

## 12. PR Checklist

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

## 13. UI Direction Options

### Minimalist Direction (default)

- Keep status/footer compact.
- Emphasize list/detail reading flow.
- Show only core actions by default; keep advanced actions in palette/help.
- Tradeoff: Lowest cognitive load, but advanced capabilities are less visible.

### Feature-Rich Direction

- Surface more contextual hints and enrichment states inline.
- Keep advanced features visible in footer/status with short labels.
- Tradeoff: Higher discoverability for power features, but denser visual load.

## 14. User Theme Overrides (`user.tcss`)

Power users can layer their own Textual CSS on top of the embedded stylesheet
by creating a `user.tcss` file next to `config.json` in the platform-specific
config directory:

- Linux: `~/.config/arxiv-browser/user.tcss`
- macOS: `~/Library/Application Support/arxiv-browser/user.tcss`
- Windows: `%APPDATA%/arxiv-browser/user.tcss`

The file is loaded after the app's embedded CSS, so any selectors defined
there win over defaults. Only create it when present; the app starts with no
overrides by default. Example:

```tcss
/* Make the detail pane background slightly darker */
PaperDetails {
    background: #181818;
}

/* Brighter accent for starred rows */
.paper-starred {
    color: #ffd75f;
}
```

Invalid CSS is reported at startup but will not crash the app. Keep overrides
minimal; relying on internal widget IDs is fragile across releases.
