# 🔥 HuggingFace Trending

Surface trending signals from HuggingFace Daily Papers — community upvotes, comments, GitHub info, AI summaries, and keywords.

## Setup

Enable at runtime with `Ctrl+h`, or add to `config.json`:

```json
{
  "hf_enabled": true,
  "hf_cache_ttl_hours": 6
}
```

| Key | Description | Default |
|-----|-------------|---------|
| `hf_enabled` | Enable HF trending on startup | `false` |
| `hf_cache_ttl_hours` | Hours to cache HF data (trending changes frequently) | `6` |

## Usage

1. Press `Ctrl+h` to toggle on/off
2. Data is auto-fetched and cross-matched against loaded papers
3. Trending papers show upvote badges in the list view
4. The detail pane shows a HuggingFace section with upvotes, comments, GitHub repository, AI summary, and keywords
5. Sort by trending with `s` (cycle sort order)
