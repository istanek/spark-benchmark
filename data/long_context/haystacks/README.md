# Long-context haystacks

This directory holds the large public-domain filler texts the
`long_context_retrieval` suite hides needles inside. The `.txt` files are
**not committed** (they are megabytes of Project Gutenberg books) — they
are git-ignored and fetched on demand.

## Fetch them

```bash
scripts/fetch_haystacks.sh          # fetch any missing texts
scripts/fetch_haystacks.sh --force  # re-download everything
```

## What gets downloaded

| File | Source | License |
| --- | --- | --- |
| `melville_moby_dick.txt` | [Project Gutenberg #2701](https://www.gutenberg.org/files/2701/2701-0.txt) | Public Domain |
| `darwin_origin_of_species.txt` | [Project Gutenberg #1228](https://www.gutenberg.org/files/1228/1228-0.txt) | Public Domain |

Both are long enough (each well over 130k tokens for common tokenizers)
to serve as filler up to the suite's 131072-token ceiling. The exact
provenance is also recorded in
`data/long_context/long_context_retrieval_v1.json`.
