# dataroots brand — for on-brand artifact generation

This is the mandatory design system for artifacts generated from reports in this
repo (O&O fiches, one-pagers, decks). Apply it whenever you render a report into
a human-facing view. Generated artifacts are **gitignored** (see `.gitignore`).

## Colour — always as `--dr-*` custom properties

Never hard-code hex in markup; declare tokens in `:root` and reference `var(--dr-*)`.

| Token            | Hex        | Use                                             |
|------------------|------------|-------------------------------------------------|
| `--dr-green`     | `#38b580`  | Primary. Accents, highlights, primary actions.  |
| `--dr-indigo`    | `#445ba7`  | Secondary. Headers, links, secondary emphasis.  |
| `--dr-coral`     | `#f37d82`  | Spot only — **one** use per screen/page.        |
| `--dr-ink`       | `#212121`  | Body text on light.                             |
| `--dr-midnight`  | `#0f172a`  | Dark surfaces / dark-mode background.           |

Derive tints/shades from these (e.g. `color-mix()` or rgba) rather than
introducing new base colours. Coral is a scalpel: a single spot per view.

## Type

- **Inter** — display, headings, UI. Headings weight **700**.
- **Roboto Mono** — metadata, stats, code, dates, numeric tables.
- Set a sensible base size (~13–16px) and line-height (~1.4).

## Wordmark & iconography

- Wordmark is always lowercase: **dataroots**.
- Icons: **Lucide** only. **No emoji** in decks or UI.

## Voice

- **Verb-led, outcome-first.** Lead with what happened / what to do.
- Sentence case for headings and UI copy.
- Concrete over abstract; results over activity. Negative results are valid.

## Generation notes

- The tokenized fiche template lives at `templates/oo-projectfiche.html` — fill
  its `{{...}}` placeholders from a report's front matter + body, then print to PDF.
- Decks/one-pagers: `marp-slides` skill / `python-pptx` / pandoc, styled with the
  tokens above.
