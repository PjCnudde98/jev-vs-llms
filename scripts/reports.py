# /// script
# requires-python = ">=3.14"
# dependencies = ["pyyaml>=6", "rapidfuzz>=3"]
# ///
"""Search, validate, and inspect R&D reports in this repo.

This is Claude's routing entry point. Reports are the source of truth; git and
GitHub own dates/authorship/lifecycle, so this tool *derives* those rather than
reading them from files.

Usage (always via uv so deps resolve):
    uv run scripts/reports.py search [--competence X] [--tag Y] [--owner Z] [--text "..."] [--min-score N] [--exact] [--json]
    uv run scripts/reports.py validate
    uv run scripts/reports.py show <slug>

`--text` is fuzzy (typo-tolerant) and ranked: exact substring hits float to the
top, then rapidfuzz matches scoring at least --min-score (default 75) below them.
Pass --exact for the old substring-only behaviour (in slug order).

Grep fallback if this can't run:  grep -rl "competence:" reports/*/report.md
"""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

# Controlled vocabulary for `competence`. Keep in sync with CLAUDE.md, the
# report template, and .github/labels.yml.
COMPETENCE_VOCAB = {
    "software-engineering",
    "genai",
    "mlops",
    "cloud-engineering",
    "databricks",
    "azure",
    "analytics-engineering",
    "data-governance",
    "traditional-bi",
}

# Front-matter keys that must be present and non-empty.
REQUIRED_KEYS = ("title", "description", "competence", "owner", "people")


# --------------------------------------------------------------------------- #
# Git-derived metadata (never typed into files). Low-level plumbing over a Path,
# used both when building a Report and by `show`. Degrades gracefully when the
# file has no commit history yet (e.g. before the first push).
# --------------------------------------------------------------------------- #
def _git(*args: str) -> str | None:
    """Run git in the repo root; return trimmed stdout, or None on any failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def rel(path: Path) -> str:
    """Path relative to the repo root, for display."""
    return str(path.relative_to(REPO_ROOT))


def concluded_date(path: Path) -> str | None:
    """First commit that added the file ≈ when it concluded (a proxy)."""
    out = _git("log", "--follow", "--diff-filter=A", "--format=%cs", "--", rel(path))
    return out.splitlines()[-1] if out else None


def last_updated(path: Path) -> str | None:
    """Date (YYYY-MM-DD) of the file's most recent commit, or None if uncommitted."""
    return _git("log", "-1", "--format=%cs", "--", rel(path))


def contributors(path: Path) -> list[str]:
    """Distinct commit-author names for the file, newest-first, order-preserving."""
    out = _git("log", "--format=%an", "--", rel(path))
    return list(dict.fromkeys(n for n in out.splitlines() if n)) if out else []


def started_from_issue(path: Path) -> str | None:
    """Best-effort: first-commit → PR → 'Closes #N' → issue createdAt (via gh).

    Returns None on any failure (no gh, no network, no PR, unresolvable) — this
    field is explicitly best-effort and may be absent.
    """
    first_hash = _git("log", "--follow", "--diff-filter=A", "--format=%H", "--", rel(path))
    if not first_hash:
        return None
    first_hash = first_hash.splitlines()[-1]
    try:
        pulls = subprocess.run(
            ["gh", "api", f"repos/{{owner}}/{{repo}}/commits/{first_hash}/pulls"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        if pulls.returncode != 0 or not pulls.stdout.strip():
            return None
        pr_body = "\n".join(pr.get("body") or "" for pr in json.loads(pulls.stdout))
        import re
        m = re.search(r"(?:close[sd]?|fixe?[sd]?|resolve[sd]?)\s+#(\d+)", pr_body, re.I)
        if not m:
            return None
        issue = subprocess.run(
            ["gh", "issue", "view", m.group(1), "--json", "createdAt", "-q", ".createdAt"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        if issue.returncode != 0:
            return None
        return (issue.stdout.strip() or None)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Domain model — each class owns its own parsing, validation, and search
# behaviour; the cmd_* handlers stay thin and delegate here.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Person:
    """A named contributor on the R&D team."""
    name: str
    role: str = ""

    @classmethod
    def from_mapping(cls, d: dict) -> Self:
        """Build from a front-matter {name, role} mapping."""
        return cls(name=d.get("name", ""), role=d.get("role", ""))


@dataclass(frozen=True, slots=True)
class Link:
    """An externally-hosted deliverable / related repo."""
    type: str
    url: str

    @classmethod
    def from_mapping(cls, d: dict) -> Self:
        """Build from a front-matter {type, url} mapping."""
        return cls(type=d.get("type", ""), url=d.get("url", ""))


@dataclass(frozen=True, slots=True)
class Report:
    """One concluded initiative, assembled from front matter + git."""
    slug: str
    path: str
    title: str
    description: str
    competence: tuple[str, ...]
    owner: str
    people: tuple[Person, ...]
    tags: tuple[str, ...]
    links: tuple[Link, ...]
    updated: str | None

    # --- discovery + YAML parsing ------------------------------------------ #
    @staticmethod
    def all_paths() -> list[Path]:
        """Every reports/*/report.md, sorted by slug."""
        return sorted(REPORTS_DIR.glob("*/report.md")) if REPORTS_DIR.is_dir() else []

    @staticmethod
    def parse_front_matter(path: Path) -> tuple[dict | None, str | None]:
        """Return (raw front-matter dict, error); exactly one is non-None."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"cannot read file: {exc}"
        if not text.startswith("---"):
            return None, "no front matter (file must start with '---')"
        lines = text.splitlines()
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            return None, "unterminated front matter (missing closing '---')"
        try:
            data = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError as exc:
            return None, f"invalid YAML: {exc}"
        if data is None:
            return None, "empty front matter"
        if not isinstance(data, dict):
            return None, "front matter is not a key:value mapping"
        return data, None

    @classmethod
    def from_path(cls, path: Path) -> tuple[Self | None, str | None]:
        """Parse one report file into an instance (coercing defensively), or (None, error)."""
        data, err = cls.parse_front_matter(path)
        if err:
            return None, err
        comp, tags = data.get("competence"), data.get("tags")
        people, links = data.get("people"), data.get("links")
        report = cls(
            slug=path.parent.name,
            path=rel(path),
            title=data.get("title", ""),
            description=" ".join(str(data.get("description", "")).split()),
            competence=tuple(comp) if isinstance(comp, list) else (),
            owner=data.get("owner", ""),
            people=tuple(Person.from_mapping(p) for p in people if isinstance(p, dict)) if isinstance(people, list) else (),
            tags=tuple(tags) if isinstance(tags, list) else (),
            links=tuple(Link.from_mapping(link) for link in links if isinstance(link, dict)) if isinstance(links, list) else (),
            updated=last_updated(path),
        )
        return report, None

    @classmethod
    def load_all(cls) -> tuple[list[Self], list[tuple[str, str]]]:
        """Load every report; return (reports, [(relpath, parse_error), ...])."""
        parsed = [(p, *cls.from_path(p)) for p in cls.all_paths()]   # (path, report, err)
        errors = [(rel(p), err) for p, _, err in parsed if err]
        reports = [r for _, r, _ in parsed if r is not None]
        return reports, errors

    # --- validation (staticmethods on the raw dict — messages unchanged) --- #
    @staticmethod
    def _errs_required(data: dict) -> list[str]:
        """Every required key must be present and non-empty."""
        return [f"missing/empty required key: {key}" for key in REQUIRED_KEYS
                if key not in data or data[key] in (None, "", [], {})]

    @staticmethod
    def _errs_competence(data: dict) -> list[str]:
        """Competence must be a list drawn from the controlled vocab."""
        comp = data.get("competence")
        if comp is None:
            return []
        if not isinstance(comp, list):
            return ["competence must be a list"]
        return [f"competence '{c}' not in vocab ({', '.join(sorted(COMPETENCE_VOCAB))})"
                for c in comp if c not in COMPETENCE_VOCAB]

    @staticmethod
    def _errs_people(data: dict) -> list[str]:
        """People must be a list of {name, role} mappings, each carrying a name."""
        people = data.get("people")
        if isinstance(people, list):
            return [f"people[{i}] must be a mapping with a 'name'"
                    for i, person in enumerate(people)
                    if not isinstance(person, dict) or not person.get("name")]
        return ["people must be a list of {name, role}"] if people is not None else []

    @staticmethod
    def _errs_tags(data: dict) -> list[str]:
        """Tags, if present, must be a list of strings."""
        tags = data.get("tags")
        if tags is not None and not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
            return ["tags must be a list of strings"]
        return []

    @staticmethod
    def _errs_links(data: dict) -> list[str]:
        """Links, if present, must be a list of {type, url} mappings."""
        links = data.get("links")
        if links is None:
            return []
        if not isinstance(links, list):
            return ["links must be a list"]
        return [f"links[{i}] must have 'type' and 'url'"
                for i, link in enumerate(links)
                if not isinstance(link, dict) or "type" not in link or "url" not in link]

    @staticmethod
    def validation_errors(data: dict) -> list[str]:
        """All front-matter problems for a raw dict, in stable check order."""
        return [*Report._errs_required(data), *Report._errs_competence(data),
                *Report._errs_people(data), *Report._errs_tags(data), *Report._errs_links(data)]

    @staticmethod
    def problems_at(path: Path) -> list[str]:
        """Parse then validate one file: [parse-error] on failure, else the problem list."""
        data, err = Report.parse_front_matter(path)
        return [err] if err else Report.validation_errors(data)

    # --- search ------------------------------------------------------------ #
    def matches_filters(self, args: argparse.Namespace) -> bool:
        """Exact AND-gates on competence/tag/owner (unchanged semantics)."""
        return (not args.competence or args.competence in self.competence) \
            and (not args.tag or args.tag in self.tags) \
            and (not args.owner or args.owner == self.owner)

    def search_fields(self) -> list[str]:
        """Text fields fuzzy --text scores against (adds slug + people names; links excluded)."""
        return [v for v in (self.title, self.description, self.owner, self.slug,
                            *self.competence, *self.tags, *(p.name for p in self.people)) if v]

    def text_score(self, query: str) -> tuple[float, str]:
        """Relevance of `query`: exact substring in the joined haystack -> (100.0,'exact');
        else max per-field rapidfuzz WRatio (per-field max stops a short title match
        being diluted by a long description).

        >>> r = Report("pytest-agent-eval","p","Agent evaluation","",(),"", (),("agents",),(),None)
        >>> r.text_score("agent evaluation")[1]
        'exact'
        >>> r.text_score("evaluaton")[0] > 75
        True
        """
        from rapidfuzz import fuzz, utils          # lazy: only search pays for rapidfuzz
        fields = self.search_fields()
        if query.strip().lower() in " ".join(fields).lower():
            return 100.0, "exact"
        best = max((fuzz.WRatio(query, v, processor=utils.default_process) for v in fields), default=0.0)
        return best, "fuzzy"


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A report plus its relevance to the current --text query."""
    report: Report
    score: float | None            # None when no --text
    match: str | None              # "exact" | "fuzzy" | None

    @classmethod
    def score_report(cls, report: Report, query: str) -> Self:
        """Score one report against the query."""
        return cls(report, *report.text_score(query))

    @classmethod
    def rank(cls, reports: list[Report], args: argparse.Namespace) -> list[Self]:
        """Score by --text, drop sub-threshold fuzzy hits, sort exact-first then score desc."""
        scored = [cls.score_report(r, args.text) for r in reports]
        kept = [h for h in scored
                if h.match == "exact" or (not args.exact and h.score >= args.min_score)]
        return sorted(kept, key=lambda h: (h.match != "exact", -(h.score or 0.0), h.report.slug))

    def to_json(self) -> dict:
        """Flat, skill-compatible JSON: report fields + score/match."""
        return asdict(self.report) | {"score": self.score, "match": self.match}

    def format(self, show_match: bool) -> str:
        """Human block. Title/description/path lines stay byte-identical to today."""
        r = self.report
        meta = f"  competence: {', '.join(r.competence)}   owner: {r.owner}"
        meta += f"   updated: {r.updated}" if r.updated else ""
        if show_match:
            meta += "   match: exact" if self.match == "exact" else f"   match: {self.score:g}"
        return f"\n• {r.title}\n  {r.description}\n{meta}\n  path: {r.path}"


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def cmd_search(args: argparse.Namespace) -> int:
    """Filter reports, optionally fuzzy-rank by --text, and print human/JSON output."""
    reports, errors = Report.load_all()
    for path, err in errors:
        print(f"warning: skipped {path}: {err}", file=sys.stderr)
    matched = [r for r in reports if r.matches_filters(args)]
    hits = SearchHit.rank(matched, args) if args.text else [SearchHit(r, None, None) for r in matched]
    if args.json:
        print(json.dumps([h.to_json() for h in hits], indent=2, ensure_ascii=False))
        return 0
    if not hits:
        print("No matching reports.")
        return 0
    print("\n".join(h.format(bool(args.text)) for h in hits))
    print(f"\n{len(hits)} report(s).")
    return 0


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def _print_result(relpath: str, problems: list[str]) -> None:
    """Print one ✓/✗ validation block for a report."""
    if problems:
        print(f"✗ {relpath}")
        print("\n".join(f"    {p}" for p in problems))
    else:
        print(f"✓ {relpath}")


def cmd_validate(_: argparse.Namespace) -> int:
    """Validate every report's front matter (the CI gate); non-zero on any problem."""
    paths = Report.all_paths()
    if not paths:
        print("No reports found under reports/*/report.md — nothing to validate.")
        return 0
    results = [(rel(p), Report.problems_at(p)) for p in paths]
    for relpath, problems in results:
        _print_result(relpath, problems)
    total = sum(len(pr) for _, pr in results)
    if total:
        print(f"\nFAILED: {total} problem(s).")
        return 1
    print(f"\nOK: {len(paths)} report(s) valid.")
    return 0


# --------------------------------------------------------------------------- #
# show
# --------------------------------------------------------------------------- #
def cmd_show(args: argparse.Namespace) -> int:
    """Print full metadata for one report: raw front matter + git-derived block."""
    path = REPORTS_DIR / args.slug / "report.md"
    if not path.is_file():
        print(f"No report at {rel(path) if path.is_relative_to(REPO_ROOT) else path}", file=sys.stderr)
        return 2
    data, err = Report.parse_front_matter(path)
    if err:
        print(f"✗ {rel(path)}: {err}", file=sys.stderr)
        return 1

    print(f"# {data.get('title', args.slug)}")
    print(f"slug:        {args.slug}")
    print(f"path:        {rel(path)}")
    print(f"description: {' '.join(str(data.get('description', '')).split())}")
    print(f"competence:  {', '.join(data.get('competence', []) or [])}")
    print(f"owner:       {data.get('owner', '')}")
    people = data.get("people", []) or []
    print("people:      " + (
        "; ".join(f"{p.get('name','?')} ({p.get('role','')})".strip() for p in people)
        if people else "—"))
    print(f"tags:        {', '.join(data.get('tags', []) or []) or '—'}")
    links = data.get("links", []) or []
    print("links:       " + ("; ".join(f"{l.get('type','')}: {l.get('url','')}" for l in links) if links else "—"))

    print("\n# derived from git (approximate — see CLAUDE.md caveats)")
    print(f"concluded:    {concluded_date(path) or '(uncommitted)'}")
    print(f"last updated: {last_updated(path) or '(uncommitted)'}")
    contribs = contributors(path)
    print(f"contributors: {', '.join(contribs) if contribs else '(uncommitted)'}")
    print(f"started:      {started_from_issue(path) or '(unresolved — best-effort)'}")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    """Parse arguments and dispatch to the chosen subcommand."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="find reports by front matter")
    p_search.add_argument("--competence")
    p_search.add_argument("--tag")
    p_search.add_argument("--owner")
    p_search.add_argument("--text")
    p_search.add_argument("--min-score", type=float, default=75.0,
                          help="minimum fuzzy score (0–100) for --text hits (default: 75)")
    p_search.add_argument("--exact", action="store_true",
                          help="substring-only --text (disable fuzzy matching)")
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_validate = sub.add_parser("validate", help="validate front matter (CI gate)")
    p_validate.set_defaults(func=cmd_validate)

    p_show = sub.add_parser("show", help="full metadata for one report")
    p_show.add_argument("slug")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
