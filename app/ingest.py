"""Ingestion: load -> clean (crop sidebars, drop boilerplate + repeated headers/footers) -> heading-aware
chunking -> metadata (source, title, heading, page, allowed roles) -> dedupe.
Per-file cleaning rules live in docs/_sources.json (title, url, max_x, start_at, stop_after, drop, ...)."""
import csv, hashlib, json, re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ALL_ROLES = ("viewer", "analyst", "admin")
ITEM_HEAD = re.compile(r"^Item\s+\d{1,2}[A-C]?\.\s+[A-Z]")                 # SEC 10-K items
NUM_HEAD = re.compile(r"^\d{1,2}\.\s+[A-Z][^.]{3,80}\.\s*[—–-]")            # numbered rules/sections


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int = 1
    allowed_roles: tuple = ALL_ROLES
    heading: str = ""
    title: str = ""

    @property
    def index_text(self):                 # what gets indexed: title + heading help recall
        return f"{self.title} {self.heading} {self.text}"

    @property
    def label(self):
        h = f" · {self.heading[:60]}" if self.heading else ""
        return f"{self.title or self.source}{h} · p.{self.page}"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _pdf_pages(path: Path, cfg):
    import pymupdf
    pages = []
    with pymupdf.open(path) as doc:
        for p in doc:
            blocks = sorted(p.get_text("blocks"), key=lambda b: (round(b[1]), b[0]))
            mx = cfg.get("max_x")
            t = "\n".join(b[4] for b in blocks if b[6] == 0 and not (mx and b[0] > mx)).replace("\xa0", " ")
            pages.append(re.sub(r"^(Item\s+\d{1,2}[A-C]?\.)[ \t]*\n", r"\1 ", t, flags=re.M))   # join "Item 1A." + title lines
    if len(pages) >= 5:                   # running headers/footers/page numbers repeat on most pages
        norm = lambda l: re.sub(r"\d+", "#", l.strip())
        freq = Counter(norm(l) for pg in pages for l in set(pg.splitlines()) if l.strip())
        bad = {k for k, v in freq.items() if v >= 0.4 * len(pages)}
        pages = ["\n".join(l for l in pg.splitlines() if norm(l) not in bad) for pg in pages]
    return pages


def load_file(path: Path, cfg=None):
    """Return list of (page_number, text)."""
    cfg, ext = cfg or {}, path.suffix.lower()
    if ext == ".pdf":
        rx = lambda s: re.compile(r"\s+".join(map(re.escape, s.split())), re.I)
        out, started = [], not cfg.get("start_at")
        for i, t in enumerate(_pdf_pages(path, cfg)):
            if not started:
                m = rx(cfg["start_at"]).search(t)
                if not m:
                    continue
                t, started = t[m.start():], True
            if cfg.get("stop_after"):
                m = rx(cfg["stop_after"]).search(t)
                if m:
                    out.append((i + 1, t[:m.end()]))
                    break
            out.append((i + 1, t))
        drop = {d.lower() for d in cfg.get("drop", [])}
        return [(p, "\n".join(l for l in t.splitlines() if l.strip().lower() not in drop)) for p, t in out]
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return [(i + 1, "; ".join(f"{k}: {v}" for k, v in row.items())) for i, row in enumerate(csv.DictReader(f))]
    if ext in (".txt", ".md"):
        return [(1, path.read_text(encoding="utf-8"))]
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, size: int = 120, overlap: int = 30):
    """Word-based sliding window. size/overlap are in words."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = clean(text).split()
    if not words:
        return []
    step, out = size - overlap, []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def roles_for(path: Path):
    """Access control by filename: admin_* -> admin only; analyst_* -> analyst + admin."""
    n = path.name.lower()
    return ("admin",) if n.startswith("admin_") else ("analyst", "admin") if n.startswith("analyst_") else ALL_ROLES


def _segments(text, numbered, carry):
    segs, cur, buf = [], carry, []
    for line in text.splitlines():
        s = line.strip()
        if ITEM_HEAD.match(s) or (numbered and NUM_HEAD.match(s)):
            if buf:
                segs.append((cur, "\n".join(buf)))
            cur, buf = s[:90], []
        buf.append(line)
    segs.append((cur, "\n".join(buf)))
    return segs, cur


def ingest_dir(directory, size=120, overlap=30):
    directory = Path(directory)
    meta = json.loads((directory / "_sources.json").read_text()) if (directory / "_sources.json").exists() else {}
    seen, chunks = set(), []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".pdf", ".csv", ".txt", ".md"):
            continue
        cfg = meta.get(path.name, {})
        roles, carry, title = roles_for(path), "", cfg.get("title", path.stem)
        for page, text in load_file(path, cfg):
            segs, carry = _segments(text, cfg.get("numbered_headings", False), carry)
            for si, (head, seg) in enumerate(segs):
                for i, c in enumerate(chunk_text(seg, size, overlap)):
                    h = hashlib.md5(c.lower().encode()).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    chunks.append(Chunk(f"{path.stem}-p{page}-{si}-{i}", c, path.name, page, roles, head, title))
    return chunks
