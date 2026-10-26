"""Locate the Olist SQLite DB. The repo ships data/olist.db.gz (small enough for GitHub);
it is unpacked once into a writable temp dir."""
import gzip, os, shutil, sqlite3, tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def get_db_path() -> Path:
    if os.getenv("OLIST_DB"):
        return Path(os.environ["OLIST_DB"])
    plain, gz = DATA / "olist.db", DATA / "olist.db.gz"
    if plain.exists():
        return plain
    out = Path(tempfile.gettempdir()) / "olist_copilot.db"
    if not out.exists() or out.stat().st_mtime < gz.stat().st_mtime:
        tmp = out.with_suffix(".part")
        with gzip.open(gz, "rb") as f, open(tmp, "wb") as g:
            shutil.copyfileobj(f, g)
        tmp.replace(out)
    return out


def connect_ro(path=None):
    return sqlite3.connect(f"file:{path or get_db_path()}?mode=ro", uri=True, check_same_thread=False)
