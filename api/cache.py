"""JSON loader with per-path mtime cache. Re-reads only when the file changes on disk."""
import json
from pathlib import Path

_cache: dict[Path, tuple[float, object]] = {}


def load_json_cached(path: Path) -> object | None:
    """Return parsed JSON, or None if the path does not exist."""
    p = Path(path)
    if not p.is_file():
        _cache.pop(p, None)
        return None
    mtime = p.stat().st_mtime
    cached = _cache.get(p)
    if cached and cached[0] == mtime:
        return cached[1]
    data = json.loads(p.read_text(encoding="utf-8"))
    _cache[p] = (mtime, data)
    return data
