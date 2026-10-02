"""Resumable HTTP downloads with checksums and provenance receipts."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request


@dataclass(frozen=True)
class Download:
    url: str
    filename: str
    md5: str | None = None
    sha256: str | None = None
    size: int | None = None

    def __post_init__(self):
        if Path(self.filename).name != self.filename or self.filename in {"", ".", ".."}:
            raise ValueError("Download filename must be a basename")
        if not self.url.startswith(("https://", "http://")):
            raise ValueError("Downloads require an HTTP(S) URL")


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class IncompleteDownload(OSError):
    pass


def fetch(spec: Download, directory, attempts=3):
    if attempts < 1:
        raise ValueError("Download attempts must be positive")
    for attempt in range(attempts):
        try:
            return _fetch_once(spec, directory)
        except (IncompleteDownload, urllib.error.URLError, TimeoutError,
                http.client.IncompleteRead, ConnectionError) as error:
            if attempt + 1 == attempts:
                raise
            print(f"Retrying {spec.filename} after interrupted transfer: {error}", file=sys.stderr, flush=True)
            time.sleep(attempt + 1)


def _fetch_once(spec: Download, directory):
    """Never accept an existing file merely because it exists.

    Receipts pin mutable URLs on first successful retrieval. Partial transfers
    resume with If-Range validators; servers ignoring Range restart from zero.
    An interrupted transfer can be resumed by invoking the same command again.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / spec.filename
    receipt = path.with_name(path.name + ".download.json")
    if path.exists():
        if not receipt.exists():
            raise FileExistsError(f"Untracked existing file: {path}; move it or use another directory")
        previous = json.loads(receipt.read_text())
        if previous["url"] != spec.url or file_hash(path) != previous["sha256"]:
            raise ValueError(f"Existing download provenance/checksum mismatch: {path}")
        _check(spec, path)
        return path
    part = path.with_name(path.name + ".part")
    state_path = path.with_name(path.name + ".part.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    offset = part.stat().st_size if part.exists() else 0
    if offset and state.get("url") != spec.url:
        raise ValueError("Partial download belongs to a different URL")
    headers = {"User-Agent": "LLM_mapper/0.2 preparation", "Accept-Encoding": "identity"}
    validator = state.get("etag") or state.get("last_modified")
    if offset and validator:
        headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
    else:
        offset = 0
    with urllib.request.urlopen(urllib.request.Request(spec.url, headers=headers), timeout=120) as response:
        if response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset:
                raise ValueError("Invalid HTTP Content-Range")
            total = int(match[3])
        elif response.status == 200:
            offset = 0
            total = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
        else:
            raise ValueError(f"Unexpected download status {response.status}")
        state = {"url": spec.url, "etag": response.headers.get("ETag"),
                 "last_modified": response.headers.get("Last-Modified")}
        write_json(state_path, state)
        with part.open("ab" if offset else "wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
    if total is not None and part.stat().st_size != total:
        raise IncompleteDownload(f"Incomplete download: {part}; rerun to resume")
    _check(spec, part)
    result = {**state, "size": part.stat().st_size, "sha256": file_hash(part),
              "upstream_md5": spec.md5, "upstream_sha256": spec.sha256,
              "retrieved_at": datetime.now(timezone.utc).isoformat()}
    # Receipt first permits recovery after interruption between the two renames.
    write_json(receipt, result)
    part.replace(path)
    state_path.unlink(missing_ok=True)
    return path


def _check(spec, path):
    if spec.size is not None and path.stat().st_size != spec.size:
        raise ValueError(f"Size mismatch: {path}")
    for algorithm in ("md5", "sha256"):
        expected = getattr(spec, algorithm)
        if expected and file_hash(path, algorithm).lower() != expected.lower():
            raise ValueError(f"{algorithm} mismatch: {path}")
