#!/usr/bin/env python3
"""Serve the SONIC review artifacts with byte-range support.

``python3 -m http.server`` answers range requests with 200 and the whole body and
never advertises ``Accept-Ranges``.  Chrome then marks every video unseekable:
``HTMLMediaElement.seekable`` stays ``[0, 0]`` even after the file is fully
buffered, assignments to ``currentTime`` are discarded, and the native timeline
slider cannot be moved.  This handler implements single-range GETs (206 with
``Content-Range``/``Accept-Ranges``) so review videos seek in any browser.

Usage: python3 scripts/serve-sonic-review.py [--port 8765] [--bind 127.0.0.1]
       [--directory DIR]
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import re
import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

EXTRA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".json": "application/json",
    ".npz": "application/octet-stream",
    ".parquet": "application/octet-stream",
    ".yaml": "text/yaml; charset=utf-8",
    ".yml": "text/yaml; charset=utf-8",
}
for suffix, mime in EXTRA_TYPES.items():
    mimetypes.add_type(mime, suffix)


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Byte range for a single-range request, or None when unsatisfiable.

    Only the single-range form browsers send for media is supported; anything
    else falls back to a full 200 response, which is legal for a server.
    """
    match = RANGE_RE.match(header.strip())
    if match is None:
        return None
    first, last = match.group(1), match.group(2)
    if first == "" and last == "":
        return None
    if first == "":  # suffix range: last N bytes
        length = int(last)
        if length == 0:
            return None
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(first)
        if start >= size:
            return None
        end = size - 1 if last == "" else min(int(last), size - 1)
    if end < start:
        return None
    return start, end


class RangeRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SonicReviewHTTP/1.0"

    def __init__(self, *args, **kwargs):
        self._range_length: int | None = None
        super().__init__(*args, **kwargs)

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            file = open(path, "rb")  # noqa: SIM115 - handed to the caller, closed by do_GET
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            stat = os.fstat(file.fileno())
            size = stat.st_size
            last_modified = self.date_time_string(stat.st_mtime)
            content_type = self.guess_type(path)
            if "If-Modified-Since" in self.headers and self.headers["If-Modified-Since"] == last_modified:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("Last-Modified", last_modified)
                self.end_headers()
                file.close()
                return None
            range_header = self.headers.get("Range")
            if range_header:
                span = parse_range(range_header, size)
                if span is None:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    file.close()
                    return None
                start, end = span
                self._range_length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(self._range_length))
                self.send_header("Last-Modified", last_modified)
                self.end_headers()
                file.seek(start)
                return file
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.send_header("Last-Modified", last_modified)
            self.end_headers()
            return file
        except Exception:
            file.close()
            raise

    def copyfile(self, source, outputfile):
        if self._range_length is None:
            super().copyfile(source, outputfile)
            return
        remaining = self._range_length
        while remaining > 0:
            chunk = source.read(min(256 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def guess_type(self, path):
        suffix = Path(path).suffix.lower()
        if suffix in EXTRA_TYPES:
            return EXTRA_TYPES[suffix]
        return super().guess_type(path)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "datasets" / "first_tur_processed",
        help="root that holds sonic_pilot_review.html and the pilot run directories",
    )
    args = parser.parse_args()
    root = args.directory.resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    handler = partial(RangeRequestHandler, directory=str(root))
    try:
        server = ThreadingHTTPServer((args.bind, args.port), handler)
    except OSError as error:
        print(f"error: cannot listen on {args.bind}:{args.port} ({error})", file=sys.stderr)
        print("hint: another server (for example python3 -m http.server) may own the port", file=sys.stderr)
        return 2
    server.daemon_threads = True
    page = root / "sonic_pilot_review.html"
    print(f"serving {root} on http://{args.bind}:{args.port}/ (byte ranges enabled)")
    if page.is_file():
        print(f"review page: http://{args.bind}:{args.port}/sonic_pilot_review.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
