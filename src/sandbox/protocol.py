"""
Wire protocol shared by the sandbox parent and worker.

Every message is a 4-byte big-endian length header followed by a body, so a
partial read on a pipe cannot desynchronise the stream.

The two directions use *different* encodings, on purpose:

    parent -> worker   pickle   (jobs: the payload is a DataFrame)
    worker -> parent   JSON     (results)

The worker is the process that executes LLM-generated code, so it is the
untrusted end of this pipe. `pickle.loads` on data from an untrusted peer is
arbitrary code execution: a compromised worker could reply with a crafted
pickle and take over the web app — turning a contained sandbox breach into a
full one. JSON has no such opcodes, so the blast radius of a worker
compromise stops at the worker.

The reverse direction stays pickle because the parent is trusted and a
DataFrame has no JSON representation worth inventing. Figure bytes are
base64-encoded to survive the JSON hop.
"""

from __future__ import annotations

import base64
import json
import pickle
import struct
from typing import Any, BinaryIO

_HEADER = struct.Struct(">I")
MAX_MESSAGE_BYTES = 256 * 1024 * 1024  # 256 MB ceiling on a single message

_FIGURE_ENCODING = "figure_encoding"


# ── Framing ───────────────────────────────────────────────────


def _write_frame(stream: BinaryIO, body: bytes) -> None:
    stream.write(_HEADER.pack(len(body)))
    stream.write(body)
    stream.flush()


def _read_exactly(stream: BinaryIO, count: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None  # EOF — peer died or closed the pipe
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream: BinaryIO) -> bytes | None:
    header = _read_exactly(stream, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(f"Message too large: {length} bytes")
    return _read_exactly(stream, length)


# ── parent -> worker (jobs) ───────────────────────────────────


def write_job(stream: BinaryIO, payload: Any) -> None:
    """Send a job to the worker. Pickled: the payload carries a DataFrame."""
    _write_frame(stream, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def read_job(stream: BinaryIO) -> Any | None:
    """Read a job in the worker. Returns None on clean EOF."""
    body = _read_frame(stream)
    if body is None:
        return None
    # Trusted direction: this data came from our own parent process.
    return pickle.loads(body)  # noqa: S301


# ── worker -> parent (results) ────────────────────────────────


def write_result(stream: BinaryIO, payload: dict) -> None:
    """Send a result to the parent. JSON, so the parent never unpickles."""
    figure = payload.get("figure")
    if isinstance(figure, bytes):
        payload = {
            **payload,
            "figure": base64.b64encode(figure).decode("ascii"),
            _FIGURE_ENCODING: "base64",
        }
    _write_frame(stream, json.dumps(payload).encode("utf-8"))


def read_result(stream: BinaryIO) -> dict | None:
    """Read a result in the parent. Returns None on clean EOF."""
    body = _read_frame(stream)
    if body is None:
        return None

    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("Sandbox sent a non-object result.")
    if payload.pop(_FIGURE_ENCODING, None) == "base64":
        payload["figure"] = base64.b64decode(payload["figure"])
    return payload
