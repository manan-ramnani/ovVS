#!/usr/bin/env python3
"""Download and verify optional ANN datasets in ``./data``."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2] / "data"

SIFT_FILENAME = "sift-128-euclidean.hdf5"
SIFT_URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
SIFT_SHA256 = "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984"
SIFT_SIZE_BYTES = 525_128_288
SIFT_SCHEMA = (
    ("train", (1_000_000, 128)),
    ("test", (10_000, 128)),
    ("neighbors", (10_000, 100)),
    ("distances", (10_000, 100)),
)

_AUTO_H5PY = object()


class DatasetValidationError(ValueError):
    """The dataset does not match the pinned artifact contract."""


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    """Return the SHA-256 digest of *path* without reading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sift_schema(path: Path, h5py_module: Any = _AUTO_H5PY) -> str:
    """Validate the pinned SIFT1M datasets when h5py is importable.

    Passing ``None`` explicitly skips the optional schema check. This keeps the
    checksum and download helpers testable without an h5py installation.
    """

    if h5py_module is _AUTO_H5PY:
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError:
            h5py_module = None
        else:
            h5py_module = h5py

    if h5py_module is None:
        return "HDF5 schema not checked (h5py unavailable)"

    try:
        with h5py_module.File(path, "r") as handle:
            for name, expected_shape in SIFT_SCHEMA:
                if name not in handle:
                    raise DatasetValidationError(f"missing HDF5 dataset {name!r}")
                actual_shape = tuple(int(value) for value in handle[name].shape)
                if actual_shape != expected_shape:
                    raise DatasetValidationError(
                        f"HDF5 dataset {name!r} shape mismatch: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )
    except DatasetValidationError:
        raise
    except Exception as exc:
        raise DatasetValidationError(
            f"could not read HDF5 schema: {type(exc).__name__}: {exc}"
        ) from exc

    return "HDF5 schema verified"


def verify_sift(
    path: Path,
    *,
    expected_sha256: str = SIFT_SHA256,
    expected_size: int | None = SIFT_SIZE_BYTES,
    h5py_module: Any = _AUTO_H5PY,
) -> tuple[bool, str]:
    """Verify the pinned checksum and, when possible, the HDF5 schema."""

    if not path.exists():
        return False, f"missing file: {path}"
    if not path.is_file():
        return False, f"not a regular file: {path}"
    if expected_size is not None and path.stat().st_size != expected_size:
        return False, f"size mismatch: expected {expected_size}, got {path.stat().st_size}"

    try:
        actual_sha256 = sha256_file(path)
    except OSError as exc:
        return False, f"could not hash {path}: {type(exc).__name__}: {exc}"

    if actual_sha256.lower() != expected_sha256.lower():
        return (
            False,
            f"SHA-256 mismatch: expected {expected_sha256.lower()}, "
            f"got {actual_sha256.lower()}",
        )

    try:
        schema_status = validate_sift_schema(path, h5py_module=h5py_module)
    except DatasetValidationError as exc:
        return False, str(exc)

    return True, f"sha256={actual_sha256.lower()}; {schema_status}"


def download_sift(
    dest: Path,
    *,
    url: str = SIFT_URL,
    expected_sha256: str = SIFT_SHA256,
    expected_size: int | None = SIFT_SIZE_BYTES,
    h5py_module: Any = _AUTO_H5PY,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[bool, str]:
    """Download SIFT1M to a sibling temporary file, verify it, then replace."""

    if urlparse(url).scheme.lower() != "https":
        return False, f"refusing non-HTTPS dataset URL: {url}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ovVS-fetch/0.2"})
        with opener(request, timeout=60) as response:
            final_url = response.geturl()
            if urlparse(final_url).scheme.lower() != "https":
                raise RuntimeError(f"refusing HTTPS downgrade redirect to {final_url}")
            content_length = getattr(response, "headers", {}).get("Content-Length")
            if expected_size is not None and content_length is not None:
                try:
                    declared_size = int(content_length)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"invalid Content-Length: {content_length!r}") from exc
                if declared_size != expected_size:
                    raise RuntimeError(
                        f"download size mismatch: expected {expected_size}, server declared {declared_size}"
                    )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=dest.parent,
                prefix=f".{dest.name}.",
                suffix=".part",
                delete=False,
            ) as stream:
                temp_path = Path(stream.name)
                received = 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    received += len(chunk)
                    if expected_size is not None and received > expected_size:
                        raise RuntimeError(
                            f"download exceeded pinned size {expected_size}; refusing further data"
                        )
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())

        if temp_path is None:
            raise RuntimeError("download temporary file was not created")

        valid, detail = verify_sift(
            temp_path,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            h5py_module=h5py_module,
        )
        if not valid:
            return False, f"downloaded artifact failed validation: {detail}"

        os.replace(temp_path, dest)
        temp_path = None
        return True, detail
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def ensure_sift(
    dest: Path,
    *,
    expected_sha256: str = SIFT_SHA256,
    expected_size: int | None = SIFT_SIZE_BYTES,
    h5py_module: Any = _AUTO_H5PY,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[bool, str]:
    """Verify an existing dataset or safely fetch it when it is absent."""

    if dest.exists():
        valid, detail = verify_sift(
            dest,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            h5py_module=h5py_module,
        )
        if not valid:
            return False, f"existing file left unchanged: {detail}"
        return True, detail

    return download_sift(
        dest,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        h5py_module=h5py_module,
        opener=opener,
    )


def main() -> int:
    print("ovVS dataset kit")
    print(f"target: {ROOT}")
    ROOT.mkdir(parents=True, exist_ok=True)
    marker = ROOT / "README.txt"
    marker.write_text(
        "Optional datasets. Unit tests use generated tensors.\n"
        f"{SIFT_FILENAME} is accepted only when its pinned SHA-256 matches; "
        "h5py also verifies its SIFT1M schema when installed.\n"
        "A corrupt existing file is reported and left unchanged.\n",
        encoding="utf-8",
    )

    dest = ROOT / SIFT_FILENAME
    existed = dest.exists()
    if not existed:
        print(f"fetch {SIFT_URL}")

    valid, detail = ensure_sift(dest)
    if not valid:
        label = "SIFT1M validation failed" if existed else "SIFT1M download failed"
        print(f"{label}: {dest}: {detail}", file=sys.stderr)
        return 1

    action = "verified" if existed else "wrote and verified"
    print(f"{action} {dest} bytes={dest.stat().st_size} {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
