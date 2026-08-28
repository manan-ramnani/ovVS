from __future__ import annotations

import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any


def _load_fetch_module() -> Any:
    path = Path(__file__).with_name("fetch.py")
    spec = importlib.util.spec_from_file_location("ovvs_data_fetch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load_fetch_module()


class _Dataset:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _H5File(dict[str, _Dataset]):
    def __enter__(self) -> "_H5File":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _H5Module:
    def __init__(self, datasets: dict[str, _Dataset]) -> None:
        self.datasets = datasets

    def File(self, path: Path, mode: str) -> _H5File:  # noqa: N802 - mirrors h5py
        del path
        if mode != "r":
            raise AssertionError(f"unexpected mode {mode}")
        return _H5File(self.datasets)


class _Response(io.BytesIO):
    def geturl(self) -> str:
        return "https://ann-benchmarks.com/sift-128-euclidean.hdf5"

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FetchTests(unittest.TestCase):
    def test_verify_accepts_matching_checksum_without_h5py(self) -> None:
        payload = b"known local fixture"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.hdf5"
            path.write_bytes(payload)

            valid, detail = fetch.verify_sift(
                path, expected_sha256=expected, expected_size=len(payload), h5py_module=None
            )

        self.assertTrue(valid)
        self.assertIn(expected, detail)
        self.assertIn("h5py unavailable", detail)

    def test_verify_reports_checksum_mismatch_without_modifying_file(self) -> None:
        payload = b"corrupt fixture"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.hdf5"
            path.write_bytes(payload)

            valid, detail = fetch.verify_sift(
                path, expected_sha256="0" * 64, expected_size=len(payload), h5py_module=None
            )

            self.assertFalse(valid)
            self.assertIn("SHA-256 mismatch", detail)
            self.assertEqual(path.read_bytes(), payload)

    def test_schema_requires_all_pinned_shapes(self) -> None:
        datasets = {
            name: _Dataset(shape) for name, shape in fetch.SIFT_SCHEMA
        }
        datasets["train"] = _Dataset((999_999, 128))

        with self.assertRaisesRegex(
            fetch.DatasetValidationError,
            r"train.*expected \(1000000, 128\), got \(999999, 128\)",
        ):
            fetch.validate_sift_schema(Path("unused"), _H5Module(datasets))

    def test_download_verifies_before_atomic_replace(self) -> None:
        payload = b"download fixture"
        expected = hashlib.sha256(payload).hexdigest()

        def opener(request: object, timeout: int) -> _Response:
            del request
            self.assertEqual(timeout, 60)
            return _Response(payload)

        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "sift.hdf5"
            valid, detail = fetch.download_sift(
                dest,
                expected_sha256=expected,
                expected_size=len(payload),
                h5py_module=None,
                opener=opener,
            )

            self.assertTrue(valid, detail)
            self.assertEqual(dest.read_bytes(), payload)
            self.assertEqual(list(dest.parent.glob("*.part")), [])

    def test_download_refuses_data_beyond_pinned_size(self) -> None:
        payload = b"oversized payload"

        def opener(request: object, timeout: int) -> _Response:
            del request, timeout
            return _Response(payload)

        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "sift.hdf5"
            valid, detail = fetch.download_sift(
                dest,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size=4,
                h5py_module=None,
                opener=opener,
            )

            self.assertFalse(valid)
            self.assertIn("exceeded pinned size", detail)
            self.assertFalse(dest.exists())
            self.assertEqual(list(dest.parent.glob("*.part")), [])

    def test_corrupt_existing_file_is_not_replaced_or_downloaded(self) -> None:
        payload = b"preserve me"

        def opener(*args: object, **kwargs: object) -> object:
            raise AssertionError(f"unexpected download: {args}, {kwargs}")

        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "sift.hdf5"
            dest.write_bytes(payload)
            valid, detail = fetch.ensure_sift(
                dest,
                expected_sha256="0" * 64,
                expected_size=len(payload),
                h5py_module=None,
                opener=opener,
            )

            self.assertFalse(valid)
            self.assertIn("existing file left unchanged", detail)
            self.assertEqual(dest.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
