"""
tests/test_ycb_downloader.py — Unit tests for data/ycb/downloader.py.

All HTTP requests are mocked with unittest.mock so no network calls are made.
Filesystem operations use pytest's tmp_path fixture.
"""

from __future__ import annotations

import gzip
import io
import json
import struct
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.ycb.downloader import YCBDownloader, DownloadResult, _REQUIRED_FILES


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tgz(name: str) -> bytes:
    """Build a minimal valid .tgz that satisfies _REQUIRED_FILES for `name`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for tmpl in _REQUIRED_FILES:
            rel  = tmpl.format(name=name)
            data = f"# {rel}\n".encode()
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _mock_response(content: bytes, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.headers     = {"Content-Length": str(len(content))}
    resp.iter_content = lambda chunk_size: (
        (content[i : i + chunk_size] for i in range(0, len(content), chunk_size))
    )
    resp.raise_for_status = MagicMock()
    if status >= 400:
        from requests import HTTPError
        resp.raise_for_status.side_effect = HTTPError(response=resp)
    return resp


# ── construction ──────────────────────────────────────────────────────────────

def test_downloader_creates_dest_dir(tmp_path):
    dl = YCBDownloader(tmp_path / "raw")
    assert (tmp_path / "raw").exists()


def test_downloader_stores_dest_dir(tmp_path):
    dl = YCBDownloader(tmp_path / "raw")
    assert dl.dest_dir == tmp_path / "raw"


# ── DownloadResult ────────────────────────────────────────────────────────────

def test_download_result_str_success():
    r = DownloadResult(name="002_master_chef_can", success=True,
                       bytes_dl=5_000_000, elapsed_s=3.2)
    s = str(r)
    assert "ok" in s and "002_master_chef_can" in s


def test_download_result_str_skipped():
    r = DownloadResult(name="002_master_chef_can", success=True, skipped=True)
    assert "skip" in str(r)


def test_download_result_str_fail():
    r = DownloadResult(name="002_master_chef_can", success=False,
                       error="HTTP 403")
    assert "fail" in str(r) and "HTTP 403" in str(r)


# ── unknown object ────────────────────────────────────────────────────────────

def test_download_unknown_object_returns_failure(tmp_path):
    dl = YCBDownloader(tmp_path)
    r  = dl.download("999_imaginary_object")
    assert not r.success
    assert "Unknown" in r.error


# ── skip if already verified ──────────────────────────────────────────────────

def test_download_skips_if_verified(tmp_path):
    name = "002_master_chef_can"
    # Pre-populate required files
    for tmpl in _REQUIRED_FILES:
        p = tmp_path / tmpl.format(name=name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy")
    dl = YCBDownloader(tmp_path, force=False)
    r  = dl.download(name)
    assert r.success and r.skipped


def test_force_redownloads_even_if_verified(tmp_path):
    name    = "002_master_chef_can"
    tgz_data = _make_tgz(name)
    for tmpl in _REQUIRED_FILES:
        p = tmp_path / tmpl.format(name=name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("old")

    dl = YCBDownloader(tmp_path, force=True)
    with patch("requests.get", return_value=_mock_response(tgz_data)):
        r = dl.download(name)
    assert r.success and not r.skipped


# ── successful download ───────────────────────────────────────────────────────

def test_download_success(tmp_path):
    name     = "005_tomato_soup_can"
    tgz_data = _make_tgz(name)
    dl       = YCBDownloader(tmp_path)
    with patch("requests.get", return_value=_mock_response(tgz_data)):
        r = dl.download(name)
    assert r.success
    assert r.path is not None
    assert r.bytes_dl == len(tgz_data)


def test_download_creates_required_files(tmp_path):
    name     = "005_tomato_soup_can"
    tgz_data = _make_tgz(name)
    dl       = YCBDownloader(tmp_path)
    with patch("requests.get", return_value=_mock_response(tgz_data)):
        dl.download(name)
    for tmpl in _REQUIRED_FILES:
        assert (tmp_path / tmpl.format(name=name)).exists()


def test_download_removes_tgz_after_extraction(tmp_path):
    name     = "007_tuna_fish_can"
    tgz_data = _make_tgz(name)
    dl       = YCBDownloader(tmp_path)
    with patch("requests.get", return_value=_mock_response(tgz_data)):
        dl.download(name)
    assert not any(tmp_path.glob("*.tgz"))


def test_download_elapsed_positive(tmp_path):
    name     = "007_tuna_fish_can"
    tgz_data = _make_tgz(name)
    dl       = YCBDownloader(tmp_path)
    with patch("requests.get", return_value=_mock_response(tgz_data)):
        r = dl.download(name)
    assert r.elapsed_s >= 0.0


# ── HTTP errors ───────────────────────────────────────────────────────────────

def test_download_http_404_returns_failure(tmp_path):
    import requests as req_mod
    name = "003_cracker_box"
    dl   = YCBDownloader(tmp_path)
    resp = _mock_response(b"Not Found", status=404)
    with patch("requests.get", return_value=resp):
        r = dl.download(name)
    assert not r.success
    assert r.error is not None


def test_download_timeout_returns_failure(tmp_path):
    import requests as req_mod
    name = "004_sugar_box"
    dl   = YCBDownloader(tmp_path, timeout=1)
    with patch("requests.get", side_effect=req_mod.Timeout()):
        r = dl.download(name)
    assert not r.success
    assert "timeout" in r.error.lower()


def test_download_connection_error_returns_failure(tmp_path):
    import requests as req_mod
    name = "006_mustard_bottle"
    dl   = YCBDownloader(tmp_path)
    with patch("requests.get", side_effect=req_mod.ConnectionError("no network")):
        r = dl.download(name)
    assert not r.success
    assert "Connection" in r.error


# ── tarfile safety ────────────────────────────────────────────────────────────

def test_extract_strips_absolute_paths(tmp_path):
    """Archives with absolute paths must not escape dest."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"malicious"
        info = tarfile.TarInfo(name="/etc/passwd")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    # _extract should silently skip the dangerous member
    YCBDownloader._extract(
        tgz  = _write_tgz(tmp_path, buf.getvalue()),
        dest = tmp_path / "out",
    )
    assert not (tmp_path / "out" / "etc" / "passwd").exists()


def test_extract_strips_traversal_paths(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"bad"
        info = tarfile.TarInfo(name="../../../evil.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    YCBDownloader._extract(
        tgz  = _write_tgz(tmp_path, buf.getvalue()),
        dest = tmp_path / "out",
    )
    assert not (tmp_path / "evil.txt").exists()


def _write_tgz(base: Path, data: bytes) -> Path:
    p = base / "test.tgz"
    p.write_bytes(data)
    return p


# ── download_all ──────────────────────────────────────────────────────────────

def test_download_all_returns_one_per_name(tmp_path):
    names = ["002_master_chef_can", "005_tomato_soup_can"]
    def fake_get(url, **kwargs):
        for name in names:
            if name in url:
                return _mock_response(_make_tgz(name))
        return _mock_response(b"", status=404)

    dl = YCBDownloader(tmp_path)
    with patch("requests.get", side_effect=fake_get):
        results = dl.download_all(names=names, n_workers=2)
    assert len(results) == 2


def test_download_all_sorted_by_name(tmp_path):
    names = ["005_tomato_soup_can", "002_master_chef_can"]
    def fake_get(url, **kwargs):
        for name in names:
            if name in url:
                return _mock_response(_make_tgz(name))
        return _mock_response(b"", status=404)

    dl = YCBDownloader(tmp_path)
    with patch("requests.get", side_effect=fake_get):
        results = dl.download_all(names=names, n_workers=1)
    result_names = [r.name for r in results]
    assert result_names == sorted(result_names)


# ── status ────────────────────────────────────────────────────────────────────

def test_status_returns_dict_of_bools(tmp_path):
    dl = YCBDownloader(tmp_path)
    st = dl.status()
    assert isinstance(st, dict)
    assert all(isinstance(v, bool) for v in st.values())


def test_status_false_for_empty_dir(tmp_path):
    dl = YCBDownloader(tmp_path)
    st = dl.status()
    assert all(v is False for v in st.values())


def test_status_true_after_download(tmp_path):
    name     = "002_master_chef_can"
    tgz_data = _make_tgz(name)
    dl       = YCBDownloader(tmp_path)
    with patch("requests.get", return_value=_mock_response(tgz_data)):
        dl.download(name)
    assert dl.status()[name] is True


# ── path helpers ──────────────────────────────────────────────────────────────

def test_mesh_obj_path(tmp_path):
    dl = YCBDownloader(tmp_path)
    p  = dl.mesh_obj("002_master_chef_can")
    assert p == tmp_path / "002_master_chef_can" / "google_16k" / "textured.obj"


def test_mesh_stl_path(tmp_path):
    dl = YCBDownloader(tmp_path)
    p  = dl.mesh_stl("002_master_chef_can")
    assert p == tmp_path / "002_master_chef_can" / "google_16k" / "nontextured.stl"


# ── persistence ───────────────────────────────────────────────────────────────

def test_save_status_creates_json(tmp_path):
    dl   = YCBDownloader(tmp_path / "raw")
    dest = tmp_path / "status.json"
    dl.save_status(dest)
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert "objects" in data
    assert "timestamp" in data


def test_load_status_roundtrip(tmp_path):
    dl   = YCBDownloader(tmp_path / "raw")
    dest = tmp_path / "status.json"
    dl.save_status(dest)
    loaded = YCBDownloader.load_status(dest)
    assert "objects" in loaded


# ── progress callback ─────────────────────────────────────────────────────────

def test_progress_callback_called(tmp_path):
    name     = "002_master_chef_can"
    tgz_data = _make_tgz(name)
    calls    = []
    def cb(n, done, total):
        calls.append((n, done, total))

    dl = YCBDownloader(tmp_path, progress_cb=cb)
    with patch("requests.get", return_value=_mock_response(tgz_data)):
        dl.download(name)
    assert len(calls) > 0
    assert all(c[0] == name for c in calls)
