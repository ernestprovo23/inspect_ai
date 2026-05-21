"""Tests for the resume-side FS-copy helpers against a moto-backed S3.

Covers ``_fs_copy_cross_cutting`` and ``_fs_copy_repo`` downloading a
remote sample dir's contents into a local staging dir, plus the
``seed_manifest`` step that marks them as already-shipped so the next
host_egress doesn't re-upload the resume payload.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai._util.asyncfiles import AsyncFilesystem
from inspect_ai.util._checkpoint._host_egress import (
    MANIFEST_FILENAME,
    host_egress,
    seed_manifest,
)
from inspect_ai.util._checkpoint.hydrate import (
    _fs_copy_cross_cutting,
    _fs_copy_repo,
)

S3_BUCKET = "s3://test-bucket"


async def _put(fs: AsyncFilesystem, uri: str, content: bytes) -> None:
    await fs.write_file(uri, content)


async def test_fs_copy_cross_cutting_downloads_from_s3(
    tmp_path: Path, mock_s3: None
) -> None:
    src = f"{S3_BUCKET}/old-eval.checkpoints/s__0"
    new = tmp_path / "staging"
    new.mkdir()

    async with AsyncFilesystem() as fs:
        await _put(
            fs,
            f"{src}/restic/restic-config.json",
            b'{"restic_password":"the-pw"}',
        )
        await _put(fs, f"{src}/ckpt-00001.json", b'{"checkpoint_id":1}')
        await _put(fs, f"{src}/ckpt-00002.json", b'{"checkpoint_id":2}')

        written = await _fs_copy_cross_cutting(src, str(new))

    assert set(written) == {
        "restic/restic-config.json",
        "ckpt-00001.json",
        "ckpt-00002.json",
    }
    assert (
        new / "restic" / "restic-config.json"
    ).read_bytes() == b'{"restic_password":"the-pw"}'
    assert (new / "ckpt-00001.json").read_bytes() == b'{"checkpoint_id":1}'
    assert (new / "ckpt-00002.json").read_bytes() == b'{"checkpoint_id":2}'


async def test_fs_copy_cross_cutting_noop_when_source_missing(
    tmp_path: Path, mock_s3: None
) -> None:
    """A source dir with no relevant files (fresh resume edge) returns []."""
    src = f"{S3_BUCKET}/empty-eval.checkpoints/s__0"
    new = tmp_path / "staging"
    new.mkdir()

    async with AsyncFilesystem():
        written = await _fs_copy_cross_cutting(src, str(new))

    assert written == []
    assert not (new / "restic").exists()


async def test_fs_copy_repo_downloads_tree_from_s3(
    tmp_path: Path, mock_s3: None
) -> None:
    src_root = f"{S3_BUCKET}/eval.checkpoints/s__0"
    new_repo = tmp_path / "staging" / "restic" / "host"

    async with AsyncFilesystem() as fs:
        await _put(fs, f"{src_root}/restic/host/config", b"cfg")
        await _put(fs, f"{src_root}/restic/host/keys/key01", b"k")
        await _put(fs, f"{src_root}/restic/host/data/ab/cdef", b"pack-data")
        await _put(fs, f"{src_root}/restic/host/index/11", b"idx")
        await _put(fs, f"{src_root}/restic/host/snapshots/22", b"snap")

        written = await _fs_copy_repo(
            src_root, "restic/host", str(new_repo), label="host"
        )

    assert set(written) == {
        "restic/host/config",
        "restic/host/keys/key01",
        "restic/host/data/ab/cdef",
        "restic/host/index/11",
        "restic/host/snapshots/22",
    }
    assert (new_repo / "config").read_bytes() == b"cfg"
    assert (new_repo / "keys" / "key01").read_bytes() == b"k"
    assert (new_repo / "data" / "ab" / "cdef").read_bytes() == b"pack-data"


async def test_fs_copy_repo_raises_when_source_missing(
    tmp_path: Path, mock_s3: None
) -> None:
    src_root = f"{S3_BUCKET}/eval.checkpoints/s__0"
    new_repo = tmp_path / "staging" / "restic" / "host"

    async with AsyncFilesystem():
        try:
            await _fs_copy_repo(src_root, "restic/host", str(new_repo), label="host")
        except RuntimeError as e:
            assert "no files were found" in str(e)
        else:
            raise AssertionError("expected RuntimeError when source missing")


async def test_seed_manifest_after_remote_resume_blocks_reupload(
    tmp_path: Path, mock_s3: None
) -> None:
    """Seeding the manifest blocks the first post-resume host_egress.

    After resume populates staging from the destination and we
    seed_manifest, a subsequent host_egress with no new files should be
    a no-op — the destination is unchanged (no re-upload).
    """
    src_root = f"{S3_BUCKET}/eval.checkpoints/s__0"
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "context").mkdir()

    async with AsyncFilesystem() as fs:
        # Set up the destination with a complete sample subtree.
        await _put(
            fs, f"{src_root}/restic/restic-config.json", b'{"restic_password":"p"}'
        )
        await _put(fs, f"{src_root}/restic/host/config", b"cfg")
        await _put(fs, f"{src_root}/restic/host/data/ab/cd", b"pack")
        await _put(fs, f"{src_root}/ckpt-00001.json", b"side1")

        # Resume: download into a fresh local staging dir.
        await _fs_copy_cross_cutting(src_root, str(staging))
        await _fs_copy_repo(
            src_root, "restic/host", str(staging / "restic" / "host"), label="host"
        )
        seed_manifest(str(staging))

        # Manifest should list everything we just downloaded.
        manifest_lines = (staging / MANIFEST_FILENAME).read_text().splitlines()
        assert set(manifest_lines) == {
            "restic/restic-config.json",
            "restic/host/config",
            "restic/host/data/ab/cd",
            "ckpt-00001.json",
        }

        # Tamper with the destination to prove the next host_egress doesn't
        # touch already-manifested files.
        await fs.write_file(f"{src_root}/restic/host/config", b"untouched")

        # Run host_egress with no new staging files — should be a no-op.
        await host_egress(staging_dir=str(staging), destination_dir=src_root)

        assert await fs.read_file(f"{src_root}/restic/host/config") == b"untouched"
