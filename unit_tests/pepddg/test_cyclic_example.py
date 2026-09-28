"""Official example refuses changed sources and unsafe archive membership."""
from pathlib import Path
import hashlib
import importlib.util
import io
import tarfile

import pytest

spec = importlib.util.spec_from_file_location("cyclic_example", Path(__file__).resolve().parents[2] / "examples/skempi_cyclic/run.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_changed_cached_source_is_refused(tmp_path):
    path = tmp_path / "source.csv"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        example.verify_file(path, hashlib.sha256(b"expected").hexdigest())


def test_archive_reads_only_the_exact_regular_member(tmp_path):
    path = tmp_path / "source.tgz"
    with tarfile.open(path, "w:gz") as archive:
        data = b"ATOM expected\n"
        info = tarfile.TarInfo("PDBs/1SMF.pdb")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("../../outside")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    assert example.member_bytes(path, "PDBs/1SMF.pdb") == data
    with pytest.raises(ValueError, match="regular"):
        example.member_bytes(path, "../../outside")
    assert not (tmp_path / "outside").exists()


def test_duplicate_archive_member_is_refused(tmp_path):
    path = tmp_path / "source.tgz"
    with tarfile.open(path, "w:gz") as archive:
        for data in [b"first", b"second"]:
            info = tarfile.TarInfo("PDBs/1SMF.pdb")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="exactly one"):
        example.member_bytes(path, "PDBs/1SMF.pdb")
