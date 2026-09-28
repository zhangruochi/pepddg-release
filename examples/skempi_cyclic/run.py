"""Verify the official SKEMPI snapshot and run a complete example mutation cohort."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request


def verify_file(path, expected):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def member_bytes(path, member):
    # Read exact members without extracting upstream paths or links.
    with tarfile.open(path, "r:gz") as archive:
        matches = [item for item in archive.getmembers() if item.name == member]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one archive member: {member}")
        if not matches[0].isfile():
            raise ValueError(f"expected regular archive member: {member}")
        with archive.extractfile(matches[0]) as stream:
            return stream.read()


def download(cache, name, source):
    destination = cache / name
    if destination.exists():
        verify_file(destination, source["sha256"])
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(source["url"], timeout=120) as response, temporary.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    verify_file(temporary, source["sha256"])
    temporary.replace(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, default=Path(".cache/skempi_cyclic"))
    parser.add_argument("--target", choices=("1SMF", "3EQS", "3EQY", "5XCO"), default="1SMF")
    parser.add_argument("--platform", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    base = Path(__file__).resolve().parent
    manifest = json.loads((base / "manifest.json").read_text())
    args.download_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: download(args.download_dir, name, source)
             for name, source in manifest["official_sources"].items()}
    target = manifest["targets"][args.target]
    for name, expected in target["source_members"].items():
        if hashlib.sha256(member_bytes(paths["SKEMPI2_PDBs.tgz"], name)).hexdigest() != expected:
            raise ValueError(f"official structure member changed: {name}")
    for name, expected in target["prepared_files"].items():
        verify_file(base / name, expected)
    if args.verify_only:
        print(f"Verified official and prepared {args.target} inputs ({target['n_mutations']} mutations).")
        return 0
    command = [sys.executable, "-m", "pepddg", "score-structures", "--structure",
               str(base / target["pdb"]), "--mutations", str(base / target["mutations"]),
               "--peptide-chain", target["peptide_chain"], "--receptor-chain", target["receptor_chain"],
               "--target", args.target, "--parent-id", target["parent_id"],
               "--closure", target["closure_kind"], "--n-restarts", "7", "--seed", "20260302",
               "--platform", args.platform, "--cpu-threads", "2", "--output", str(args.output)]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
