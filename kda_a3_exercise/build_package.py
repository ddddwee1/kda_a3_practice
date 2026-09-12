"""Build a deterministic archive containing only the public exercise kit."""

import argparse
import hashlib
from pathlib import Path
import zipfile


def build_archive(output: Path) -> str:
    root = Path(__file__).resolve().parent
    names = (
        "__init__.py", "auxiliary.py", "build_package.py", "cases.json", "cases.py", "check.py",
        "contract.json", "contract.py", "golden.py", "host_guard.py", "EVALUATION.md", "PROVENANCE.md", "README.md",
        "requirements.txt", "submission.py", "tests/conftest.py",
        "tests/test_checker.py", "tests/test_golden.py", "tests/test_host_guard.py", "tests/test_auxiliary.py", "upstream/__init__.py",
        "upstream/fla_naive.py", "upstream/LICENSE.fla", "upstream/SOURCES.json",
    )
    files = [root / name for name in sorted(names)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError("missing package source: " + str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = "kda_a3_exercise/" + path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "  " + output.name + "\n")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_archive(args.output))
