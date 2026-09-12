"""Build a deterministic archive of the participant description and exercise kit."""

import argparse
import hashlib
from pathlib import Path
import runpy
import tempfile
import zipfile


def build_archive(output: Path) -> str:
    root = Path(__file__).resolve().parent
    build_kit = runpy.run_path(str(root / "kda_a3_exercise/build_package.py"))["build_archive"]
    entries = {name: (root / name).read_bytes() for name in ("README.md", "build_package.py", ".gitignore")}
    with tempfile.TemporaryDirectory(prefix="kda_a3_package_") as staging:
        kit_path = Path(staging) / "kit.zip"
        build_kit(kit_path)
        with zipfile.ZipFile(kit_path) as kit:
            for name in kit.namelist():
                entries[name] = kit.read(name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo("kda_a3_practice/" + name, date_time=(2026, 9, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "  " + output.name + "\n")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_archive(args.output))
