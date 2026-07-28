#!/usr/bin/env python3

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "golden"

FAULTS = {
    "syntax_missing_semicolon": (
        "c[i] = a[i] + b[i];",
        "c[i] = a[i] + b[i]",
    ),
    "functional_subtraction": (
        "c[i] = a[i] + b[i];",
        "c[i] = a[i] - b[i];",
    ),
    "indexing_off_by_one": (
        "c[i] = a[i] + b[i];",
        "c[i] = a[(i + 1) % VECTOR_SIZE] + b[i];",
    ),
    "interface_wrong_top_name": (
        "void vector_add(",
        "void vector_sum(",
    ),
}


def main() -> None:
    source = (GOLDEN / "src" / "vector_add.cpp").read_text(encoding="utf-8")

    for fault_name, (old, new) in FAULTS.items():
        destination = ROOT / "faults" / fault_name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(GOLDEN, destination)

        faulty_source = source.replace(old, new, 1)
        if faulty_source == source:
            raise RuntimeError(f"Fault replacement did not apply: {fault_name}")

        (destination / "src" / "vector_add.cpp").write_text(
            faulty_source,
            encoding="utf-8",
        )
        (destination / "fault.txt").write_text(
            f"fault_type={fault_name}\n",
            encoding="utf-8",
        )
        print(f"Created {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
