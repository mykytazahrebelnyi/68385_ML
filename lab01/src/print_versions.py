import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

REQ = Path(__file__).resolve().parents[2] / "requirements.txt"
print(REQ)
print(__file__)

print(sys.version)
print(sys.version.split())


def main():
    lines = [f"python=={sys.version.split()[0]}"]
    for raw in REQ.read_text().splitlines():
        name = raw.split("==")[0].strip()
        if not name or name.startswith("#"):
            continue
        try:
            lines.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            lines.append(f"{name}==NOT INSTALLED")
    out = "\n".join(lines)
    print(out)
    res = Path(__file__).resolve().parents[1] / "results" / "versions.txt"
    res.write_text(out + "\n")


if __name__ == "__main__":
    main()
