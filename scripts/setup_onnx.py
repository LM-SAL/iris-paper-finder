"""Download or verify the pinned local all-MiniLM-L6-v2 ONNX model."""

import argparse
import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"
MODEL_DIR = MODEL_ROOT / "onnx"
URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
ARCHIVE_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
FILE_SHA256 = {
    "config.json": "b567c7d5a55b636c95186aaf993f9a8920842b7e05a9e703e68b23cab2c3a670",
    "model.onnx": "4f148ba8ae9c2c7fbee4af2b132db8d06c6a6545b47fc83bbb98c3d22b8393e6",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    "tokenizer_config.json": "7702051bbc4953b94d47fa1d61b42ed4cbb3c71b501a8dd7183a823f8bea1f20",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def problems() -> list[str]:
    failures = []
    for name, expected in FILE_SHA256.items():
        path = MODEL_DIR / name
        if not path.is_file():
            failures.append(f"missing {name}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch for {name}")
    return failures


def install() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=MODEL_ROOT, suffix=".tar.gz", delete=False) as stream:
        archive = Path(stream.name)
        with urllib.request.urlopen(URL) as response:
            while block := response.read(1024 * 1024):
                stream.write(block)
    try:
        if sha256(archive) != ARCHIVE_SHA256:
            msg = "Downloaded ONNX archive checksum mismatch"
            raise SystemExit(msg)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(MODEL_ROOT, filter="data")
    finally:
        archive.unlink(missing_ok=True)
    if failures := problems():
        raise SystemExit("Invalid extracted ONNX model: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    failures = problems()
    if not failures:
        print(f"ONNX model verified: {MODEL_DIR}")
        return
    if args.check:
        raise SystemExit("Invalid ONNX model: " + "; ".join(failures))
    install()
    print(f"ONNX model installed and verified: {MODEL_DIR}")


if __name__ == "__main__":
    main()
