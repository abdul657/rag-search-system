from pathlib import Path

import requests

from models import Document

BASE_URL = "http://127.0.0.1:8000"
SAMPLE_FILE = Path(__file__).parent / "data" / "sample_documents.txt"


def load_sample_documents() -> list[Document]:
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Sample document file not found: {SAMPLE_FILE}")

    documents: list[Document] = []
    for index, line in enumerate(SAMPLE_FILE.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if text:
            documents.append(
                Document(
                    id=f"sample-{index + 1}",
                    text=text,
                    metadata={"source": "sample_documents.txt"},
                )
            )
    return documents


def main() -> None:
    documents = load_sample_documents()
    if not documents:
        raise RuntimeError("No non-empty sample documents were found.")

    response = requests.post(
        f"{BASE_URL}/index",
        json=[document.model_dump() for document in documents],
        timeout=60,
    )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
