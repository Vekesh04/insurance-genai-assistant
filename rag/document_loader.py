from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from config import BASE_DIR

POLICY_DOC_DIR = BASE_DIR / "data" / "policy_documents"
SECTION_RE = re.compile(r"(?m)^\s*(\d+)\.\s+([A-Z][A-Z /\-]+)\s*$")


@dataclass
class PolicyChunk:
    chunk_id: str
    product_name: str
    policy_type: str
    section_number: str
    section_title: str
    text: str
    source_file: str

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_header(text: str, path: Path) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    product = lines[0] if lines else path.stem.replace("_", " ")
    policy_type = "Unknown"
    for line in lines[:5]:
        if line.lower().startswith("policy type:"):
            policy_type = line.split(":", 1)[1].strip()
            break
    return product, policy_type


def load_policy_document(path: Path) -> list[PolicyChunk]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    product_name, policy_type = _parse_header(text, path)
    matches = list(SECTION_RE.finditer(text))
    chunks: list[PolicyChunk] = []

    if not matches:
        chunks.append(
            PolicyChunk(
                chunk_id=f"{path.stem}:0",
                product_name=product_name,
                policy_type=policy_type,
                section_number="0",
                section_title="DOCUMENT",
                text=text,
                source_file=path.name,
            )
        )
        return chunks

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        section_number = match.group(1)
        section_title = match.group(2).strip()
        full_text = f"{product_name}. Section {section_number}: {section_title}. {body}".strip()
        chunks.append(
            PolicyChunk(
                chunk_id=f"{path.stem}:{section_number}",
                product_name=product_name,
                policy_type=policy_type,
                section_number=section_number,
                section_title=section_title,
                text=full_text,
                source_file=path.name,
            )
        )
    return chunks


def load_all_policy_chunks(directory: Path = POLICY_DOC_DIR) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    for path in sorted(directory.glob("*.txt")):
        chunks.extend(load_policy_document(path))
    return chunks
