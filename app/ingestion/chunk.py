from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings


@dataclass
class Chunk:
    section: str
    chunk_index: int
    content: str


def chunk_sections(sections: list[tuple[str, str]]) -> list[Chunk]:
    """Split each section into overlapping chunks, preserving the section label.

    Chunking per-section (not across the whole doc) keeps each chunk inside one
    semantic part of the filing, which makes the section metadata accurate for
    citations and filtering.
    """
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    idx = 0
    for section, content in sections:
        for piece in splitter.split_text(content):
            piece = piece.strip()
            if len(piece) < 50:  # drop near-empty fragments
                continue
            chunks.append(Chunk(section=section, chunk_index=idx, content=piece))
            idx += 1
    return chunks
