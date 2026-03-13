"""
data_pipeline.py
----------------
Pipeline de préparation des données : chunking + embedding.

S'appuie sur utils/data_loader.py pour l'extraction et le nettoyage
des fichiers (PDF/OCR, DOCX, TXT, CSV, Excel).

Ce module prend en charge uniquement :
    1. Validation Pydantic des documents chargés
    2. Chunking  (LangChain RecursiveCharacterTextSplitter)
    3. Embedding (Mistral mistral-embed)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mistralai import Mistral
from dotenv import load_dotenv

# Loader existant — extraction déjà gérée
from utils.data_loader import load_and_parse_files

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBED_MODEL     = "mistral-embed"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64


# ─────────────────────────────────────────────
# Schémas Pydantic — validation des flux
# ─────────────────────────────────────────────

class LoadedDocument(BaseModel):
    """
    Valide la sortie de load_and_parse_files() avant de l'injecter
    dans le pipeline.
    """
    page_content: str
    source: str       = Field(default="unknown")
    filename: str     = Field(default="unknown")
    category: str     = Field(default="root")
    full_path: str    = Field(default="")
    sheet: Optional[str] = None

    @field_validator("page_content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError(f"Contenu trop court : '{v[:50]}'")
        return v.strip()

    @classmethod
    def from_loader_dict(cls, d: dict) -> "LoadedDocument":
        """Construit un LoadedDocument depuis un dict retourné par load_and_parse_files."""
        meta = d.get("metadata", {})
        return cls(
            page_content=d.get("page_content", ""),
            source=meta.get("source", "unknown"),
            filename=meta.get("filename", "unknown"),
            category=meta.get("category", "root"),
            full_path=meta.get("full_path", ""),
            sheet=meta.get("sheet"),
        )


class Chunk(BaseModel):
    """Chunk de texte issu du splitting, prêt à être embedé."""
    chunk_id:    str
    source:      str
    filename:    str
    category:    str
    page_content: str
    char_count:  int = 0
    embedding:   Optional[List[float]] = None

    @model_validator(mode="after")
    def set_char_count(self) -> "Chunk":
        self.char_count = len(self.page_content)
        return self

    @field_validator("page_content")
    @classmethod
    def chunk_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Un chunk ne peut pas être vide.")
        return v


class EmbeddedChunk(Chunk):
    """Chunk avec embedding validé."""
    embedding: List[float]

    @field_validator("embedding")
    @classmethod
    def embedding_valid(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("L'embedding ne peut pas être vide.")
        return v


class PipelineOutput(BaseModel):
    """Sortie complète et validée du pipeline."""
    input_dir:      str
    total_docs:     int
    total_chunks:   int
    total_embedded: int
    chunks:         List[EmbeddedChunk]
    processed_at:   datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Étape 1 : Validation des documents chargés
# ─────────────────────────────────────────────

def validate_documents(raw_docs: List[dict]) -> List[LoadedDocument]:
    """
    Valide les documents retournés par load_and_parse_files().
    Les documents invalides (trop courts, vides) sont ignorés avec un warning.
    """
    logger.info(f"Validation de {len(raw_docs)} documents...")
    validated = []
    for d in raw_docs:
        try:
            validated.append(LoadedDocument.from_loader_dict(d))
        except Exception as e:
            logger.warning(f"Document ignoré ({d.get('metadata', {}).get('source', '?')}) : {e}")

    logger.info(f"{len(validated)}/{len(raw_docs)} documents valides.")
    return validated


# ─────────────────────────────────────────────
# Étape 2 : Chunking
# ─────────────────────────────────────────────

def chunk_documents(docs: List[LoadedDocument]) -> List[Chunk]:
    """Découpe les documents validés en chunks."""
    logger.info(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[Chunk] = []

    for doc in docs:
        parts = splitter.split_text(doc.page_content)
        for i, part in enumerate(parts):
            try:
                chunk = Chunk(
                    chunk_id=f"{Path(doc.filename).stem}_c{len(chunks)+i}",
                    source=doc.source,
                    filename=doc.filename,
                    category=doc.category,
                    page_content=part,
                )
                chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Chunk ignoré ({doc.filename}, chunk {i}) : {e}")

    logger.info(f"{len(chunks)} chunks créés.")
    return chunks


# ─────────────────────────────────────────────
# Étape 3 : Embedding
# ─────────────────────────────────────────────

def embed_chunks(chunks: List[Chunk], batch_size: int = 32) -> List[EmbeddedChunk]:
    """Génère les embeddings Mistral par batch et valide chaque résultat."""
    logger.info(f"Embedding de {len(chunks)} chunks (batch={batch_size})...")

    client = Mistral(api_key=MISTRAL_API_KEY)

    embedded: List[EmbeddedChunk] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c.page_content for c in batch]
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=texts)
            for chunk, emb_obj in zip(batch, response.data):
                try:
                    embedded.append(EmbeddedChunk(
                        **chunk.model_dump(exclude={"embedding"}),
                        embedding=emb_obj.embedding,
                    ))
                except Exception as e:
                    logger.warning(f"Chunk {chunk.chunk_id} ignoré : {e}")
        except Exception as e:
            logger.error(f"Erreur batch {i // batch_size + 1} : {e}")

        logger.info(f"  Batch {i // batch_size + 1} — {len(embedded)} chunks traités")

    logger.info(f"{len(embedded)} chunks embedés.")
    return embedded


# ─────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────

def run_pipeline(input_dir: str, output_dir: str = "pipeline/output") -> PipelineOutput:
    """
    Exécute le pipeline complet :
        load_and_parse_files()  <- data_loader.py
            -> validate_documents()
            -> chunk_documents()
            -> embed_chunks()
            -> PipelineOutput (validé Pydantic)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extraction déléguée à data_loader.py
    raw_docs  = load_and_parse_files(input_dir)

    # Pipeline Pydantic
    docs      = validate_documents(raw_docs)
    chunks    = chunk_documents(docs)
    embedded  = embed_chunks(chunks)

    output = PipelineOutput(
        input_dir=input_dir,
        total_docs=len(docs),
        total_chunks=len(chunks),
        total_embedded=len(embedded),
        chunks=embedded,
    )

    # Sauvegarde du résumé (sans les vecteurs)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"pipeline_output_{ts}.json")
    summary  = output.model_dump(exclude={"chunks": {"__all__": {"embedding"}}})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Pipeline terminé — {len(embedded)} chunks. Résumé : {out_path}")
    return output


if __name__ == "__main__":
    import sys
    from utils.config import (
        INPUT_DIR,VECTOR_DB_DIR
    )
    folder_input = sys.argv[1] if len(sys.argv) > 1 else f"{INPUT_DIR}/"
    folder_output = sys.argv[2] if len(sys.argv) > 1 else f"{VECTOR_DB_DIR}/"
    
    run_pipeline(folder_input, folder_output)
