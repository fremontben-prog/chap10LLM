"""
indexer.py
----------
Orchestre le pipeline d'indexation complet avec :
    - Pydantic        : validation des flux d'entrée/sortie à chaque étape
    - Pydantic AI     : contrôle qualité sémantique des chunks
    - Pydantic Logfire: observabilité pas à pas de toute la chaîne
"""
# indexer.py — DOIT être la toute première ligne avant tous les imports
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging

from typing import Optional

import logfire
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

from utils.config import INPUT_DIR
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.vector_store import VectorStoreManager
from pipeline.data_pipeline import (
    validate_documents,
    chunk_documents,
    embed_chunks,
    LoadedDocument,
    EmbeddedChunk,
)
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─────────────────────────────────────────────
# Initialisation Logfire
# ─────────────────────────────────────────────

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="basketball-rag-indexer",
    service_version="1.0.0",
    environment=os.getenv("ENV", "development"),
)


# ─────────────────────────────────────────────
# Schémas Pydantic — validation entrée/sortie
# ─────────────────────────────────────────────

class IndexingConfig(BaseModel):
    """Valide la configuration avant de démarrer l'indexation."""
    input_directory: str
    data_url: Optional[str] = None

    @field_validator("input_directory")
    @classmethod
    def dir_must_be_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le répertoire d'entrée ne peut pas être vide.")
        return v


class IndexingReport(BaseModel):
    """Rapport final de l'indexation, validé par Pydantic."""
    input_directory:  str
    total_docs:       int = Field(ge=0)
    total_chunks:     int = Field(ge=0)
    total_embedded:   int = Field(ge=0)
    total_indexed:    int = Field(ge=0)
    chunks_rejected:  int = Field(ge=0, default=0)
    success:          bool = True
    error:            Optional[str] = None


class ChunkQualityResult(BaseModel):
    """Résultat de l'analyse qualité d'un chunk par Pydantic AI."""
    is_relevant: bool
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


# ─────────────────────────────────────────────
# Agent Pydantic AI — contrôle qualité sémantique
# ─────────────────────────────────────────────

quality_agent = Agent(
    model="mistral:mistral-small",
    result_type=ChunkQualityResult,
    system_prompt=(
        "Tu es un expert en données basketball. "
        "Évalue si le chunk de texte contient des données pertinentes "
        "(statistiques, résultats, joueurs, équipes, matchs). "
        "Retourne is_relevant (bool), reason (str), confidence (float 0-1)."
    ),
)


def filter_chunks_by_quality(
    chunks: list[EmbeddedChunk],
    confidence_threshold: float = 0.6,
    sample_size: int = 20,
) -> tuple[list[EmbeddedChunk], int]:
    """
    Utilise Pydantic AI pour évaluer la pertinence d'un échantillon de chunks.
    Retourne (chunks_valides, nb_rejetés).

    On échantillonne pour limiter les appels API — si l'échantillon
    est de mauvaise qualité, on logue un warning global.
    """
    with logfire.span("pydantic_ai.quality_check", sample_size=sample_size):
        sample = chunks[:sample_size]
        rejected_ids = set()

        for chunk in sample:
            try:
                result = quality_agent.run_sync(chunk.page_content[:300])
                qr: ChunkQualityResult = result.data

                logfire.info(
                    "chunk.quality",
                    chunk_id=chunk.chunk_id,
                    is_relevant=qr.is_relevant,
                    confidence=qr.confidence,
                    reason=qr.reason,
                )

                if not qr.is_relevant and qr.confidence >= confidence_threshold:
                    rejected_ids.add(chunk.chunk_id)
                    logfire.warning(
                        "chunk.rejected",
                        chunk_id=chunk.chunk_id,
                        reason=qr.reason,
                    )

            except Exception as e:
                logfire.error("chunk.quality_error", chunk_id=chunk.chunk_id, error=str(e))

        # Filtre global basé sur l'échantillon
        valid = [c for c in chunks if c.chunk_id not in rejected_ids]
        n_rejected = len(chunks) - len(valid)

        logfire.info(
            "quality_check.summary",
            total=len(chunks),
            valid=len(valid),
            rejected=n_rejected,
        )

    return valid, n_rejected


# ─────────────────────────────────────────────
# run_indexing — instrumenté Logfire
# ─────────────────────────────────────────────

def run_indexing(
    input_directory: str,
    data_url: Optional[str] = None,
) -> IndexingReport:
    """Exécute le processus complet d'indexation avec validation et observabilité."""

    # Validation de la config d'entrée
    config = IndexingConfig(input_directory=input_directory, data_url=data_url)

    with logfire.span("indexing.full_pipeline", input_dir=config.input_directory):
        logfire.info("indexing.start", config=config.model_dump())

        # ── Étape 1 : Téléchargement ──────────────────────────────────────
        with logfire.span("indexing.step1_download"):
            if config.data_url:
                logfire.info("indexing.download_start", url=config.data_url)
                success = download_and_extract_zip(config.data_url, config.input_directory)
                if not success:
                    logfire.error("indexing.download_failed", url=config.data_url)
                    return IndexingReport(
                        input_directory=input_directory,
                        total_docs=0, total_chunks=0,
                        total_embedded=0, total_indexed=0,
                        success=False, error="Échec du téléchargement.",
                    )
                logfire.info("indexing.download_success")
            else:
                logfire.info("indexing.no_download", reason="Aucune URL fournie, fichiers locaux utilisés.")

        # ── Étape 2 : Chargement + Validation Pydantic ────────────────────
        with logfire.span("indexing.step2_load_and_validate"):
            raw_docs = load_and_parse_files(config.input_directory)
            logfire.info("indexing.raw_docs_loaded", count=len(raw_docs))

            if not raw_docs:
                logfire.warning("indexing.no_documents")
                return IndexingReport(
                    input_directory=input_directory,
                    total_docs=0, total_chunks=0,
                    total_embedded=0, total_indexed=0,
                    success=False, error="Aucun document chargé.",
                )

            # Validation Pydantic — filtre les docs invalides
            validated_docs: list[LoadedDocument] = validate_documents(raw_docs)
            logfire.info(
                "indexing.validation_complete",
                raw=len(raw_docs),
                valid=len(validated_docs),
                rejected=len(raw_docs) - len(validated_docs),
            )

        # ── Étape 3 : Chunking ────────────────────────────────────────────
        with logfire.span("indexing.step3_chunking"):
            chunks = chunk_documents(validated_docs)
            logfire.info("indexing.chunking_complete", total_chunks=len(chunks))

        # ── Étape 4 : Embedding ───────────────────────────────────────────
        with logfire.span("indexing.step4_embedding"):
            embedded = embed_chunks(chunks)
            logfire.info("indexing.embedding_complete", total_embedded=len(embedded))

        # ── Étape 5 : Contrôle qualité Pydantic AI ────────────────────────
        with logfire.span("indexing.step5_quality_check"):
            valid_chunks, n_rejected = filter_chunks_by_quality(embedded)
            logfire.info(
                "indexing.quality_complete",
                valid=len(valid_chunks),
                rejected=n_rejected,
            )

        # ── Étape 6 : Indexation FAISS ────────────────────────────────────
        with logfire.span("indexing.step6_faiss"):
            vector_store = VectorStoreManager()
            vector_store.build_index(raw_docs)  # ton appel existant inchangé
            total_indexed = vector_store.index.ntotal if vector_store.index else 0
            logfire.info("indexing.faiss_complete", total_indexed=total_indexed)

        # ── Rapport final validé Pydantic ─────────────────────────────────
        report = IndexingReport(
            input_directory=input_directory,
            total_docs=len(validated_docs),
            total_chunks=len(chunks),
            total_embedded=len(embedded),
            total_indexed=total_indexed,
            chunks_rejected=n_rejected,
        )

        logfire.info("indexing.complete", report=report.model_dump())
        logging.info(f"--- Indexation terminée : {report.model_dump()} ---")

    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation RAG")
    parser.add_argument("--input-dir",  type=str, default=INPUT_DIR)
    parser.add_argument("--data-url",   type=str, default=None)
    args = parser.parse_args()

    report = run_indexing(
        input_directory=args.input_dir,
        data_url=args.data_url,
    )

    print("\n===== RAPPORT D'INDEXATION =====")
    for k, v in report.model_dump().items():
        print(f"  {k}: {v}")
