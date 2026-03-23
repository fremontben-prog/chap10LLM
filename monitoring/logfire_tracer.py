"""
logfire_tracer.py
-----------------
Intègre Pydantic Logfire pour observer la chaîne RAG/LLM pas à pas :

    query reçue
        → [RETRIEVAL]  recherche vectorielle FAISS
        → [RERANKING]  tri des contextes
        → [GENERATION] appel Mistral
        → [RESPONSE]   réponse finale

Chaque étape est instrumentée avec des spans Logfire contenant
les métriques clés (scores, tokens, latences).
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logfire
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Initialisation Logfire
# ─────────────────────────────────────────────

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),           # depuis .env
    service_name="basketball-rag",
    service_version="1.0.0",
    environment=os.getenv("ENV", "development"),
)


# ─────────────────────────────────────────────
# Schémas Pydantic — traces structurées
# ─────────────────────────────────────────────

class RetrievalResult(BaseModel):
    """Résultat d'un contexte récupéré."""
    chunk_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    source: str


class GenerationInput(BaseModel):
    """Entrée de la génération LLM."""
    query: str
    contexts: List[RetrievalResult]
    prompt_tokens_estimate: int = 0


class GenerationOutput(BaseModel):
    """Sortie de la génération LLM."""
    answer: str
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class RAGTrace(BaseModel):
    """Trace complète d'un cycle RAG."""
    query: str
    retrieval_results: List[RetrievalResult]
    generation_input: GenerationInput
    generation_output: GenerationOutput
    total_latency_ms: float
    success: bool = True
    error: Optional[str] = None


# ─────────────────────────────────────────────
# Tracer — instrumenté Logfire
# ─────────────────────────────────────────────

class RAGTracer:
    """
    Enveloppe la chaîne RAG et instrumente chaque étape avec Logfire.

    Utilisation :
        tracer = RAGTracer(rag_engine)
        response = tracer.run("Quel joueur a le meilleur % à 3pts ?")
    """

    def __init__(self, rag_engine: Any):
        """
        rag_engine : ton objet RAG existant qui expose :
            - .retrieve(query, k) -> List[dict]
            - .generate(query, contexts) -> dict
        """
        self.rag_engine = rag_engine

    @logfire.instrument("rag.full_pipeline")
    def run(self, query: str, k: int = 5) -> RAGTrace:
        """Exécute et trace la chaîne RAG complète."""
        start_total = time.time()

        logfire.info("rag.query_received", query=query, k=k)

        try:
            # Étape 1 : Retrieval
            retrieval_results = self._retrieve(query, k)

            # Étape 2 : Reranking
            ranked_results = self._rerank(retrieval_results)

            # Étape 3 : Génération
            gen_input  = GenerationInput(
                query=query,
                contexts=ranked_results,
                prompt_tokens_estimate=sum(len(r.text.split()) for r in ranked_results),
            )
            gen_output = self._generate(gen_input)

            total_ms = (time.time() - start_total) * 1000

            trace = RAGTrace(
                query=query,
                retrieval_results=ranked_results,
                generation_input=gen_input,
                generation_output=gen_output,
                total_latency_ms=round(total_ms, 2),
            )

            logfire.info(
                "rag.pipeline_complete",
                total_latency_ms=trace.total_latency_ms,
                chunks_retrieved=len(retrieval_results),
                chunks_used=len(ranked_results),
                answer_length=len(gen_output.answer),
            )

            return trace

        except Exception as e:
            total_ms = (time.time() - start_total) * 1000
            logfire.error("rag.pipeline_error", error=str(e), query=query)
            raise

    @logfire.instrument("rag.retrieval")
    def _retrieve(self, query: str, k: int) -> List[RetrievalResult]:
        """Étape de recherche vectorielle."""
        start = time.time()

        raw_results = self.rag_engine.retrieve(query, k=k)

        results = []
        for r in raw_results:
            try:
                result = RetrievalResult(
                    chunk_id=r.get("chunk_id", "unknown"),
                    text=r.get("text", ""),
                    score=float(r.get("score", 0.0)),
                    source=r.get("source", ""),
                )
                results.append(result)
            except Exception as e:
                logfire.warning("rag.retrieval.invalid_chunk", error=str(e))

        latency_ms = (time.time() - start) * 1000

        logfire.info(
            "rag.retrieval.complete",
            chunks_found=len(results),
            latency_ms=round(latency_ms, 2),
            top_score=results[0].score if results else 0.0,
            min_score=results[-1].score if results else 0.0,
        )

        return results

    @logfire.instrument("rag.reranking")
    def _rerank(
        self,
        results: List[RetrievalResult],
        score_threshold: float = 0.3,
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        """Filtre et trie les contextes par score de pertinence."""
        filtered = [r for r in results if r.score >= score_threshold]
        ranked   = sorted(filtered, key=lambda r: r.score, reverse=True)[:top_k]

        logfire.info(
            "rag.reranking.complete",
            before=len(results),
            after=len(ranked),
            threshold=score_threshold,
            top_scores=[round(r.score, 3) for r in ranked],
        )

        return ranked

    @logfire.instrument("rag.generation")
    def _generate(self, gen_input: GenerationInput) -> GenerationOutput:
        """Appel au LLM Mistral avec les contextes récupérés."""
        start = time.time()

        raw = self.rag_engine.generate(
            query=gen_input.query,
            contexts=[r.text for r in gen_input.contexts],
        )

        latency_ms = (time.time() - start) * 1000

        output = GenerationOutput(
            answer=raw.get("answer", ""),
            completion_tokens=raw.get("completion_tokens", 0),
            total_tokens=raw.get("total_tokens", 0),
            latency_ms=round(latency_ms, 2),
        )

        logfire.info(
            "rag.generation.complete",
            latency_ms=output.latency_ms,
            completion_tokens=output.completion_tokens,
            total_tokens=output.total_tokens,
            answer_preview=output.answer[:100],
        )

        return output


# ─────────────────────────────────────────────
# Métriques agrégées (dashboard Logfire)
# ─────────────────────────────────────────────

def log_evaluation_metrics(results_df: Any) -> None:
    """
    Envoie les métriques RAGAS dans Logfire après une session d'évaluation.
    Appelle cette fonction depuis evaluate_ragas.py après run_evaluation().

    results_df : DataFrame retourné par build_results_table()
    """
    with logfire.span("ragas.evaluation_summary"):
        for _, row in results_df.iterrows():
            logfire.info(
                "ragas.case_score",
                case_id=row["ID"],
                category=row["Catégorie"],
                faithfulness=row["Faithfulness"],
                context_precision=row["Context Precision"],
                context_recall=row["Context Recall"],
                score_moyen=row["Score Moyen"],
            )

        # Moyennes par catégorie
        for cat in ["SIMPLE", "COMPLEX", "NOISY"]:
            subset = results_df[results_df["Catégorie"] == cat]
            if not subset.empty:
                logfire.info(
                    "ragas.category_avg",
                    category=cat,
                    avg_faithfulness=round(subset["Faithfulness"].mean(), 3),
                    avg_context_precision=round(subset["Context Precision"].mean(), 3),
                    avg_context_recall=round(subset["Context Recall"].mean(), 3),
                    avg_score=round(subset["Score Moyen"].mean(), 3),
                )


# ─────────────────────────────────────────────
# Mock RAG Engine — pour tests standalone
# ─────────────────────────────────────────────

class MockRAGEngine:
    """Simule un moteur RAG pour tester le tracer sans infrastructure."""

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        return [
            {"chunk_id": f"chunk_{i}", "text": f"Contexte {i} pour : {query}",
             "score": round(0.9 - i * 0.1, 2), "source": f"doc_{i}.pdf"}
            for i in range(k)
        ]

    def generate(self, query: str, contexts: List[str]) -> Dict:
        time.sleep(0.1)  # simule latence LLM
        return {
            "answer": f"Réponse générée pour : {query}",
            "completion_tokens": 42,
            "total_tokens": 256,
        }


if __name__ == "__main__":
    engine = MockRAGEngine()
    tracer = RAGTracer(engine)

    test_query = "Quel joueur a le meilleur % à 3 points sur les 5 derniers matchs ?"
    trace = tracer.run(test_query)

    print(f"\nQuery     : {trace.query}")
    print(f"Contextes : {len(trace.retrieval_results)} récupérés")
    print(f"Réponse   : {trace.generation_output.answer}")
    print(f"Latence   : {trace.total_latency_ms} ms")
    print("\n✅ Traces disponibles dans le dashboard Logfire.")
