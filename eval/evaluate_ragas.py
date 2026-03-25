"""
evaluate_ragas.py
-----------------
Évalue la qualité de la chaîne RAG+SQL sur des questions métier basketball
en utilisant le framework RAGAS.

Branché sur le vrai moteur nba_engine.py :
    - repondre_avec_contextes() → même moteur hybride que l'app
    - SQL Tool + RAG FAISS → contextes complets pour RAGAS
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import logging
from typing import List, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

import pandas as pd
from dotenv import load_dotenv

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from nba_engine import repondre_avec_contextes, is_statistical_question

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────

class TestCase(BaseModel):
    id:           str
    category:     str = Field(..., pattern="^(SIMPLE|COMPLEX|NOISY)$")
    question:     str
    ground_truth: str
    contexts:     List[str] = Field(default_factory=list)
    answer:       str = ""
    notes:        str = ""


# ─────────────────────────────────────────────
# Jeu de questions métier basketball
# ─────────────────────────────────────────────

TEST_CASES: List[TestCase] = [

    # --- SIMPLES ---
    TestCase(
        id="S01", category="SIMPLE",
        question="Combien de points par match marque Shai Gilgeous-Alexander ?",
        ground_truth=(
            "Shai Gilgeous-Alexander marque 32.7 points par match cette saison, "
            "pour un total de 2485 points en 76 matchs."
        ),
        notes="Source : SQL v_top_scorers — pts_avg",
    ),
    TestCase(
        id="S02", category="SIMPLE",
        question="Quel joueur a le meilleur pourcentage à 3 points cette saison ?",
        ground_truth=(
            "Le joueur avec le meilleur pourcentage à 3 points cette saison est "
            "Seth Curry (CHA) avec 45.6% sur 184 tentatives."
        ),
        notes="Source : SQL v_top_three_point",
    ),
    TestCase(
        id="S03", category="SIMPLE",
        question="Quel est le pourcentage à 3 points de Nikola Jokic ?",
        ground_truth=(
            "Nikola Jokic a un pourcentage à 3 points de 41.7% cette saison."
        ),
        notes="Source : SQL v_player_stats — three_pct",
    ),

    # --- COMPLEXES ---
    TestCase(
        id="C01", category="COMPLEX",
        question="Quel joueur a marqué le plus de points cette saison ?",
        ground_truth=(
            "Shai Gilgeous-Alexander (OKC) est le meilleur scoreur de la saison 2024-2025 "
            "avec 2485 points au total, soit 32.7 points par match."
        ),
        notes="Source : SQL v_top_scorers — pts_avg DESC",
    ),
    TestCase(
        id="C02", category="COMPLEX",
        question="Compare les statistiques de LeBron James et Nikola Jokic cette saison.",
        ground_truth=(
            "LeBron James et Nikola Jokic sont deux des meilleurs joueurs cette saison. "
            "Jokic affiche 29.6 points, 12.5 rebonds et 10.0 passes par match. "
            "LeBron James affiche environ 24 points, 8 rebonds et 8 passes par match."
        ),
        notes="Source : SQL v_player_stats — comparaison deux joueurs",
    ),
    TestCase(
        id="C03", category="COMPLEX",
        question="Quelle équipe a le meilleur Net Rating cette saison ?",
        ground_truth=(
            "L'équipe avec le meilleur Net Rating cette saison est "
            "Oklahoma City Thunder (OKC)."
        ),
        notes="Source : SQL v_team_stats — avg_netrtg DESC",
    ),
    TestCase(
        id="C04", category="COMPLEX",
        question="Qui a le plus de triple-doubles cette saison ?",
        ground_truth=(
            "Nikola Jokic est le joueur avec le plus de triple-doubles cette saison."
        ),
        notes="Source : SQL v_player_stats — td3 DESC",
    ),

    # --- BRUITÉS ---
    TestCase(
        id="N01", category="NOISY",
        question="sga cmb de pts",
        ground_truth=(
            "Shai Gilgeous-Alexander marque 32.7 points par match "
            "pour un total de 2485 points cette saison."
        ),
        notes="Abréviation SGA + cmb — source SQL",
    ),
    TestCase(
        id="N02", category="NOISY",
        question="c ki le meilleur scoreur",
        ground_truth=(
            "Le meilleur scoreur cette saison est Shai Gilgeous-Alexander (OKC) "
            "avec 32.7 points par match."
        ),
        notes="Langage SMS — source SQL",
    ),
    TestCase(
        id="N03", category="NOISY",
        question="jokic cmb de reb",
        ground_truth=(
            "Nikola Jokic a une moyenne de 12.5 rebonds par match cette saison, "
            "dont 3.1 offensifs et 9.4 défensifs."
        ),
        notes="Abréviation + oral — source SQL oreb_avg dreb_avg",
    ),
]


# ─────────────────────────────────────────────
# Connexion au vrai moteur hybride
# ─────────────────────────────────────────────

def load_answers_and_contexts(test_cases: List[TestCase]) -> List[TestCase]:
    """
    Utilise le moteur hybride de nba_engine.py :
        - Questions statistiques → SQL + RAG FAISS
        - Questions narratives   → RAG FAISS seul
    Retourne réponse ET contextes pour RAGAS.
    """
    logger.info("Chargement via nba_engine (mode hybride SQL + RAG)...")

    for tc in test_cases:
        logger.info(f"Traitement {tc.id} : {tc.question[:60]}...")
        try:
            tc.answer, tc.contexts = repondre_avec_contextes(tc.question)
            logger.info(
                f"  ✓ réponse={len(tc.answer)} chars, "
                f"contextes={len(tc.contexts)}"
            )
        except Exception as e:
            logger.error(f"Erreur cas {tc.id} : {e}")
            tc.answer   = ""
            tc.contexts = []

    return test_cases


# ─────────────────────────────────────────────
# Pipeline RAGAS
# ─────────────────────────────────────────────

def build_ragas_dataset(test_cases: List[TestCase]) -> Tuple[Dataset, List[TestCase]]:
    """Construit le Dataset HuggingFace attendu par RAGAS."""
    valid = [tc for tc in test_cases if tc.answer and tc.contexts]
    logger.info(f"{len(valid)}/{len(test_cases)} cas valides pour RAGAS.")
    dataset = Dataset.from_dict({
        "question":    [tc.question     for tc in valid],
        "answer":      [tc.answer       for tc in valid],
        "contexts":    [tc.contexts     for tc in valid],
        "ground_truth":[tc.ground_truth for tc in valid],
    })
    return dataset, valid


def build_results_table(
    test_cases: List[TestCase],
    ragas_scores,
) -> pd.DataFrame:
    scores_df = ragas_scores.to_pandas()
    records = []
    for i, tc in enumerate(test_cases):
        row = scores_df.iloc[i] if i < len(scores_df) else {}
        records.append({
            "ID":               tc.id,
            "Catégorie":        tc.category,
            "Question":         tc.question[:80] + "..." if len(tc.question) > 80 else tc.question,
            "Notes":            tc.notes,
            "Faithfulness":     round(float(row.get("faithfulness",     0) or 0), 3),
            "Answer Relevancy": round(float(row.get("answer_relevancy", 0) or 0), 3),
            "Context Precision":round(float(row.get("context_precision",0) or 0), 3),
            "Context Recall":   round(float(row.get("context_recall",   0) or 0), 3),
        })
    df = pd.DataFrame(records)
    df["Score Moyen"] = df[[
        "Faithfulness", "Answer Relevancy",
        "Context Precision", "Context Recall",
    ]].mean(axis=1).round(3)
    return df.sort_values(["Catégorie", "ID"])


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 110)
    print("TABLEAU COMPARATIF DES SCORES RAGAS PAR CATÉGORIE")
    print("=" * 110)
    for cat, label in [("SIMPLE", "🟢 Simples"), ("COMPLEX", "🟡 Complexes"), ("NOISY", "🔴 Bruités")]:
        subset = df[df["Catégorie"] == cat]
        if subset.empty:
            continue
        print(f"\n{label}")
        print(subset[[
            "ID", "Question", "Faithfulness", "Answer Relevancy",
            "Context Precision", "Context Recall", "Score Moyen"
        ]].to_string(index=False))
    print("\n" + "-" * 110)
    print("MOYENNES GLOBALES PAR CATÉGORIE")
    print("-" * 110)
    summary = df.groupby("Catégorie")[[
        "Faithfulness", "Answer Relevancy",
        "Context Precision", "Context Recall", "Score Moyen"
    ]].mean().round(3)
    print(summary.to_string())
    print("=" * 110 + "\n")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────

def run_evaluation(output_dir: str = "eval/results") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Réponses + contextes via le moteur hybride
    test_cases = load_answers_and_contexts(TEST_CASES)

    # 2. Dataset RAGAS
    dataset, valid_cases = build_ragas_dataset(test_cases)
    if len(dataset) == 0:
        logger.error("Aucun cas valide — vérifiez l'index FAISS et la clé API.")
        return pd.DataFrame()

    # 3. LLM Mistral via LangchainLLMWrapper
    ragas_llm = LangchainLLMWrapper(
        ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, model="mistral-small")
    )

    # 4. Embeddings HuggingFace locaux pour AnswerRelevancy
    logger.info("Chargement des embeddings HuggingFace (all-MiniLM-L6-v2)...")
    ragas_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. Métriques
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    # 6. Évaluation RAGAS
    logger.info("Évaluation RAGAS en cours...")
    ragas_scores = evaluate(dataset=dataset, metrics=metrics)

    # 7. Tableau comparatif
    df = build_results_table(valid_cases, ragas_scores)
    print_summary(df)

    # 8. Sauvegarde CSV + JSON
    csv_path  = os.path.join(output_dir, f"ragas_results_{timestamp}.csv")
    json_path = os.path.join(output_dir, f"ragas_results_{timestamp}.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    logger.info(f"Résultats sauvegardés : {csv_path}")

    # 9. Envoi Logfire
    from monitoring.logfire_tracer import log_evaluation_metrics
    log_evaluation_metrics(df)
    logger.info("Métriques envoyées dans Logfire.")

    return df


if __name__ == "__main__":
    run_evaluation()