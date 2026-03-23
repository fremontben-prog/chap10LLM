"""
evaluate_ragas.py
-----------------
Évalue la qualité de la chaîne RAG sur des questions métier basketball
en utilisant le framework RAGAS.

Branché sur le vrai moteur RAG de MistralChat.py :
    - VectorStoreManager  pour la recherche FAISS
    - Mistral             pour la génération de réponses
    - log_evaluation_metrics() pour envoyer les scores dans Logfire

Métriques RAGAS :
    - Faithfulness       → LLM only (Mistral via LangchainLLMWrapper)
    - AnswerRelevancy    → ragas 0.4.x + Mistral + LangchainLLMWrapper ne supporte pas cette métrique sans OpenAI.
    - ContextPrecision   → LLM only
    - ContextRecall      → LLM only
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
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager
from monitoring.logfire_tracer import log_evaluation_metrics

from nba_engine import repondre_avec_agent, is_statistical_question

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
        question="Combien de points au total sur la saison 2014-2025 a marqué Shai Gilgeous-Alexander ?",
        ground_truth="Shai Gilgeous-Alexander a marqué 2485 points.",
        notes="Stat directe depuis colonne PTS",
    ),
    TestCase(
        id="S02", category="SIMPLE",
        question="Quel est le pourcentage à 3 points de Nikola Jokic ?",
        ground_truth="Nikola Jokic a un pourcentage à 3 points de 41.7%.",
        notes="Stat directe colonne 3P%",
    ),
    TestCase(
        id="S03", category="SIMPLE",
        question="Combien de rebonds au total Nikola Jokic a-t-il pris ?",
        ground_truth="Nikola Jokic a pris 889 rebonds.",
        notes="Stat directe colonne REB",
    ),

    # --- COMPLEXES ---
    TestCase(
        id="C01", category="COMPLEX",
        question="Quel joueur a marqué le plus de points ?",
        ground_truth="Le joueur qui a marqué le plus de points en moyenne cette saison 2024-2025 est Shai Gilgeous-Alexander avec une moyenne de 32,7 points par match.",
        notes="Max sur PTS et moyenne PTS",
    ),
    TestCase(
        id="C02", category="COMPLEX",
        question="Quels sont les 3 meilleurs scoreurs par match?",
        ground_truth=(
            "Shai Gilgeous-Alexander (OKC) : 32,7 points par match, Giannis Antetokounmpo (MIL) : 30,4 points par match, Nikola Jokić (DEN) : 29,6 points par match"
        ),
        notes="Top 3 PTS",
    ),
    TestCase(
        id="C03", category="COMPLEX",
        question="Quel joueur a le meilleur pourcentage à 3 points ?",
        ground_truth="Le joueur avec le meilleur pourcentage à 3 points en saison 2024-2025 est Seth Curry (Charlotte Hornets) avec un pourcentage de 45,6%",
        notes="Max 3P%",
    ),
    TestCase(
        id="C04", category="COMPLEX",
        question="Quel joueur a le plus de passes décisives en moyenne ?",
        ground_truth="Le joueur avec le plus de passes décisives en moyenne par match pour la saison 2024-2025 est Trae Young (Atlanta Hawks) avec 11,6 passes décisives par match.",
        notes="Max AST",
    ),

    # --- NOISY ---
    TestCase(
        id="N01", category="NOISY",
        question="sga cmb de pts",
        ground_truth="Shai Gilgeous-Alexander (SGA) a marqué 2 485 points lors de la saison régulière.",
        notes="Abréviation + SMS",
    ),
    TestCase(
        id="N02", category="NOISY",
        question="c ki le meilleur scoreur",
        ground_truth="Le meilleur scoreur de la saison NBA 2024-2025 est **Shai Gilgeous-Alexander** (OKC) avec une moyenne de 32,7 points par match.",
        notes="Question vague",
    ),
    TestCase(
        id="N03", category="NOISY",
        question="jokic cmb de reb",
        ground_truth="Nikola Jokić a une moyenne de 12,8 rebonds combinés",
        notes="Fautes + oral",
    ),
]


# ─────────────────────────────────────────────
# Connexion au vrai moteur RAG
# ─────────────────────────────────────────────

def load_answers_and_contexts(test_cases: List[TestCase]) -> List[TestCase]:
    """
    Utilise le vrai moteur de nba_engine.py :
        - is_statistical_question() → même routage que l'app
        - repondre_avec_agent()     → même réponse que l'app
    """
    logger.info("Chargement via nba_engine (même moteur que l'app)...")
    vector_store = VectorStoreManager()

    for tc in test_cases:
        logger.info(f"Traitement {tc.id} : {tc.question[:60]}...")
        try:
            # Même réponse que l'app réelle
            tc.answer = repondre_avec_agent(tc.question)

            # Contexte selon la route utilisée
            if is_statistical_question(tc.question):
                # SQL Tool → la réponse SQL est le contexte
                tc.contexts = [tc.answer]
            else:
                # RAG FAISS → vrais chunks récupérés
                results = vector_store.search(tc.question, k=SEARCH_K)
                tc.contexts = [r["text"] for r in results]

            logger.info(f"  ✓ ({len(tc.answer)} caractères)")

        except Exception as e:
            logger.error(f"Erreur cas {tc.id} : {e}")
            tc.answer = ""
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
            "Context Precision":round(float(row.get("context_precision",0) or 0), 3),
            "Context Recall":   round(float(row.get("context_recall",   0) or 0), 3),
            "Source": "SQL" if is_statistical_question(tc.question) else "RAG",
        })
    df = pd.DataFrame(records)
    df["Score Moyen"] = df[[
        "Faithfulness", 
        "Context Precision", "Context Recall",
    ]].mean(axis=1).round(3)
    return df.sort_values(["Catégorie", "ID"])


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("TABLEAU COMPARATIF DES SCORES RAGAS PAR CATÉGORIE")
    print("=" * 100)
    for cat, label in [("SIMPLE", "🟢 Simples"), ("COMPLEX", "🟡 Complexes"), ("NOISY", "🔴 Bruités")]:
        subset = df[df["Catégorie"] == cat]
        if subset.empty:
            continue
        print(f"\n{label}")
        print(subset[[
            "ID", "Question", "Faithfulness",
            "Context Precision", "Context Recall", "Score Moyen"
        ]].to_string(index=False))
    print("\n" + "-" * 100)
    print("MOYENNES GLOBALES PAR CATÉGORIE")
    print("-" * 100)
    summary = df.groupby("Catégorie")[[
        "Faithfulness", 
        "Context Precision", "Context Recall", "Score Moyen"
    ]].mean().round(3)
    print(summary.to_string())
    print("=" * 100 + "\n")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────

def run_evaluation(output_dir: str = "eval/results") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Réponses + contextes via le vrai RAG
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
    #    Premier lancement : télécharge ~90MB (sentence-transformers/all-MiniLM-L6-v2)
    logger.info("Chargement des embeddings HuggingFace (all-MiniLM-L6-v2)...")
    
    ragas_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    # 5. Métriques — ragas.metrics (compatible LangchainLLMWrapper)
    metrics = [
        Faithfulness(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    # 6. Évaluation RAGAS
    logger.info("Évaluation RAGAS en cours...")
    ragas_scores = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

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
    log_evaluation_metrics(df)
    logger.info("Métriques envoyées dans Logfire.")

    return df


if __name__ == "__main__":
    run_evaluation()