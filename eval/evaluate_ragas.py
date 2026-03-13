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
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager
from monitoring.logfire_tracer import log_evaluation_metrics

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
        question="Combien de points a marqué LeBron James lors du dernier match ?",
        ground_truth="LeBron James a marqué 28 points lors du dernier match.",
        notes="Stat directe, un joueur, un match",
    ),
    TestCase(
        id="S02", category="SIMPLE",
        question="Quel est le pourcentage aux tirs à 3 points de Stephen Curry cette saison ?",
        ground_truth="Stephen Curry affiche un pourcentage à 3 points de 42.8% cette saison.",
        notes="Stat de saison, un joueur",
    ),
    TestCase(
        id="S03", category="SIMPLE",
        question="Combien de rebonds par match réalise Nikola Jokic en moyenne ?",
        ground_truth="Nikola Jokic moyenne 12.4 rebonds par match cette saison.",
        notes="Moyenne de saison, stat simple",
    ),

    # --- COMPLEXES ---
    TestCase(
        id="C01", category="COMPLEX",
        question="Quel joueur a le meilleur pourcentage à 3 points sur les 5 derniers matchs ?",
        ground_truth=(
            "Sur les 5 derniers matchs, Klay Thompson affiche le meilleur "
            "pourcentage à 3 points avec 47.3%, devant Stephen Curry (44.1%) "
            "et Damian Lillard (39.8%)."
        ),
        notes="Agrégation sur fenêtre glissante, comparaison multi-joueurs",
    ),
    TestCase(
        id="C02", category="COMPLEX",
        question=(
            "Compare l'efficacité offensive (points + passes décisives) "
            "de Luka Doncic et Jayson Tatum ce mois-ci."
        ),
        ground_truth=(
            "Ce mois-ci, Luka Doncic cumule 31.2 points et 9.1 passes (40.3 total) "
            "contre 28.7 points et 4.5 passes pour Jayson Tatum (33.2 total)."
        ),
        notes="Calcul composite, comparaison deux joueurs",
    ),
    TestCase(
        id="C03", category="COMPLEX",
        question="Quelle équipe a la meilleure défense sur les 10 derniers matchs ?",
        ground_truth=(
            "Sur les 10 derniers matchs, les Boston Celtics sont la meilleure "
            "défense avec 102.3 points encaissés par match."
        ),
        notes="Agrégation équipe, fenêtre glissante longue",
    ),
    TestCase(
        id="C04", category="COMPLEX",
        question="Quel pivot a le meilleur ratio points/minutes jouées cette semaine ?",
        ground_truth=(
            "Cette semaine, Anthony Davis affiche le meilleur ratio "
            "points/minutes parmi les pivots titulaires avec 1.02 point par minute."
        ),
        notes="Filtre par poste, ratio calculé",
    ),

    # --- BRUITÉS ---
    TestCase(
        id="N01", category="NOISY",
        question="kari irving cmb de point il a mi hier ??",
        ground_truth="Kyrie Irving a inscrit 24 points lors du match d'hier.",
        notes="Fautes de frappe, langage SMS",
    ),
    TestCase(
        id="N02", category="NOISY",
        question="Qui est le meilleur joueur ?",
        ground_truth=(
            "La notion de meilleur joueur dépend du critère : "
            "Nikola Jokic mène en MVP votes, Stephen Curry en % à 3 points."
        ),
        notes="Question ambiguë sans critère",
    ),
    TestCase(
        id="N03", category="NOISY",
        question="Est-ce que les Lakers ont gagné ou perdu et combien de points de Lebron ?",
        ground_truth=(
            "Les Lakers ont gagné leur dernier match 118-112. "
            "LeBron James a contribué avec 26 points."
        ),
        notes="Double question, formulation maladroite",
    ),
]


# ─────────────────────────────────────────────
# Connexion au vrai moteur RAG
# ─────────────────────────────────────────────

def load_answers_and_contexts(test_cases: List[TestCase]) -> List[TestCase]:
    """
    Reproduit exactement la logique de MistralChat.py :
        - VectorStoreManager.search() → contextes FAISS
        - client.chat.complete()      → réponse générée
    """
    logger.info("Chargement du VectorStoreManager...")
    vector_store = VectorStoreManager()

    if vector_store.index is None:
        logger.error("Index FAISS non trouvé. Lance d'abord : python indexer.py")
        return test_cases

    logger.info(f"Index chargé : {vector_store.index.ntotal} vecteurs.")

    client = Mistral(api_key=MISTRAL_API_KEY)

    SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

    for tc in test_cases:
        logger.info(f"Traitement du cas {tc.id} : {tc.question[:60]}...")
        try:
            results = vector_store.search(tc.question, k=SEARCH_K)
            tc.contexts = [r["text"] for r in results]

            context_str = "\n\n---\n\n".join([
                f"Source: {r['metadata'].get('source', 'Inconnue')} "
                f"(Score: {r['score']:.1f}%)\nContenu: {r['text']}"
                for r in results
            ]) if results else "Aucune information pertinente trouvée."

            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": SYSTEM_PROMPT.format(
                    context_str=context_str,
                    question=tc.question,
                )}],
                temperature=0.1,
            )
            tc.answer = response.choices[0].message.content
            logger.info(f"  ✓ Réponse générée ({len(tc.answer)} caractères)")

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
    print("\n" + "=" * 100)
    print("TABLEAU COMPARATIF DES SCORES RAGAS PAR CATÉGORIE")
    print("=" * 100)
    for cat, label in [("SIMPLE", "🟢 Simples"), ("COMPLEX", "🟡 Complexes"), ("NOISY", "🔴 Bruités")]:
        subset = df[df["Catégorie"] == cat]
        if subset.empty:
            continue
        print(f"\n{label}")
        print(subset[[
            "ID", "Question", "Faithfulness", "Answer Relevancy",
            "Context Precision", "Context Recall", "Score Moyen"
        ]].to_string(index=False))
    print("\n" + "-" * 100)
    print("MOYENNES GLOBALES PAR CATÉGORIE")
    print("-" * 100)
    summary = df.groupby("Catégorie")[[
        "Faithfulness", "Answer Relevancy",
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