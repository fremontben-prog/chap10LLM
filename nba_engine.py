"""
nba_engine.py
-------------
Moteur métier NBA — logique pure sans Streamlit.
Importable par MistralChat.py ET evaluate_ragas.py.

Architecture HYBRIDE :
    Questions statistiques → SQL Tool + RAG FAISS → Mistral synthèse
    Questions narratives   → RAG FAISS seul       → Mistral synthèse
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import logging
from functools import lru_cache

from dotenv import load_dotenv
from mistralai import Mistral
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager
from database.sql_tool import get_sql_tool

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "basketball.db")

SYSTEM_PROMPT_HYBRIDE = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue NBA.
Tu disposes de deux sources complémentaires pour répondre.

DONNÉES STATISTIQUES (SOURCE PRIORITAIRE — FIABLE) :
---
{sql_context}
---

CONTEXTE DOCUMENTAIRE (SOURCE SECONDAIRE — PEUT ÊTRE IMPRÉCISE) :
---
{rag_context}
---

RÈGLES :
- Les données statistiques sont la source de vérité pour tous les chiffres
- Si une valeur chiffrée est présente dans les données statistiques, ignore toute valeur différente dans le contexte documentaire
- Ne jamais mentionner deux valeurs contradictoires
- Commence toujours par les chiffres issus des données statistiques
- Enrichis avec le contexte narratif uniquement si cohérent avec les données
- Ne mentionne jamais les sources techniques (SQL, FAISS, Excel, base de données)
- Si une source est vide → utilise uniquement l'autre
- Ne jamais inventer de données
- Réponds en français, de façon claire et engageante

QUESTION : {question}
RÉPONSE :"""

SYSTEM_PROMPT_RAG = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

CONTEXTE DOCUMENTAIRE :
---
{context_str}
---

QUESTION : {question}
RÉPONSE :"""

SYSTEM_PROMPT_AGENT = """Tu es 'NBA Analyst AI', un assistant expert sur la NBA.
Tu as accès à un outil SQL nommé basketball_sql_query pour interroger une base SQLite contenant les statistiques NBA saison 2024-2025.

OBJECTIF :
Répondre précisément aux questions en utilisant les données SQL quand nécessaire.

RÈGLES :

1. Quand utiliser SQL
- Utilise l'outil basketball_sql_query pour toute question impliquant :
  - des statistiques (points, rebonds, passes, %, etc.)
  - des classements (meilleur, top, max, min)
  - des comparaisons entre joueurs
- N'utilise PAS SQL pour les questions générales ou narratives

2. Génération SQL
- Utilise toujours les tables : players et season_stats
- Jointure obligatoire :
  JOIN players ON season_stats.player_id = players.player_id
- Utilise la colonne full_name_normalized pour filtrer les joueurs
- Les noms doivent être en minuscules et sans accents

3. Filtrage des joueurs
- Si le nom est complet → utilise '='
  WHERE full_name_normalized = 'giannis antetokounmpo'
- Si seul le prénom ou nom partiel est fourni → utilise LIKE
  WHERE full_name_normalized LIKE '%giannis%'

4. Colonnes disponibles dans v_player_stats
- Moyennes : pts_avg, reb_avg, oreb_avg, dreb_avg, ast_avg,
             stl_avg, blk_avg, tov_avg, pf_avg, fp_avg
- Totaux   : pts, reb, oreb, dreb, ast, stl, blk, tov, pf, fp
- Tir      : fg_pct, three_pa, three_pct, ftm, fta, ft_pct
- Avancées : offrtg, defrtg, netrtg, ts_pct, efg_pct,
             usg_pct, pie, pace, plus_minus, ast_to

5. Exemples
Question: Combien de points marque Nikola Jokic ?
SQL:
SELECT full_name, pts, pts_avg, gp
FROM v_player_stats
WHERE full_name_normalized = 'nikola jokic';

Question: Combien de rebonds pour Jokic ?
SQL:
SELECT full_name, reb, reb_avg, oreb_avg, dreb_avg, gp
FROM v_player_stats
WHERE full_name_normalized = 'nikola jokic';

Question: Quel joueur a le plus de points ?
SQL:
SELECT full_name, team_code, pts, pts_avg
FROM v_top_scorers LIMIT 1;

Question: Quel joueur a le meilleur pourcentage à 3 points ?
SQL:
SELECT full_name, team_code, three_pa, three_pct
FROM v_top_three_point LIMIT 1;

Question: Giannis points
SQL:
SELECT full_name, pts, pts_avg, gp
FROM v_player_stats
WHERE full_name_normalized LIKE '%giannis%';

6. Réponse finale
- Toujours reformuler le résultat SQL en français
- Inclure les chiffres clés (moyenne ET total si pertinent)
- Ne jamais inventer de données
- Si aucune donnée n'est trouvée → dire clairement que l'information n'est pas disponible
"""

# ─────────────────────────────────────────────
# Détection du type de question
# ─────────────────────────────────────────────

STAT_KEYWORDS = [
    # Mots complets
    "point", "rebond", "passe", "assist", "pourcentage", "%", "tir",
    "moyenne", "classement", "meilleur", "top", "compare", "stats",
    "statistique", "score", "marque", "rating", "netrtg", "offrtg",
    "pie", "triple", "double", "interception", "contre", "blk", "stl",
    "combien", "quel joueur", "quelle équipe", "saison", "match",
    "efficacit", "ratio", "impact", "win", "loss", "victoire", "défaite",
    # Abréviations statistiques NBA
    "pts", "reb", "ast", "fg", "ft", "3p", "tov", "cmb",
    "avg", "pct", "rtg", "usg", "ts", "gp", "min",
]

def is_statistical_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in STAT_KEYWORDS)


# ─────────────────────────────────────────────
# Singletons — lru_cache remplace st.cache_resource
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_vector_store() -> VectorStoreManager:
    vs = VectorStoreManager()
    if vs.index is None:
        logger.warning("Index FAISS non trouvé — RAG désactivé")
    else:
        logger.info(f"Index FAISS chargé : {vs.index.ntotal} vecteurs")
    return vs


@lru_cache(maxsize=1)
def load_agent() -> AgentExecutor:
    """Crée l'AgentExecutor LangChain avec SQL Tool."""
    llm   = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, model=MODEL_NAME, temperature=0.0)
    tools = [get_sql_tool(db_path=DB_PATH)]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_AGENT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    logger.info("AgentExecutor LangChain initialisé")
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)


# ─────────────────────────────────────────────
# Moteur hybride SQL + RAG
# ─────────────────────────────────────────────

def _get_sql_context(question: str) -> str:
    """Interroge le SQL Tool — retourne les données brutes."""
    try:
        agent_executor = load_agent()
        result = agent_executor.invoke({"input": question})
        sql_context = result["output"]
        # FIX CRITIQUE : forcer string
        if not isinstance(sql_context, str):
            logger.warning(f"SQL output non-string détecté: {type(sql_context)}")
            sql_context = str(sql_context)
        logger.info(f"  SQL ✓ : {sql_context[:80]}")
        return sql_context
    except Exception as e:
        logger.error(f"  SQL erreur : {e}")
        return ""


def _get_rag_context(question: str) -> tuple[str, list[str]]:
    """
    Interroge FAISS — retourne (texte formaté, liste de chunks).
    Le texte formaté est pour Mistral, la liste pour RAGAS.
    """
    vector_store = load_vector_store()
    if vector_store.index is None:
        return "", []
    try:
        results = vector_store.search(question, k=SEARCH_K)
        if not results:
            return "", []
        chunks = [r["text"] for r in results]
        context_str = "\n\n".join([
            f"Source: {r['metadata'].get('source', 'Inconnue')}\n{r['text']}"
            for r in results
        ])
        logger.info(f"  RAG ✓ : {len(results)} chunks")
        return context_str, chunks
    except Exception as e:
        logger.error(f"  RAG erreur : {e}")
        return "", []


def repondre_avec_agent(question: str) -> str:
    """
    Architecture hybride :
    - Questions statistiques → SQL + RAG → synthèse Mistral
    - Questions narratives   → RAG seul  → synthèse Mistral
    """
    client = Mistral(api_key=MISTRAL_API_KEY)

    if is_statistical_question(question):
        # ── Hybride SQL + RAG ────────────────────────────
        logger.info(f"→ Route HYBRIDE : {question[:60]}")

        sql_context          = _get_sql_context(question)
        rag_context_str, _   = _get_rag_context(question)

        if not sql_context and not rag_context_str:
            return "Désolé, aucune information disponible pour cette question." , "HYBRIDE"

        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": SYSTEM_PROMPT_HYBRIDE.format(
                    sql_context=sql_context or "Aucune donnée statistique disponible.",
                    rag_context=rag_context_str or "Aucun contexte documentaire disponible.",
                    question=question,
                )}],
                temperature=0.1,
            )
            return response.choices[0].message.content , "HYBRIDE"
        except Exception as e:
            logger.error(f"Erreur synthèse hybride : {e}")
            return sql_context or "Erreur lors de la synthèse."

    else:
        # ── RAG seul ─────────────────────────────────────
        logger.info(f"→ Route RAG : {question[:60]}")
        rag_context_str, _ = _get_rag_context(question)

        if not rag_context_str:
            return "L'index documentaire n'est pas disponible. Lance d'abord : python indexer.py"

        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": SYSTEM_PROMPT_RAG.format(
                    context_str=rag_context_str,
                    question=question,
                )}],
                temperature=0.1,
            )
            return response.choices[0].message.content , "RAG"
        except Exception as e:
            logger.error(f"Erreur RAG : {e}")
            return f"Désolé, une erreur est survenue : {e}" , "RAG"


def repondre_avec_contextes(question: str) -> tuple[str, list[str]]:
    """
    Variante pour evaluate_ragas.py — retourne (réponse, contextes).
    Les contextes contiennent SQL + chunks RAG pour RAGAS.
    """
    client = Mistral(api_key=MISTRAL_API_KEY)

    if is_statistical_question(question):
        logger.info(f"→ Route HYBRIDE (RAGAS) : {question[:60]}")

        sql_context              = _get_sql_context(question)
        rag_context_str, chunks  = _get_rag_context(question)

        # Contextes pour RAGAS = SQL + chunks RAG
        contexts = []
        if sql_context:
            contexts.append(f"[SQL PRIORITAIRE]\n{str(sql_context)}")

        for chunk in chunks:
            contexts.append(f"[RAG]\n{str(chunk)}")
        

        if not sql_context and not rag_context_str:
            return "Aucune information disponible.", []

        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": SYSTEM_PROMPT_HYBRIDE.format(
                    sql_context=sql_context or "Aucune donnée statistique disponible.",
                    rag_context=rag_context_str or "Aucun contexte documentaire disponible.",
                    question=question,
                )}],
                temperature=0.1,
            )
            return response.choices[0].message.content, contexts
        except Exception as e:
            logger.error(f"Erreur synthèse hybride : {e}")
            return sql_context or "Erreur.", contexts

    else:
        logger.info(f"→ Route RAG (RAGAS) : {question[:60]}")
        rag_context_str, chunks = _get_rag_context(question)

        if not rag_context_str:
            return "Index non disponible.", []

        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": SYSTEM_PROMPT_RAG.format(
                    context_str=rag_context_str,
                    question=question,
                )}],
                temperature=0.1,
            )
            return response.choices[0].message.content, chunks
        except Exception as e:
            logger.error(f"Erreur RAG : {e}")
            return f"Erreur : {e}", []