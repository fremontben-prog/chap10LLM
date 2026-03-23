"""
nba_engine.py
-------------
Moteur métier NBA — logique pure sans Streamlit.
Importable par MistralChat_updated.py ET evaluate_ragas.py.

Contient :
    - is_statistical_question()  → routage par mots-clés
    - load_vector_store()        → index FAISS (singleton)
    - load_agent()               → AgentExecutor LangChain (singleton)
    - repondre_avec_agent()      → réponse finale (SQL ou RAG)
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

4. Exemples
Question: Combien de points marque Nikola Jokic ?
SQL:
SELECT pts FROM season_stats
JOIN players ON season_stats.player_id = players.player_id
WHERE full_name_normalized = 'nikola jokic';

Question: Quel joueur a le plus de points ?
SQL:
SELECT full_name, pts FROM season_stats
JOIN players ON season_stats.player_id = players.player_id
ORDER BY pts DESC LIMIT 1;

Question: Quel joueur a le meilleur pourcentage à 3 points ?
SQL:
SELECT full_name, three_pct FROM season_stats
JOIN players ON season_stats.player_id = players.player_id
ORDER BY three_pct DESC LIMIT 1;

Question: Giannis points
SQL:
SELECT pts FROM season_stats
JOIN players ON season_stats.player_id = players.player_id
WHERE full_name_normalized LIKE '%giannis%';

5. Réponse finale
- Toujours reformuler le résultat SQL en français
- Inclure les chiffres clés
- Ne jamais inventer de données
- Ne jamais utiliser des données Basketball-Reference, Reddit ou d'autres sources externes
- Si aucune donnée n'est trouvée → dire clairement que l'information n'est pas disponible
"""

# ─────────────────────────────────────────────
# Détection du type de question
# ─────────────────────────────────────────────

STAT_KEYWORDS = [
    "point", "rebond", "passe", "assist", "pourcentage", "%", "tir",
    "moyenne", "classement", "meilleur", "top", "compare", "stats",
    "statistique", "score", "marque", "rating", "netrtg", "offrtg",
    "pie", "triple", "double", "interception", "contre", "blk", "stl",
    "combien", "quel joueur", "quelle équipe", "saison", "match",
    "efficacit", "ratio", "impact", "win", "loss", "victoire", "défaite",
    "pts", "reb", "ast", "fg%", "3p%", "ft%",
    "blk", "tov", "min", "gp", "cmb",
    "avg", "pct", "rtg", "usg", "ts%",
    "offrtg", "defrtg", "netrtg", "pie",
]

def is_statistical_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in STAT_KEYWORDS)

# ─────────────────────────────────────────────
# Détection des nicknames des joueurs
# ─────────────────────────────────────────────

NBA_NICKNAMES = {
    "sga":      "Shai Gilgeous-Alexander",
    "lebron":   "LeBron James",
    "ad":       "Anthony Davis",
    "pg":       "Paul George",
    "kd":       "Kevin Durant",
    "cp3":      "Chris Paul",
    "luka":     "Luka Doncic",
    "giannis":  "Giannis Antetokounmpo",
    "jokic":    "Nikola Jokic",
    "embiid":   "Joel Embiid",
    "steph":    "Stephen Curry",
    "tatum":    "Jayson Tatum",
    "ja":       "Ja Morant",
}

def resolve_nicknames(question: str) -> str:
    """Remplace les surnoms par les vrais noms avant envoi au LLM."""
    q = question.lower()
    for nickname, full_name in NBA_NICKNAMES.items():
        if nickname in q.split():  # mot entier uniquement
            question = question.lower().replace(nickname, full_name)
    return question

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
    llm   = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, model=MODEL_NAME, temperature=0.1)
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
# Fonction principale — même logique que l'app
# ─────────────────────────────────────────────

def repondre_avec_agent(question: str) -> str:
    """
    Route vers SQL Tool ou RAG selon la nature de la question.
    Utilisable par Streamlit ET evaluate_ragas.py.
    """
    agent_executor = load_agent()
    vector_store   = load_vector_store()
    
    # Résoudre les surnoms avant routage
    question = resolve_nicknames(question)
    logger.info(f"Question normalisée : {question}")

    if is_statistical_question(question):
        # ── Voie 1 : Agent SQL ───────────────────────────────
        logger.info(f"→ Route SQL : {question[:60]}")
        try:
            result = agent_executor.invoke({"input": question})
            return result["output"]
        except Exception as e:
            logger.error(f"Erreur agent SQL : {e}")
            return f"Désolé, je n'ai pas pu récupérer les statistiques : {e}"

    else:
        # ── Voie 2 : RAG FAISS ───────────────────────────────
        logger.info(f"→ Route RAG : {question[:60]}")
        if vector_store.index is None:
            return "L'index documentaire n'est pas disponible. Lance d'abord : python indexer.py"

        try:
            results     = vector_store.search(question, k=SEARCH_K)
            context_str = "\n\n---\n\n".join([
                f"Source: {r['metadata'].get('source', 'Inconnue')} "
                f"(Score: {r['score']:.1f}%)\nContenu: {r['text']}"
                for r in results
            ]) if results else "Aucune information pertinente trouvée."

            client   = Mistral(api_key=MISTRAL_API_KEY)
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": SYSTEM_PROMPT_RAG.format(
                    context_str=context_str,
                    question=question,
                )}],
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Erreur RAG : {e}")
            return f"Désolé, une erreur est survenue : {e}"
