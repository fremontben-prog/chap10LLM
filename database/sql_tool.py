"""
sql_tool.py
-----------
Tool LangChain pour requêtes SQL dynamiques sur la base basketball SQLite.
Adapté à la structure réelle du fichier regular_NBA.xlsx (saison régulière).

Tables : teams, players, season_stats
Vues   : v_player_stats, v_top_scorers, v_top_three_point,
         v_top_rebounders, v_top_assists, v_team_stats, v_top_impact
"""

import os
import re
import sqlite3
import logging
from typing import Optional

from pydantic import BaseModel, Field
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATABASE_FILE   = os.getenv("DATABASE_FILE", "database/basketball.db")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


# ── Schéma Pydantic résultat ─────────────────────────────────

class SQLResult(BaseModel):
    query:     str
    rows:      list
    columns:   list[str]
    row_count: int = Field(ge=0)
    error:     Optional[str] = None
    success:   bool = True


# ── Few-shot examples basés sur les vraies colonnes ──────────

FEW_SHOT_EXAMPLES = """
-- Q: Qui sont les 5 meilleurs scoreurs cette saison ?
SELECT full_name, team_code, gp, pts_avg, reb_avg, ast_avg, ts_pct
FROM v_top_scorers LIMIT 5;

-- Q: Quel joueur a le meilleur % à 3 points (min 100 tirs) ?
SELECT full_name, team_code, three_pa, three_pct, three_pm_per_game
FROM v_top_three_point LIMIT 10;

-- Q: Quel est le % à 3 points de Stephen Curry ?
SELECT full_name, team_code, gp, three_pm, three_pa, three_pct,
       ROUND(CAST(three_pm AS REAL)/gp, 1) AS three_pm_per_game
FROM v_player_stats
WHERE full_name LIKE '%Curry%';

-- Q: Compare LeBron James et Nikola Jokic
SELECT full_name, team_code, gp, pts_avg, reb_avg, ast_avg,
       fg_pct, three_pct, ts_pct, pie, plus_minus
FROM v_player_stats
WHERE full_name LIKE '%LeBron%' OR full_name LIKE '%Joki%';

-- Q: Quels sont les meilleurs rebondeurs ?
SELECT full_name, team_code, gp, reb_avg, oreb_pct, dreb_pct
FROM v_top_rebounders LIMIT 10;

-- Q: Quels sont les meilleurs passeurs ?
SELECT full_name, team_code, gp, ast_avg, ast_to, tov_avg
FROM v_top_assists LIMIT 10;

-- Q: Quelle équipe a le meilleur Net Rating ?
SELECT team_code, team_name, avg_offrtg, avg_defrtg, avg_netrtg
FROM v_team_stats ORDER BY avg_netrtg DESC LIMIT 10;

-- Q: Classement des joueurs par impact global (PIE) ?
SELECT full_name, team_code, gp, pts_avg, reb_avg, ast_avg, pie, netrtg
FROM v_top_impact LIMIT 10;

-- Q: Qui a le plus de triple-doubles cette saison ?
SELECT full_name, team_code, gp, td3, dd2, pts_avg, reb_avg, ast_avg
FROM v_player_stats
WHERE td3 > 0 ORDER BY td3 DESC LIMIT 10;

-- Q: Quel est le ratio passes/pertes (AST/TO) des meilleurs meneurs ?
SELECT full_name, team_code, gp, ast_avg, tov_avg, ast_to, ast_pct
FROM v_player_stats
WHERE ast_avg >= 5
ORDER BY ast_to DESC LIMIT 10;

-- Q: Quels joueurs ont le meilleur True Shooting % (min 40 matchs) ?
SELECT full_name, team_code, gp, pts_avg, ts_pct, efg_pct, usg_pct
FROM v_player_stats
WHERE gp >= 40
ORDER BY ts_pct DESC LIMIT 10;

-- Q: Stats de l'équipe des Lakers ?
SELECT full_name, gp, pts_avg, reb_avg, ast_avg, fg_pct, three_pct, ts_pct, plus_minus
FROM v_player_stats
WHERE team_code = 'LAL'
ORDER BY pts_avg DESC;

-- Q: Qui a le plus d'interceptions par match ?
SELECT full_name, team_code, gp,
       ROUND(CAST(stl AS REAL)/gp, 2) AS stl_avg,
       ROUND(CAST(blk AS REAL)/gp, 2) AS blk_avg,
       pts_avg
FROM v_player_stats
WHERE gp >= 30
ORDER BY stl_avg DESC LIMIT 10;

-- Q: Quels joueurs ont le meilleur Offensive Rating ?
SELECT full_name, team_code, gp, offrtg, defrtg, netrtg, pts_avg, usg_pct
FROM v_player_stats
WHERE gp >= 20
ORDER BY offrtg DESC LIMIT 10;

-- Q: Comparaison domicile/extérieur de Giannis ?
-- (les stats sont de saison, pas par match domicile/extérieur dans ce fichier)
SELECT full_name, team_code, gp, wins, losses,
       ROUND(CAST(wins AS REAL)/NULLIF(gp,0)*100,1) AS win_pct,
       pts_avg, reb_avg, ast_avg
FROM v_player_stats
WHERE full_name LIKE '%Giannis%' OR full_name LIKE '%Antetokounmpo%';
"""


# ── Prompt SQL ────────────────────────────────────────────────

SQL_PROMPT = PromptTemplate(
    input_variables=["question", "schema", "examples", "season"],
    template="""Tu es un expert SQL spécialisé en statistiques basketball NBA.
Génère UNE SEULE requête SQLite valide pour répondre à la question.

SCHÉMA DE LA BASE (saison régulière NBA) :
{schema}

SAISON : {season}

EXEMPLES :
{examples}

RÈGLES :
1. Génère UNIQUEMENT la requête SQL brute, sans markdown ni explication
2. Utilise LIKE pour les noms : WHERE full_name LIKE '%Curry%'
3. Limite avec LIMIT (max 20 lignes)
4. Préfère les vues (v_player_stats, v_top_scorers, etc.) aux jointures manuelles
5. Les stats PTS, REB, AST sont des TOTAUX de saison → divise par gp pour la moyenne
   (ou utilise pts_avg, reb_avg, ast_avg de v_player_stats qui sont déjà calculées)
6. Pour les équipes : utilise le code 3 lettres (LAL, BOS, OKC, etc.)

QUESTION : {question}

REQUÊTE SQL :"""
)


# ── Schéma pour le prompt ────────────────────────────────────

DB_SCHEMA = """
TABLE teams        : team_code (PK), team_name
TABLE players      : player_id (PK), full_name, team_code, age
TABLE season_stats : stat_id, player_id, season,
    gp (matchs joués), wins, losses, min_avg (min/match),
    pts (total), fgm, fga, fg_pct (%),
    three_pm (3PM total), three_pa, three_pct (%),
    ftm, fta, ft_pct (%),
    oreb, dreb, reb (totaux),
    ast, tov, stl, blk, pf (totaux),
    fp (fantasy pts), dd2, td3, plus_minus,
    offrtg, defrtg, netrtg,
    ast_pct, ast_to, ast_ratio,
    oreb_pct, dreb_pct, reb_pct, to_ratio,
    efg_pct, ts_pct, usg_pct, pace, pie, poss

VUE v_player_stats : toutes les colonnes ci-dessus +
    pts_avg, reb_avg, ast_avg, stl_avg, blk_avg, tov_avg (= total/gp),
    win_pct, team_name

VUE v_top_scorers      : triée par pts_avg DESC
VUE v_top_three_point  : triée par three_pct DESC (filtre three_pa >= 100)
VUE v_top_rebounders   : triée par reb_avg DESC
VUE v_top_assists      : triée par ast_avg DESC
VUE v_team_stats       : agrégations par équipe (avg_offrtg, avg_defrtg, avg_netrtg, avg_pie...)
VUE v_top_impact       : triée par pie DESC (filtre gp >= 20)

ÉQUIPES NBA : ATL, BKN, BOS, CHA, CHI, CLE, DAL, DEN, DET, GSW,
              HOU, IND, LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK,
              OKC, ORL, PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS
"""


# ── Exécution SQL ────────────────────────────────────────────

FORBIDDEN_KEYWORDS = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"}

def execute_sql(query: str, db_path: str = DATABASE_FILE) -> SQLResult:
    query_upper = query.upper().strip()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", query_upper):
            return SQLResult(query=query, rows=[], columns=[], row_count=0,
                             error=f"Mot-clé interdit : {kw}", success=False)

    if not os.path.exists(db_path):
        return SQLResult(query=query, rows=[], columns=[], row_count=0,
                         error=f"Base introuvable : {db_path}", success=False)
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur  = conn.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        return SQLResult(query=query, rows=[dict(r) for r in rows],
                         columns=cols, row_count=len(rows), success=True)
    except sqlite3.Error as e:
        logger.error(f"Erreur SQL : {e}\n{query}")
        return SQLResult(query=query, rows=[], columns=[], row_count=0,
                         error=str(e), success=False)


def format_results(result: SQLResult) -> str:
    if not result.success:
        return f"Erreur SQL : {result.error}"
    if result.row_count == 0:
        return "Aucun résultat trouvé."
    lines = [f"Résultats ({result.row_count} ligne(s)) :"]
    lines.append(" | ".join(result.columns))
    lines.append("-" * 70)
    for row in result.rows[:20]:
        lines.append(" | ".join(str(v) if v is not None else "—" for v in row.values()))
    return "\n".join(lines)


# ── Tool LangChain ───────────────────────────────────────────

class SQLBasketballTool:
    def __init__(self, db_path: str = DATABASE_FILE, season: str = "2024-2025"):
        self.db_path = db_path
        self.season  = season
        self.llm     = ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model="mistral-small",
            temperature=0.0,
        )

    def _generate_sql(self, question: str) -> str:
        prompt = SQL_PROMPT.format(
            question=question,
            schema=DB_SCHEMA,
            examples=FEW_SHOT_EXAMPLES,
            season=self.season,
        )
        response = self.llm.invoke(prompt)
        sql = response.content.strip()
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```\s*", "", sql)
        return sql.strip()

    def run(self, question: str) -> str:
        logger.info(f"SQL Tool → {question[:80]}...")
        try:
            sql    = self._generate_sql(question)
            logger.info(f"SQL généré : {sql[:120]}...")
            result = execute_sql(sql, self.db_path)
            return f"Requête SQL :\n{sql}\n\n{format_results(result)}"
        except Exception as e:
            logger.error(f"Erreur SQL Tool : {e}")
            return f"Erreur lors de l'exécution : {e}"


def get_sql_tool(db_path: str = DATABASE_FILE, season: str = "2024-2025") -> Tool:
    tool = SQLBasketballTool(db_path=db_path, season=season)
    return Tool(
        name="basketball_sql_query",
        func=tool.run,
        description=(
            "Utilise ce tool pour toute question chiffrée sur les statistiques NBA : "
            "points, rebonds, passes, pourcentages de tir, classements, comparaisons entre joueurs, "
            "stats par équipe, triple-doubles, impact global (PIE), ratings offensif/défensif. "
            "Input : question en langage naturel. Output : données statistiques précises."
        )
    )
