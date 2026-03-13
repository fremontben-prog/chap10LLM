"""
load_excel_to_db.py
-------------------
Pipeline d'ingestion du fichier regular_NBA.xlsx vers SQLite.

Feuilles exploitées :
    - 'Données NBA'  → players + season_stats (569 joueurs, 44 colonnes utiles)
    - 'Equipe'       → teams (30 franchises NBA)

Note : la colonne Excel '15:00:00' (datetime.time bug) correspond à 3PM.

Usage :
    python load_excel_to_db.py
    python load_excel_to_db.py --excel data/regular_NBA.xlsx --season 2024-2025
"""

import os
import sys
import sqlite3
import logging
import argparse
from typing import Optional

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
DATABASE_DIR  = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "basketball.db")
SCHEMA_FILE   = "schema.sql"
DEFAULT_EXCEL  = "data/regular_NBA.xlsx"
DEFAULT_SEASON = "2024-2025"


# ── Schémas Pydantic ─────────────────────────────────────────

class TeamRow(BaseModel):
    team_code: str
    team_name: str

    @field_validator("team_code")
    @classmethod
    def upper_code(cls, v: str) -> str:
        return v.strip().upper()


class PlayerRow(BaseModel):
    full_name: str
    team_code: str
    age:       Optional[int] = None

    @field_validator("full_name", "team_code")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("Champ obligatoire vide")
        return v


class SeasonStatRow(BaseModel):
    player_name: str
    team_code:   str
    season:      str
    gp:     int   = Field(default=0, ge=0)
    wins:   int   = Field(default=0, ge=0)
    losses: int   = Field(default=0, ge=0)
    min_avg: float = Field(default=0.0, ge=0)
    pts:    int   = Field(default=0, ge=0)
    fgm:    int   = Field(default=0, ge=0)
    fga:    int   = Field(default=0, ge=0)
    fg_pct: float = Field(default=0.0, ge=0)
    three_pm:  int   = Field(default=0, ge=0)
    three_pa:  int   = Field(default=0, ge=0)
    three_pct: float = Field(default=0.0, ge=0)
    ftm:    int   = Field(default=0, ge=0)
    fta:    int   = Field(default=0, ge=0)
    ft_pct: float = Field(default=0.0, ge=0)
    oreb:   int   = Field(default=0, ge=0)
    dreb:   int   = Field(default=0, ge=0)
    reb:    int   = Field(default=0, ge=0)
    ast:    int   = Field(default=0, ge=0)
    tov:    int   = Field(default=0, ge=0)
    stl:    int   = Field(default=0, ge=0)
    blk:    int   = Field(default=0, ge=0)
    pf:     int   = Field(default=0, ge=0)
    fp:     float = Field(default=0.0)
    dd2:    int   = Field(default=0, ge=0)
    td3:    int   = Field(default=0, ge=0)
    plus_minus: float = Field(default=0.0)
    offrtg:   float = Field(default=0.0)
    defrtg:   float = Field(default=0.0)
    netrtg:   float = Field(default=0.0)
    ast_pct:  float = Field(default=0.0)
    ast_to:   float = Field(default=0.0)
    ast_ratio: float = Field(default=0.0)
    oreb_pct: float = Field(default=0.0)
    dreb_pct: float = Field(default=0.0)
    reb_pct:  float = Field(default=0.0)
    to_ratio: float = Field(default=0.0)
    efg_pct:  float = Field(default=0.0)
    ts_pct:   float = Field(default=0.0)
    usg_pct:  float = Field(default=0.0)
    pace:     float = Field(default=0.0)
    pie:      float = Field(default=0.0)
    poss:     int   = Field(default=0, ge=0)

    @model_validator(mode="after")
    def gp_coherence(self) -> "SeasonStatRow":
        if self.wins + self.losses > self.gp + 1:  # +1 tolérance
            raise ValueError(f"W+L ({self.wins+self.losses}) > GP ({self.gp})")
        return self


# ── Helpers ──────────────────────────────────────────────────

def si(val, default: int = 0) -> int:
    """Safe int."""
    try:
        v = val.item() if hasattr(val, "item") else val
        return int(v) if pd.notna(v) and v == v else default
    except (ValueError, TypeError):
        return default

def sf(val, default: float = 0.0) -> float:
    """Safe float."""
    try:
        v = val.item() if hasattr(val, "item") else val
        return float(v) if pd.notna(v) and v == v else default
    except (ValueError, TypeError):
        return default

def ss(val, default: str = "") -> str:
    """Safe str."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return str(val).strip()


# ── Init DB ──────────────────────────────────────────────────

def init_db(db_path: str = DATABASE_FILE) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    if os.path.exists(SCHEMA_FILE):
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        logger.info(f"Schéma appliqué depuis {SCHEMA_FILE}")
    else:
        logger.error(f"Fichier schéma introuvable : {SCHEMA_FILE}")
        sys.exit(1)
    conn.commit()
    return conn


# ── Ingestion équipes ─────────────────────────────────────────

def ingest_teams(conn: sqlite3.Connection, xl: pd.ExcelFile) -> int:
    df = xl.parse("Equipe")
    inserted = 0
    for _, row in df.iterrows():
        try:
            t = TeamRow(
                team_code=ss(row.get("Code", "")),
                team_name=ss(row.get("Nom complet de l'équipe", ""))
            )
            conn.execute(
                "INSERT OR IGNORE INTO teams (team_code, team_name) VALUES (?,?)",
                (t.team_code, t.team_name)
            )
            inserted += 1
        except Exception as e:
            logger.warning(f"Équipe ignorée : {e}")
    conn.commit()
    logger.info(f"  {inserted} équipes insérées")
    return inserted


# ── Ingestion joueurs + stats ─────────────────────────────────

def ingest_players_and_stats(
    conn: sqlite3.Connection,
    xl: pd.ExcelFile,
    season: str,
    source: str,
) -> tuple[int, int]:
    """
    Lit la feuille 'Données NBA' et insère players + season_stats.
    Mapping exact des colonnes du fichier :
        col index 11 (datetime 15:00:00) → 3PM (three_pm)
    """
    df = xl.parse("Données NBA", header=1)

    # Renommer la colonne datetime buggée en '3PM'
    col_map = {}
    for col in df.columns:
        if hasattr(col, 'hour'):  # datetime.time object
            col_map[col] = "3PM"
    df = df.rename(columns=col_map)

    # Supprimer les colonnes Unnamed vides
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]

    logger.info(f"  Colonnes mappées : {list(df.columns)}")
    logger.info(f"  {len(df)} lignes à traiter")

    players_inserted = 0
    stats_inserted   = 0
    errors           = 0

    for idx, row in df.iterrows():
        player_name = ss(row.get("Player", ""))
        team_code   = ss(row.get("Team", "")).upper()

        if not player_name or not team_code:
            continue

        try:
            # ── Validation Pydantic ──────────────────────────
            stat = SeasonStatRow(
                player_name=player_name,
                team_code=team_code,
                season=season,
                gp=si(row.get("GP")),
                wins=si(row.get("W")),
                losses=si(row.get("L")),
                min_avg=sf(row.get("Min")),
                pts=si(row.get("PTS")),
                fgm=si(row.get("FGM")),
                fga=si(row.get("FGA")),
                fg_pct=sf(row.get("FG%")),
                three_pm=si(row.get("3PM")),    # colonne renommée
                three_pa=si(row.get("3PA")),
                three_pct=sf(row.get("3P%")),
                ftm=si(row.get("FTM")),
                fta=si(row.get("FTA")),
                ft_pct=sf(row.get("FT%")),
                oreb=si(row.get("OREB")),
                dreb=si(row.get("DREB")),
                reb=si(row.get("REB")),
                ast=si(row.get("AST")),
                tov=si(row.get("TOV")),
                stl=si(row.get("STL")),
                blk=si(row.get("BLK")),
                pf=si(row.get("PF")),
                fp=sf(row.get("FP")),
                dd2=si(row.get("DD2")),
                td3=si(row.get("TD3")),
                plus_minus=sf(row.get("+/-")),
                offrtg=sf(row.get("OFFRTG")),
                defrtg=sf(row.get("DEFRTG")),
                netrtg=sf(row.get("NETRTG")),
                ast_pct=sf(row.get("AST%")),
                ast_to=sf(row.get("AST/TO")),
                ast_ratio=sf(row.get("AST RATIO")),
                oreb_pct=sf(row.get("OREB%")),
                dreb_pct=sf(row.get("DREB%")),
                reb_pct=sf(row.get("REB%")),
                to_ratio=sf(row.get("TO RATIO")),
                efg_pct=sf(row.get("EFG%")),
                ts_pct=sf(row.get("TS%")),
                usg_pct=sf(row.get("USG%")),
                pace=sf(row.get("PACE")),
                pie=sf(row.get("PIE")),
                poss=si(row.get("POSS")),
            )

            # ── Équipe (si absente) ──────────────────────────
            conn.execute(
                "INSERT OR IGNORE INTO teams (team_code, team_name) VALUES (?,?)",
                (team_code, team_code)
            )

            # ── Joueur ───────────────────────────────────────
            age = si(row.get("Age"), None)
            conn.execute(
                "INSERT OR IGNORE INTO players (full_name, team_code, age) VALUES (?,?,?)",
                (player_name, team_code, age)
            )
            player_id = conn.execute(
                "SELECT player_id FROM players WHERE full_name=? AND team_code=?",
                (player_name, team_code)
            ).fetchone()[0]
            players_inserted += 1

            # ── Stats ────────────────────────────────────────
            conn.execute("""
                INSERT OR REPLACE INTO season_stats (
                    player_id, season,
                    gp, wins, losses, min_avg,
                    pts, fgm, fga, fg_pct,
                    three_pm, three_pa, three_pct,
                    ftm, fta, ft_pct,
                    oreb, dreb, reb,
                    ast, tov, stl, blk, pf,
                    fp, dd2, td3, plus_minus,
                    offrtg, defrtg, netrtg,
                    ast_pct, ast_to, ast_ratio,
                    oreb_pct, dreb_pct, reb_pct, to_ratio,
                    efg_pct, ts_pct, usg_pct,
                    pace, pie, poss
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
            """, (
                player_id, stat.season,
                stat.gp, stat.wins, stat.losses, stat.min_avg,
                stat.pts, stat.fgm, stat.fga, stat.fg_pct,
                stat.three_pm, stat.three_pa, stat.three_pct,
                stat.ftm, stat.fta, stat.ft_pct,
                stat.oreb, stat.dreb, stat.reb,
                stat.ast, stat.tov, stat.stl, stat.blk, stat.pf,
                stat.fp, stat.dd2, stat.td3, stat.plus_minus,
                stat.offrtg, stat.defrtg, stat.netrtg,
                stat.ast_pct, stat.ast_to, stat.ast_ratio,
                stat.oreb_pct, stat.dreb_pct, stat.reb_pct, stat.to_ratio,
                stat.efg_pct, stat.ts_pct, stat.usg_pct,
                stat.pace, stat.pie, stat.poss
            ))
            stats_inserted += 1

        except Exception as e:
            logger.warning(f"Ligne {idx} ({player_name}) ignorée : {e}")
            errors += 1

    conn.commit()
    logger.info(f"  {players_inserted} joueurs, {stats_inserted} stats insérés, {errors} erreurs")
    return players_inserted, stats_inserted


# ── Pipeline principal ────────────────────────────────────────

def run_ingestion(
    excel_path: str = DEFAULT_EXCEL,
    season:     str = DEFAULT_SEASON,
    db_path:    str = DATABASE_FILE,
) -> dict:
    logger.info(f"Ingestion : {excel_path} → {db_path} (saison {season})")

    if not os.path.exists(excel_path):
        logger.error(f"Fichier Excel introuvable : {excel_path}")
        return {}

    conn   = init_db(db_path)
    xl     = pd.ExcelFile(excel_path)
    source = os.path.basename(excel_path)

    # 1. Équipes
    logger.info("Ingestion des équipes...")
    n_teams = ingest_teams(conn, xl)

    # 2. Joueurs + stats
    logger.info("Ingestion des joueurs et statistiques...")
    n_players, n_stats = ingest_players_and_stats(conn, xl, season, source)

    # 3. Bilan
    report = {
        "teams":   conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0],
        "players": conn.execute("SELECT COUNT(*) FROM players").fetchone()[0],
        "stats":   conn.execute("SELECT COUNT(*) FROM season_stats").fetchone()[0],
        "errors":  0,
    }
    conn.close()

    logger.info(f"✅ Ingestion terminée : {report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion Excel NBA → SQLite")
    parser.add_argument("--excel",  default=DEFAULT_EXCEL,  help="Chemin fichier Excel")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Saison (ex: 2024-2025)")
    parser.add_argument("--db",     default=DATABASE_FILE,  help="Chemin base SQLite")
    args = parser.parse_args()

    report = run_ingestion(args.excel, args.season, args.db)
    print("\n===== RAPPORT D'INGESTION =====")
    for k, v in report.items():
        print(f"  {k:10s}: {v}")
