-- ============================================================
-- schema.sql
-- Schéma relationnel SQLite adapté au fichier regular_NBA.xlsx
-- Source : statistiques de saison régulière NBA (570 joueurs)
-- ============================================================

-- ── Table teams ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS teams (
    team_code   TEXT PRIMARY KEY,
    team_name   TEXT NOT NULL
);

-- ── Table players ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT    NOT NULL,
    team_code   TEXT    NOT NULL REFERENCES teams(team_code),
    age         INTEGER,
    created_at  TEXT    DEFAULT (datetime('now')),
    UNIQUE(full_name, team_code)
);

-- ── Table season_stats ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS season_stats (
    stat_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER NOT NULL REFERENCES players(player_id),
    season      TEXT    NOT NULL DEFAULT '2024-2025',
    gp          INTEGER DEFAULT 0,
    wins        INTEGER DEFAULT 0,
    losses      INTEGER DEFAULT 0,
    min_avg     REAL    DEFAULT 0,
    pts         INTEGER DEFAULT 0,
    fgm         INTEGER DEFAULT 0,
    fga         INTEGER DEFAULT 0,
    fg_pct      REAL    DEFAULT 0,
    three_pm    INTEGER DEFAULT 0,
    three_pa    INTEGER DEFAULT 0,
    three_pct   REAL    DEFAULT 0,
    ftm         INTEGER DEFAULT 0,
    fta         INTEGER DEFAULT 0,
    ft_pct      REAL    DEFAULT 0,
    oreb        INTEGER DEFAULT 0,
    dreb        INTEGER DEFAULT 0,
    reb         INTEGER DEFAULT 0,
    ast         INTEGER DEFAULT 0,
    tov         INTEGER DEFAULT 0,
    stl         INTEGER DEFAULT 0,
    blk         INTEGER DEFAULT 0,
    pf          INTEGER DEFAULT 0,
    fp          REAL    DEFAULT 0,
    dd2         INTEGER DEFAULT 0,
    td3         INTEGER DEFAULT 0,
    plus_minus  REAL    DEFAULT 0,
    offrtg      REAL    DEFAULT 0,
    defrtg      REAL    DEFAULT 0,
    netrtg      REAL    DEFAULT 0,
    ast_pct     REAL    DEFAULT 0,
    ast_to      REAL    DEFAULT 0,
    ast_ratio   REAL    DEFAULT 0,
    oreb_pct    REAL    DEFAULT 0,
    dreb_pct    REAL    DEFAULT 0,
    reb_pct     REAL    DEFAULT 0,
    to_ratio    REAL    DEFAULT 0,
    efg_pct     REAL    DEFAULT 0,
    ts_pct      REAL    DEFAULT 0,
    usg_pct     REAL    DEFAULT 0,
    pace        REAL    DEFAULT 0,
    pie         REAL    DEFAULT 0,
    poss        INTEGER DEFAULT 0,
    created_at  TEXT    DEFAULT (datetime('now')),
    UNIQUE(player_id, season)
);

-- ── Table reports ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reports (
    report_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type TEXT    NOT NULL,
    player_id   INTEGER REFERENCES players(player_id),
    team_code   TEXT    REFERENCES teams(team_code),
    title       TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    season      TEXT,
    source_file TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- ── Index ────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_season_stats_player ON season_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_season_stats_season ON season_stats(season);
CREATE INDEX IF NOT EXISTS idx_players_team        ON players(team_code);
CREATE INDEX IF NOT EXISTS idx_players_name        ON players(full_name);

-- ============================================================
-- Vues utilitaires
-- ============================================================

CREATE VIEW IF NOT EXISTS v_player_stats AS
SELECT
    p.player_id, p.full_name, p.team_code, t.team_name, p.age, s.season,
    s.gp, s.wins, s.losses,
    ROUND(CAST(s.wins AS REAL) / NULLIF(s.gp, 0) * 100, 1) AS win_pct,
    s.min_avg,
    ROUND(CAST(s.pts AS REAL) / NULLIF(s.gp, 0), 1)  AS pts_avg,
    ROUND(CAST(s.reb AS REAL) / NULLIF(s.gp, 0), 1)  AS reb_avg,
    ROUND(CAST(s.ast AS REAL) / NULLIF(s.gp, 0), 1)  AS ast_avg,
    ROUND(CAST(s.stl AS REAL) / NULLIF(s.gp, 0), 1)  AS stl_avg,
    ROUND(CAST(s.blk AS REAL) / NULLIF(s.gp, 0), 1)  AS blk_avg,
    ROUND(CAST(s.tov AS REAL) / NULLIF(s.gp, 0), 1)  AS tov_avg,
    s.pts, s.reb, s.ast, s.stl, s.blk, s.tov,
    s.fgm, s.fga, s.fg_pct,
    s.three_pm, s.three_pa, s.three_pct,
    s.ftm, s.fta, s.ft_pct,
    s.oreb, s.dreb,
    s.dd2, s.td3, s.plus_minus,
    s.offrtg, s.defrtg, s.netrtg,
    s.ast_pct, s.ast_to, s.ast_ratio,
    s.oreb_pct, s.dreb_pct, s.reb_pct,
    s.efg_pct, s.ts_pct, s.usg_pct,
    s.pace, s.pie, s.poss, s.fp
FROM season_stats s
JOIN players p ON s.player_id = p.player_id
LEFT JOIN teams t ON p.team_code = t.team_code;

CREATE VIEW IF NOT EXISTS v_top_scorers AS
SELECT full_name, team_code, team_name, age, gp, pts_avg, reb_avg, ast_avg, fg_pct, three_pct, ts_pct, usg_pct
FROM v_player_stats ORDER BY pts_avg DESC;

CREATE VIEW IF NOT EXISTS v_top_three_point AS
SELECT full_name, team_code, gp, three_pm, three_pa, three_pct,
       ROUND(CAST(three_pm AS REAL) / NULLIF(gp, 0), 1) AS three_pm_per_game
FROM v_player_stats WHERE three_pa >= 100 ORDER BY three_pct DESC;

CREATE VIEW IF NOT EXISTS v_top_rebounders AS
SELECT full_name, team_code, team_name, gp, reb_avg, oreb_pct, dreb_pct, reb_pct
FROM v_player_stats ORDER BY reb_avg DESC;

CREATE VIEW IF NOT EXISTS v_top_assists AS
SELECT full_name, team_code, gp, ast_avg, ast_to, ast_pct, ast_ratio, tov_avg
FROM v_player_stats ORDER BY ast_avg DESC;

CREATE VIEW IF NOT EXISTS v_team_stats AS
SELECT
    p.team_code, t.team_name,
    COUNT(DISTINCT p.player_id) AS roster_size,
    ROUND(AVG(CAST(s.pts AS REAL) / NULLIF(s.gp, 0)), 1) AS avg_pts_per_player,
    ROUND(AVG(s.fg_pct), 1) AS avg_fg_pct,
    ROUND(AVG(s.three_pct), 1) AS avg_3pt_pct,
    ROUND(AVG(s.offrtg), 1) AS avg_offrtg,
    ROUND(AVG(s.defrtg), 1) AS avg_defrtg,
    ROUND(AVG(s.netrtg), 1) AS avg_netrtg,
    ROUND(AVG(s.pie), 2) AS avg_pie
FROM season_stats s
JOIN players p ON s.player_id = p.player_id
LEFT JOIN teams t ON p.team_code = t.team_code
GROUP BY p.team_code;

CREATE VIEW IF NOT EXISTS v_top_impact AS
SELECT full_name, team_code, gp, pts_avg, reb_avg, ast_avg, pie, netrtg, plus_minus, ts_pct, usg_pct
FROM v_player_stats WHERE gp >= 20 ORDER BY pie DESC;
