"""
Gère la base de données SQLite pour les interactions :
- Enregistrement des questions et réponses
- Stockage des feedbacks utilisateurs
- Récupération des statistiques
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, Float, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# --- Configuration ---
DATABASE_DIR = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

logger = logging.getLogger(__name__)
Base = declarative_base()


# ─────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────

class Interaction(Base):
    __tablename__ = "interactions"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    timestamp        = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_query       = Column(Text,    nullable=False)
    bot_response     = Column(Text,    nullable=False)
    sources          = Column(Text,    nullable=True)   # JSON : liste de sources
    feedback_score   = Column(Float,   nullable=True)   # ex. 1.0 (👍) / 0.0 (👎)
    feedback_text    = Column(String(500), nullable=True)
    feedback_at      = Column(DateTime,    nullable=True)

    def to_dict(self) -> Dict:
        return {
            "id":             self.id,
            "timestamp":      self.timestamp.isoformat() if self.timestamp else None,
            "user_query":     self.user_query,
            "bot_response":   self.bot_response,
            "sources":        json.loads(self.sources) if self.sources else [],
            "feedback_score": self.feedback_score,
            "feedback_text":  self.feedback_text,
            "feedback_at":    self.feedback_at.isoformat() if self.feedback_at else None,
        }


# ─────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────

def init_db() -> sessionmaker:
    """Crée le dossier, les tables si absentes, et retourne une SessionFactory."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    logger.info(f"Base de données initialisée : {DATABASE_FILE}")
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


# SessionFactory globale (initialisée une fois au démarrage)
SessionFactory = init_db()


def get_session() -> Session:
    """Retourne une nouvelle session SQLAlchemy."""
    return SessionFactory()


# ─────────────────────────────────────────────
# 1. Enregistrement des questions / réponses
# ─────────────────────────────────────────────

def save_interaction(
    user_query: str,
    bot_response: str,
    sources: Optional[List[str]] = None,
) -> int:
    """
    Enregistre une paire question/réponse.
    Retourne l'ID de l'interaction créée.
    """
    session = get_session()
    try:
        interaction = Interaction(
            user_query=user_query,
            bot_response=bot_response,
            sources=json.dumps(sources or []),
        )
        session.add(interaction)
        session.commit()
        session.refresh(interaction)
        logger.debug(f"Interaction #{interaction.id} enregistrée.")
        return interaction.id
    except Exception as e:
        session.rollback()
        logger.error(f"Erreur lors de l'enregistrement : {e}")
        raise
    finally:
        session.close()


def get_interaction(interaction_id: int) -> Optional[Dict]:
    """Retourne une interaction par son ID, ou None si introuvable."""
    session = get_session()
    try:
        interaction = session.get(Interaction, interaction_id)
        return interaction.to_dict() if interaction else None
    finally:
        session.close()


def get_recent_interactions(limit: int = 20) -> List[Dict]:
    """Retourne les `limit` interactions les plus récentes."""
    session = get_session()
    try:
        rows = (
            session.query(Interaction)
            .order_by(Interaction.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]
    finally:
        session.close()


# ─────────────────────────────────────────────
# 2. Stockage des feedbacks utilisateurs
# ─────────────────────────────────────────────

def save_feedback(
    interaction_id: int,
    score: float,                   # 1.0 = 👍  /  0.0 = 👎
    feedback_text: Optional[str] = None,
) -> bool:
    """
    Attache un feedback à une interaction existante.
    Retourne True si la mise à jour a réussi, False si l'interaction est introuvable.
    """
    session = get_session()
    try:
        interaction = session.get(Interaction, interaction_id)
        if not interaction:
            logger.warning(f"Interaction #{interaction_id} introuvable pour le feedback.")
            return False

        interaction.feedback_score = score
        interaction.feedback_text  = feedback_text
        interaction.feedback_at    = datetime.utcnow()
        session.commit()
        logger.debug(f"Feedback enregistré pour l'interaction #{interaction_id}.")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Erreur lors de l'enregistrement du feedback : {e}")
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────
# 3. Récupération des statistiques
# ─────────────────────────────────────────────

def get_stats() -> Dict:
    """
    Retourne un dictionnaire de statistiques globales :
    - total_interactions   : nombre total d'interactions
    - total_feedbacks      : nombre d'interactions avec feedback
    - avg_feedback_score   : score moyen (entre 0 et 1)
    - positive_feedbacks   : nombre de 👍 (score >= 0.5)
    - negative_feedbacks   : nombre de 👎 (score < 0.5)
    - satisfaction_rate    : taux de satisfaction en %
    - interactions_today   : interactions du jour
    """
    session = get_session()
    try:
        today = datetime.utcnow().date()

        total        = session.query(func.count(Interaction.id)).scalar() or 0
        with_feedback = (
            session.query(func.count(Interaction.id))
            .filter(Interaction.feedback_score.isnot(None))
            .scalar() or 0
        )
        avg_score    = (
            session.query(func.avg(Interaction.feedback_score))
            .filter(Interaction.feedback_score.isnot(None))
            .scalar()
        )
        positive     = (
            session.query(func.count(Interaction.id))
            .filter(Interaction.feedback_score >= 0.5)
            .scalar() or 0
        )
        negative     = (
            session.query(func.count(Interaction.id))
            .filter(Interaction.feedback_score < 0.5,
                    Interaction.feedback_score.isnot(None))
            .scalar() or 0
        )
        today_count  = (
            session.query(func.count(Interaction.id))
            .filter(func.date(Interaction.timestamp) == today)
            .scalar() or 0
        )

        satisfaction = round((positive / with_feedback * 100), 1) if with_feedback else 0.0

        return {
            "total_interactions":  total,
            "total_feedbacks":     with_feedback,
            "avg_feedback_score":  round(avg_score, 3) if avg_score is not None else None,
            "positive_feedbacks":  positive,
            "negative_feedbacks":  negative,
            "satisfaction_rate":   satisfaction,
            "interactions_today":  today_count,
        }
    finally:
        session.close()
