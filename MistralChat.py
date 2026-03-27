"""
MistralChat.py
----------------------
Application Streamlit NBA Analyst AI.
La logique métier est dans nba_engine.py — ce fichier gère uniquement l'UI.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st

from nba_engine import (
    repondre_avec_agent,
    is_statistical_question,
    load_vector_store,
    load_agent,
)


# ─────────────────────────────────────────────
# Initialisation cached Streamlit
# ─────────────────────────────────────────────

@st.cache_resource
def init_vector_store():
    return load_vector_store()

@st.cache_resource
def init_agent():
    return load_agent()


# ─────────────────────────────────────────────
# Interface Streamlit
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="NBA Analyst AI",
        page_icon="🏀",
        layout="wide",
    )

    st.title("🏀 NBA Analyst AI")
    st.caption("Powered by Mistral + FAISS RAG + SQL Tool | Saison 2024-2025")

    # Pré-charger les ressources
    init_vector_store()
    init_agent()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Routage automatique")
        st.markdown("""
        | Type de question | Route |
        |---|---|
        | Stats, chiffres, classements | 🗄️ SQL Tool |
        | Narratif, tactique, histoire | 📚 RAG FAISS |
        """)
        st.divider()
        st.markdown("**Exemples SQL :**")
        for ex in [
            "Qui sont les 5 meilleurs scoreurs ?",
            "Quel est le % à 3 points de Curry ?",
            "Compare LeBron et Jokic",
            "Quelle équipe a le meilleur Net Rating ?",
            "Qui a le plus de triple-doubles ?",
        ]:
            if st.button(ex, key=f"sql_{ex}"):
                st.session_state["input_prefill"] = ex

        st.divider()
        st.markdown("**Exemples RAG :**")
        for ex in [
            "Explique la zone défensive",
            "Qu'est-ce que le pick-and-roll ?",
            "Histoire des Warriors",
        ]:
            if st.button(ex, key=f"rag_{ex}"):
                st.session_state["input_prefill"] = ex

    # Historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    prefill  = st.session_state.pop("input_prefill", "") if "input_prefill" in st.session_state else ""
    question = st.chat_input("Pose ta question NBA...", key="chat_input")

    if question or prefill:
        q = question or prefill

        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        # route = "🗄️ SQL" if is_statistical_question(q) else "📚 RAG"
        with st.chat_message("assistant"):      
            with st.spinner("Analyse en cours..."):
                reponse, route = repondre_avec_agent(q)
            labels = {"HYBRIDE": "🗄️ SQL + 📚 RAG", "RAG": "📚 RAG"}
            st.markdown(reponse)
            st.caption(f"Route utilisée : {labels[route]}")

        st.session_state.messages.append({"role": "assistant", "content": reponse})


if __name__ == "__main__":
    main()
