# Assistant RAG avec Mistral

Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) + SQL pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- 🤖 **Génération de réponses** avec les modèles Mistral (Small ou Large)
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.10 
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Créer un environnement virtuel**

N/A

3. **Installer les dépendances**

## Installation

1. `conda env create -f environment.yml`
2. `conda activate chap10llm`
3. **Windows + CUDA** : `.\setup_local.ps1`
   **Linux CPU**       : `pip install torch torchvision`

4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

## Structure du projet

```
.
├── MistralChat.py          # Application Streamlit principale
├── indexer.py              # Script pour indexer les documents
├── nba_engine.py           # Dossier pour préparer le prompt
├── inputs/                 # Dossier pour les documents sources
├── inputs_sql/             # Dossier pour les documents SQL (2ème itéraion)
├── monitoring/             # Dossier pour monitorer sur logfire
├── pipeline/               # Dossier pour le pipeline
├── vector_db/              # Dossier pour l'index FAISS et les chunks
├── database/               # Base de données SQLite pour les interactions
└── utils/                  # Modules utilitaires
    ├── config.py           # Configuration de l'application
    ├── database.py         # Gestion de la base de données
    └── vector_store.py     # Gestion de l'index vectoriel

```

## Utilisation

### 1. Ajouter des documents

Placez vos documents dans le dossier `inputs/`. Les formats supportés sont :
- PDF
- TXT
- DOCX
- CSV
- JSON

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
python indexer.py
```

Ce script va :
1. Charger les documents depuis le dossier `inputs/`
2. Découper les documents en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

### 3. Lancer l'application

```bash
streamlit run MistralChat.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.

### 4. Injecter les données SQL to db

```bash
python -m database.load_excel_to_db
```

### 5. Injecter des métriques RAGA dans logfire

```bash
python -m eval.evaluate_ragas

```

### 6. Consultation de l'activité sur logfire
Logfire sera accessible à l'adresse https://logfire-eu.pydantic.dev/fremontben-prog/chap10llm/ dans votre navigateur.

## Modules principaux

### `utils/vector_store.py`

Gère l'index vectoriel FAISS et la recherche sémantique :
- Chargement et découpage des documents
- Génération des embeddings avec Mistral
- Création et interrogation de l'index FAISS

### `nba_engine.py`

Construction des prompts pour Mistral
- Analyse des mots-clés
- Détection des questions statistiques vs générales

### `utils/database.py`

Gère la base de données SQLite pour les interactions :
- Enregistrement des questions et réponses
- Stockage des feedbacks utilisateurs
- Récupération des statistiques

### `sql_tool.py`
Tool LangChain pour requêtes SQL dynamiques sur la base basketball SQLite.

### `schema.sql`
Schéma relationnel SQLite adapté au fichier regular_NBA.xlsx


## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- clès API

