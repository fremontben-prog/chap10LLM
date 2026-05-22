# 🏀 NBA Analyst AI — RAG + SQL avec Mistral AI

Application **RAG (Retrieval-Augmented Generation)** spécialisée NBA combinant :

- Recherche vectorielle **FAISS**
- Génération **Mistral AI**
- Agent SQL **LangChain**
- Observabilité complète avec **Logfire**
- Pipeline d'ingestion multi-format
- Évaluation **RAGAS**

L'application permet de poser des questions sur les statistiques NBA, les comparaisons de joueurs, les analyses d'équipes et les documents NBA via une interface **Streamlit** interactive.

---

## 🚀 Fonctionnalités

### 🔍 RAG documentaire
- Indexation sémantique des documents NBA
- Recherche vectorielle avec FAISS
- Embeddings via Mistral `mistral-embed`

### 🧠 Génération LLM
- Génération de réponses avec Mistral AI
- Prompt engineering spécialisé NBA
- Routage SQL vs RAG

### 🗄️ Agent SQL NBA
- Requêtes dynamiques sur SQLite
- Tool LangChain SQL
- Questions statistiques NBA structurées

### 📊 Observabilité complète
- Traces détaillées avec **Logfire**
- Monitoring : retrieval, reranking, génération, latence, tokens

### 📈 Évaluation RAG
- Évaluation automatique avec **RAGAS**
- Benchmarks : questions simples, complexes, bruitées

### 📄 Pipeline documentaire
- Support : PDF, OCR, DOCX, TXT, CSV, Excel

---

## 🏗️ Architecture

```
project/
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── vector_db/
│   └── sqlite/
│
├── src/
│   └── mistralchat/
│       ├── ingestion/
│       ├── pipeline/
│       ├── prompting/
│       ├── sql/
│       ├── storage/
│       ├── observability/
│       └── config.py
│
├── tests/
│
├── environment.yml
├── pyproject.toml
└── README.md
```

---

## ⚙️ Stack technique

| Catégorie | Technologies |
|---|---|
| LLM / RAG | Mistral AI, LangChain, FAISS |
| Validation / Observabilité | Pydantic, Logfire |
| Data / SQL | SQLAlchemy, SQLite, Pandas |
| Interface | Streamlit |

---

## 📦 Installation

### 1. Cloner le projet

```bash
git clone <repo-url>
cd chap10LLM
```

### 2. Créer l'environnement Conda

```bash
conda env create -f environment.yml
conda activate chap10llm
```

### 3. Installation éditable du package

```bash
pip install -e .
```

### 4. Configuration des variables d'environnement

Créer un fichier `.env` :

```env
MISTRAL_API_KEY=your_api_key
LOGFIRE_TOKEN=your_logfire_token
```

### 5. Installation CUDA (optionnel)

**Windows + CUDA**
```powershell
.\setup_local.ps1
```

**Linux CPU**
```bash
pip install torch torchvision
```

---

## 📥 Ingestion des documents

Déposer les documents dans `data/raw/`. Formats supportés : PDF, DOCX, TXT, CSV, XLSX.

### Pipeline d'indexation

```bash
python -m mistralchat.ingestion.indexer
```

Le pipeline : charge les documents → valide via Pydantic → découpe en chunks → génère les embeddings Mistral → construit l'index FAISS → sauvegarde les chunks et embeddings.

### Injection SQL NBA

```bash
python -m mistralchat.sql.load_excel_to_db
```

---

## 🧪 Évaluation RAGAS

```bash
python -m mistralchat.eval.evaluate_ragas
```

Catégories évaluées : `SIMPLE`, `COMPLEX`, `NOISY`

| Type | Exemple |
|---|---|
| Simple | `"sga cmb de pts"` |
| Complexe | `"Qui a le meilleur Net Rating ?"` |
| Bruité | `"Compare LeBron et Jokic"` |

---

## ▶️ Lancer l'application

```bash
streamlit run app/streamlit_app.py
```

Application disponible sur : [http://localhost:8501](http://localhost:8501)

---

## 📊 Monitoring Logfire

Observabilité temps réel : retrieval, reranking, génération, scores, latence, tokens.

Exemple de trace :

```
retrieval      → top-k chunks récupérés
reranking      → scores de pertinence
génération     → réponse Mistral
latence        → temps de traitement
tokens         → consommation
```

Dashboard : [Pydantic Logfire](https://logfire.pydantic.dev)

---

## 🧠 Workflow

```
Question utilisateur
        ↓
Détection SQL vs RAG
        ↓
Recherche FAISS
        ↓
Reranking
        ↓
Contexte injecté
        ↓
Génération Mistral
        ↓
Réponse Streamlit
```

---

## 🧪 Exemples de questions

**Statistiques**
- `"Combien de points marque Shai Gilgeous-Alexander ?"`
- `"Qui a le meilleur Net Rating ?"`
- `"Quel joueur a le plus de triple-doubles ?"`

**Comparaisons**
- `"Compare LeBron James et Nikola Jokic"`

**Questions bruitées**
- `"sga cmb de pts"`
- `"c ki le meilleur scoreur"`

---

## 🔭 Roadmap

- [x] RAG documentaire
- [x] SQL Agent NBA
- [x] Monitoring Logfire
- [x] Évaluation RAGAS
- [ ] Reranking avancé
- [ ] Hybrid Search BM25 + FAISS
- [ ] Multi-agent routing
- [ ] Mémoire conversationnelle
- [ ] Déploiement cloud

---

## 📜 Licence

Projet éducatif / démonstration technique autour des architectures RAG modernes.
