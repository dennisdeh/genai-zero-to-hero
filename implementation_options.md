# Implementation Options – Questionnaire

This file records the questionnaire used to decide how the missing parts of the repository were implemented,
together with the chosen answers. It is not part of the book (no `book_` prefix), but documents the design
decisions so they can be revisited later. Recommended options are marked with *(rec.)*;
the selected answer is marked with **[x]**.

To change a decision: tick a different option, and update the affected chapters accordingly.

## 1. Scope
Which parts should be completed?
- **[x]** Everything: all empty modules across p00–p07 *(rec.)*
- [ ] Data + classic ML only (p02, p03, p04, p05)
- [ ] Neural networks + containerisation only (p06, p01, p00 FastAPI)
- [ ] LLM gaps only (p07)

## 2. Data sources
- **[x]** Offline-first: built-in or generated datasets (scikit-learn, torchvision) so scripts run without accounts
  or API keys; the Kaggle/Hugging Face/OpenML chapters still download, with clear notes *(rec.)*
- [ ] Real public datasets everywhere (requires internet and a Kaggle token)

## 3. LLM backend (p07)
- **[x]** Local Ollama through the existing `get_llm()` helper, OpenAI selectable via `LLM_TO_USE` *(rec.)*
- [ ] OpenAI only
- [ ] Hugging Face `transformers`, in-process

## 4. Deep learning stack (p06)
- **[x]** PyTorch, plus PyTorch Geometric for GNNs *(rec.)*
- [ ] Pure PyTorch (hand-written message passing)
- [ ] Keras 3

## 5. Format of examples
- **[x]** `main_*.py` scripts following the repository convention, helpers in `utils/` *(rec.)*
- [ ] Jupyter notebooks
- [ ] Both

## 6. Documentation per chapter (multiple choice)
- **[x]** `book_*.md` sections (included in the PDF build) *(rec.)*
- **[x]** `README.md` per chapter with environment and run instructions *(rec.)*
- **[x]** `conda_env_gzh_*.yml` per part
- **[x]** `sources.md` per chapter

## 7. Visualisation and applications (p04)
- **[x]** Plotly charts + a Streamlit dashboard *(rec.)*
- [ ] Plotly Dash
- [ ] Matplotlib/Seaborn only

## 8. MLOps (p05) and validation (p07)
- **[x]** MLflow: tracking, model registry, LLM evaluation, local file store *(rec.)*
- [ ] Lightweight: joblib + hand-written metrics and pytest checks

## 9. Reinforcement learning for LLMs (p07)
- **[x]** GRPO with TRL + Unsloth, rule-based reward (needs a GPU) *(rec.)*
- [ ] Tiny REINFORCE from scratch (CPU)
- [ ] Both

## 10. Scheduling (p02)
- **[x]** APScheduler + notes on cron *(rec.)*
- [ ] Prefect
- [ ] Airflow (docker)

## 11. Containerisation and deployment (p01, multiple choice)
- **[x]** Dockerfile + docker compose for the p00 FastAPI model service *(rec.)*
- **[x]** Kubernetes manifests (Deployment, Service, HPA) for kind/minikube *(rec.)*
- **[x]** Book notes on deploying to a managed cloud service

## 12. LLM workflows (p07)
- **[x]** LangGraph implementations of the classic workflow patterns: prompt chaining, routing,
  parallelisation, orchestrator-workers, evaluator-optimiser *(rec.)*
- [ ] Plain Python on top of `get_llm()`
