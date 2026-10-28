# Audit Report – Adversarial QA of the Completed Book Parts

**Scope:** everything added or changed when the empty parts were completed (commits `7112842..008d293`,
merged as dennisdeh/genai-zero-to-hero#1): about 300 files across p00–p08, the book build and the root docs.
**Date:** 2026-10-27. **Fixes:** this pull request (branch `claude/hopeful-allen-q06zpw`).

## Verdict

Yes, there were material mistakes. The audit found **75 issues: 3 critical, 23 major, 49 minor**.
All critical and major issues are fixed in this pull request. All minor issues are fixed too, except the
few listed under [Open items and owner decisions](#open-items-and-owner-decisions).

The three critical issues come from two root causes:

1. **Target leakage presented as a leakage-free result (p03, and the p04 app built on it).** The churn model
   used "days since last purchase", but in the simulated data churned customers *stop buying*, so the feature
   is a consequence of the label. The book's headline ROC AUC of 0.94 (in a section titled "leakage-free
   pipelines") was mostly leakage; the honest figure is **0.77**. The Streamlit dashboard showed the same
   0.94, called the coefficients "drivers" of churn, and let users "score" customers from recency.
2. **A reward bug in the GRPO fine-tuning example (p07 c03).** The format reward penalised the *whole*
   completion when it ended without a newline after `</answer>` (the usual case once the EOS token is stripped).
   A correct 2,400-character answer scored **1.55**, below a wrong 52-character answer (**1.95**), so
   training would have pushed the model towards short, wrong answers. The flaw was inherited from the Unsloth
   notebook the code was adapted from, and the tests only covered completions with a trailing newline.

Most major issues fall into four groups: wrong statements in the book (formulas, library behaviour,
cloud products), evaluation code that gave wrong numbers, documentation numbers that did not match the code,
and missing versions or dependencies in environment files.

## How the audit was done

- **Independent reviewers.** Eight reviewer agents, each without the authors' context, were told to assume the
  work contained mistakes and to find the material ones. Each covered one area: p00+p01, p02, p03, p04 plus
  repo-wide docs and the book build, p05, p06, the four new p07 chapters, and the p07 RL/older chapters plus
  p08. They could not edit the repository.
- **What they checked.** They ran every runnable example and test, compared every number quoted in the book
  and READMEs with the real output, re-derived the maths, and checked library calls against the source of
  the *installed* versions (TRL 0.24.0, Unsloth 2026.1.4, MLflow 3.17, scikit-learn 1.9, LightGBM 4.7,
  LangGraph 1.2, …). Code needing services that are not available in the sandbox was exercised with mocks:
  a mock Ollama server answering from the requested JSON schema, a mock OpenAI endpoint, and fake HTTP
  APIs. The p00/p01 reviewer started a local Docker daemon and built and ran the container image.
- **Verification by the coordinator.** Before any fix, the coordinator re-ran the evidence for the most
  serious claims (for example the leakage AUCs 0.942 → 0.770, the reward scores, and the metric outputs).
- **Fixes.** Separate fixer agents made the changes, one area each, and re-ran the affected scripts and tests.
  The coordinator then re-checked each set of fixes before committing it: tests, key numbers, and a full
  PDF build.

## Results by area

| Area | Critical | Major | Minor | Status |
|---|---|---|---|---|
| p00 Fundamentals + p01 Containerisation and deployment | 0 | 2 | 8 | all fixed |
| p02 Data sourcing | 0 | 2 | 6 | all fixed (one note added, see open items) |
| p03 Data processing | 1 | 2 | 6 | all fixed |
| p04 Visualisation + repo-wide docs and book build | 1 | 4 | 6 | all fixed (one owner decision) |
| p05 Classical ML | 0 | 4 | 5 | all fixed |
| p06 Neural networks | 0 | 2 | 8 | all fixed |
| p07 new chapters (c02, c05, c06, c07) | 0 | 4 | 4 | all fixed |
| p07 c00/c01/c03/c04 + p08 | 1 | 3 | 6 | all fixed (one owner decision) |
| **Total** | **3** | **23** | **49** | |

The reviewers rated the two leakage findings and the reward bug "major (borderline critical)". This report
counts them as critical, because they produce wrong results that the text presents as correct.

## Critical and major findings

Each entry gives what was wrong, why it matters, and what changed. IDs match the reviewers' reports.

### Critical

| ID | Where | What was wrong | Fix and verification |
|---|---|---|---|
| P03-1 | p03 `s01_preprocessing/utils/features.py`, `book_preprocessing.md` | `days_since_last_tx` leaks the churn label (churners stop buying in the simulation). Test AUC 0.942 with it, 0.770 without; recency/tenure alone ranks churners at AUC 0.97. | Removed from the model's features. The 0.94 result is kept as a clearly labelled "Leakage lesson 2" (post-outcome features), with the cut-off-date fix explained. Honest results: test AUC 0.770, CV 0.750. |
| P04-1 | p04 Streamlit app | Same leakage in the dashboard: "test ROC AUC 0.942", "leakage-free", "What drives churn?", and a recency slider whose prediction went from 0.02 to 0.98. | The app uses the fixed features (AUC 0.770). Recency slider removed; the chart now shows "associations", not "drivers". Unit tests check the model has no post-outcome feature. |
| P07R-1 | p07 c03 `s01_rl_unsloth/utils/rewards.py` (`count_xml`) | The trailing-text penalty used `split("\n</answer>\n")[-1]`, which returns the whole completion when there is no final newline. A correct long answer scored below a wrong short one. | Only text after the last `</answer>` is penalised. New tests: with and without a final newline, a perfect answer scores the same, and a correct long answer beats a wrong short one. 40 tests pass (10 new ones fail on the old code). |

### Major

| ID | Where | What was wrong | Fix |
|---|---|---|---|
| P01-1 | p01 c02 book | Recommended AWS App Runner, which has not accepted new customers since 30 April 2026 (per AWS's availability notice). | Recommends Amazon ECS Express Mode / ECS on Fargate and notes the App Runner closure. |
| P01-2 | p01 c02 README, workflow, book | The GitHub Actions → Google Cloud set-up omitted Workload Identity Federation (pool, provider, attribute condition, IAM roles), so the authentication step would fail. The "only this repository" claim depended on a condition that was never shown. | Complete set-up commands with the repository attribute condition and the minimal IAM bindings; the claim is tied to that condition. |
| P02-1 | p02 c00 data lake | Overwriting a day's partition with a batch that contained a few late records for that day deleted the rest of the day (2,000 rows → 5). The book presented this as an idempotency best practice. | Per-partition upsert by `transaction_id` with an atomic file swap; the book explains when a plain overwrite is valid. Late records now add to the day (2,000 → 2,005) with no duplicates. |
| P02-2 | p02 c00 storage benchmark | The table did not match the code (zstd 2.7 MB vs 3.03 MB actual), and timings were biased by one-time Parquet start-up cost. | Warm-up, best of 3 runs, the same row groups for both codecs; table regenerated from real output. |
| P03-2 | p03 text cleaning | The PII-masking regexes masked ISO dates, order numbers and ISBNs as phone numbers, and missed lower-case or hyphenated IBANs. | IBANs validated with the mod-97 checksum; a stricter phone pattern; caveat that regex masking is best-effort. |
| P03-3 | p03 text cleaning | The "meaning-preserving" cleaner applied NFKC and removed all format characters, which changes text: `10⁶ cells` → `106 cells`, `x²` → `x2`, and emoji and Persian words broken. | NFC by default; NFKC is an explicit lossy option; zero-width joiner and non-joiner kept. |
| P04-2 | p04 KPI row | "vs previous period" deltas were shown when the previous period was mostly outside the data (+887 %, +1,890 %), and boundary days were counted in both periods. | No delta when the previous period isn't covered; half-open date intervals; tests added. |
| X-1 | `build_book_pdf.sh` | 70 lines ran into the margin in the PDF (long paths in inline code, up to 320pt), and the code font lacked some characters. | Line breaks allowed in inline code (Lua filter), wrapped code blocks, DejaVu Sans Mono for code: 6 overflowing lines left, all small and in older text. |
| X-2 | book build, 10 references | Text says "part 3" meaning folder `p03`, but the PDF numbered the introduction 1 and Fundamentals 2, so every reference pointed two parts too far. | Introduction unnumbered and parts start at 0, so PDF part numbers equal folder numbers. |
| X-3 / P07R-4 | p07 c03 env | The README says to run the reward tests with pytest in `gzh_llm3`, which has no pytest. | pytest added to the env. |
| P05-1 | p05 c03 MLflow book | Presented skops as the safe alternative to pickle. In MLflow the trusted-types list is stored inside the model, so a tampered model can trust anything; loading an untrusted model can still run code. | Text corrected: load only from trusted registries; `MLFLOW_ALLOW_PICKLE_DESERIALIZATION=false` (verified that it blocks loading). |
| P05-2 | p05 env | No minimum versions, but the code needs LightGBM ≥ 4.7 (`eval_X`), scikit-learn ≥ 1.8 (`l1_ratio` without `penalty`) and MLflow ≥ 3.17 (skops by default). | Minimum versions pinned, and the reasons documented. |
| P05-3 | p05 c00 book | The Ridge formula used a 1/n factor that scikit-learn's `Ridge` doesn't, so the printed `RidgeCV` α (104.8) didn't match the book's α (about 0.3); the path plot mixed conventions. | scikit-learn's objective is shown, with α = n·α′ explained; plot axes are labelled per convention. |
| P05-4 | p05 c00 comparison | The "fair" model-family comparison used non-nested CV scores of tuned models. | Real nested CV; table regenerated (SVM 0.974, tree 0.925). |
| P06-1 | p06 c01 checkpoints | The checkpoint mixed best-epoch weights with last-epoch optimiser state and no scheduler state, while the book said training could be resumed exactly. | Model, optimiser and scheduler state are all saved from the best epoch; resuming is demonstrated (restarts at epoch 21 with a consistent learning rate). Test accuracy unchanged (0.992). |
| P06-2 | p06 c02 book | Claimed log(1 − D) is "flat" near D = 0; its derivative there is −1. The vanishing gradient appears in logit space (∂/∂l log(1 − σ(l)) = −σ(l) → 0). | Rewritten with the correct derivatives in logit space for both the minimax and the non-saturating loss. |
| P07-1 | p07 c02 retry | With OpenAI (documented as supported), a validation error was raised inside the call instead of being retried. | The retry loop also catches errors raised during the call; verified with a mock (3 attempts instead of a crash). |
| P07-2 | p07 c07 metrics | Number normalisation merged different numbers (`1.0%` = `10%`, `3.75%` = `37.5%`) and split equal ones (`10.0%` ≠ `10%`). | Numbers are protected and put into one standard form; tests added; ROUGE-L still matches the `rouge-score` package exactly. |
| P07-3 | p07 c07 injection test | It only checked that the answer didn't *start* with "APPROVED", so partial compliance with the injected instruction passed. | Fails on "APPROVED" anywhere or on the injected account number. |
| P07-4 | p07 c05 book and demo | Said Ollama's default context is "a few thousand tokens"; current Ollama picks it from VRAM (4k/32k/256k), so the truncation demo did nothing on larger GPUs. | Text corrected; the demo sets `num_ctx=4096` explicitly. |
| P07R-2 | p07 c01 README | The run steps pulled `qwen3:4b`, but the script requests `qwen3`, which then failed. | README and book corrected (pull `qwen3` or set the model). |
| P07R-3 | p07 c04 / p08 envs | `pypdf` and `docx2txt` were missing, so the documented RAG path failed on PDF/Word files. | Added to the env files. |
| P08-1 | p08 book and README | Described `s01/main_3` as a ReAct loop; the graph has no edge back from the tools, so the model never sees the tool result. | Follow-up: the owner confirmed it was meant to loop, so the missing edge `tools -> chatbot` was added and the text describes the loop (see open items). |

## Minor findings (all fixed unless noted)

- **p00/p01:** `replicas` set next to the HPA; port-forward load test only reaches one pod; liveness/readiness
  rationale; request-size limit checked after parsing; unvalidated `LOG_LEVEL` could crash-loop; outdated
  GitHub Actions versions, and the pushed image wasn't the tested one; `uv sync --locked`; unused env packages.
- **p02:** urllib3 back-off timings; GitHub rate limits on 429; APScheduler jitter; Frankfurter v1 deprecation (follow-up: migrated to v2);
  RFC 9309 vs stdlib robots.txt behaviour; zip-slip, cron/DST wording; `apscheduler<4`.
- **p03:** `strip_html` on plain text and double unescaping; `parse_numeric` sign flips and silent misparses;
  unseen categories crashed the pipeline; the BPE split pattern wasn't exactly cl100k's; special-token
  injection, surrogates and re-training state in the BPE tokeniser; counts and speed-ups in the text;
  `pandera>=0.24`.
- **p04 and repo-wide:** weak app tests; default period started with a partial month; filter wording;
  colour-blind palette claim; heatmap `zmid`, animation ranges and auto-play; pinned-vs-unpinned env wording
  and the note on tracked `.env` files.
- **p05:** drift-reference wording; tuning explanation; SHAP base value vs mean probability; PDP slopes;
  `SVC(probability=True)` deprecation; C vs α; validation-set reuse for early stopping and ranking.
- **p06:** universal-approximation attribution (Leshno et al. 1993 for non-polynomial activations); GCN "weighted
  average" → normalised weighted sum; label smoothing is 0.955, not 0.95; overstated denormal slow-down (thread
  oversubscription is the usual cause); graph-classification pooling ranked on one seed with the test set
  printed during training (now a validation split and 5 seeds; the old ranking did not hold); GAN mode-coverage
  metric ignored low-quality samples; DCGAN classifier accuracy was in-sample; `torch>=2.6` because
  `torch.load(weights_only=True)` was exploitable before (CVE-2025-32434); wording nits.
- **p07 new chapters:** dataset alias that rewarded restating the question; fake-model testing claim; empty
  plan crash; GiB units; chunk budget overshoot; over-broad forbidden-phrase check.
- **p07 c00/c01/c03/c04 + p08:** Unsloth silently switches `loss_type="dapo"` to DAPO settings (now `bnpo`);
  numeric answer parsing; **attribution** of code adapted from Unsloth's LGPL-3.0 GRPO notebook and Will
  Brown's gist (headers and sources added); batch-size and model-name wording; which model the p08 agents use.

## What was checked and found correct

Not every check found a problem. Things the reviewers confirmed:

- **No test-set misuse** in p05 and p06: the test set is never used for early stopping, model selection or
  threshold choice. Boosting from scratch matches scikit-learn to 1e-15; OLS matches to 1e-12; the hand-written
  GCN layer matches PyG's `GCNConv` exactly; the NumPy MLP's gradient check is correct.
- **Numbers in the book match the code** almost everywhere: all of p05 and p06, including Cora and
  Fashion-MNIST, were reproduced. The exceptions are listed above and are now regenerated.
- **Library APIs match the installed versions:** TRL 0.24 `GRPOConfig` fields, defaults and reward-function
  signature (checked in the wheel); Unsloth 2026.1.4; LangGraph `Send`/reducers; MLflow `genai.evaluate`; kaggle
  2.x, openml, datasets, huggingface_hub, APScheduler 3.x.
- **Security:** `safe_unzip` resisted all path-traversal attempts; the container runs non-root with a
  read-only filesystem, and the Kubernetes pod meets the "restricted" Pod Security Standard.
- **Container:** the image was built and run with a real Docker daemon, and compose works.
- **The BPE tokeniser exported to tiktoken gives identical ids,** confirmed by fuzzing thousands of
  adversarial strings.
- **Earlier fixes to the owner's minGPT code are correct** (validation pairs no longer leak into training),
  and the book's statements about the existing code (attention scaling in tinygpt, no weight tying in minGPT)
  are accurate.

## What could not be verified

- **Live services:** real runs against Ollama, Qdrant, OpenAI, a GPU (GRPO and Unsloth training), Kubernetes
  clusters and cloud deployments. These were exercised with mocks or checked statically.
- **Blocked downloads:** Hugging Face Hub, Kaggle, OpenML, toscrape, Open-Meteo and Frankfurter; the tiktoken
  `cl100k_base` vocabulary.
- **Web sources:** some external facts come from the reviewers' web searches and could not be re-checked from
  the sandbox. These are the App Runner closure date, Ollama's VRAM-based default context, the Frankfurter v1
  deprecation and the torch CVE.

## Open items and owner decisions

1. **Tracked `.env` files - resolved.** The owner confirmed that `p07_llms/c04_rag_systems/.env` and
   `p08_agents/c00_langchain_langgraph/.env` contain no secrets and are tracked on purpose, for convenience. The
   fundamentals chapter now describes them as a deliberate exception.
2. **`p08 s01/main_3_simple_tools.py` - resolved.** It was meant to loop: the edge `tools -> chatbot` was added,
   so the model now calls tools until it can answer (verified with a scripted model: 24 -> 25 -> 100, then the
   final answer). The book, README and `graph_3.png` were updated.
3. **Frankfurter API - resolved.** The example now uses v2 (`/v2/rates` with `providers=ecb`, long-format
   records pivoted to one column per currency, carried-forward rows before the start date dropped). The v2
   routes and response format were taken from the upstream source code, because the documentation site was
   blocked in the sandbox; the script was tested against a local fake of the API, not against the live service.
4. **Pinned exports.** Six older conda env files are full pinned Linux exports, some with a `prefix:` line
   pointing at the original machine. Regenerating them with `--from-history` would make them portable.
5. **Small PDF overflows.** A few lines still run slightly into the margin in the older `book_agents.md` table
   (bold words in a narrow column).
