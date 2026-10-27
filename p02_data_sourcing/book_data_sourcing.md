# Data Sourcing
In this part the sourcing of data is discussed.
In particular techniques for data sourcing are discussed for multimodal sources.

Every machine learning project starts with data, and in practice a large share of the work goes into
getting the *right* data into a *usable* shape: finding it, obtaining it legally and reliably, keeping track
of where it came from, and storing it so that later steps (processing, training, evaluation) can rely on it.
This part is about the first half of that journey: how to collect data and how to store it.

The chapters cover the most common ways of obtaining data, ordered roughly from "someone has already
prepared it for you" to "you have to build the pipeline yourself":

- **Storage (c00):** before collecting anything, we look at where to put it. CSV, Parquet, Feather and
  SQLite are compared, and a small local "data lake" with raw, curated and aggregated zones, partitioning,
  a catalog and idempotent ingestion is built.
- **Kaggle (c01):** downloading competition and community datasets with the official `kaggle` client and
  with `kagglehub`, including how to handle API credentials safely.
- **OpenML (c02):** searching thousands of curated, richly annotated datasets programmatically, and fetching
  them with the `openml` package or with scikit-learn's `fetch_openml`.
- **Hugging Face (c03):** the `datasets` library and the Hub, the de-facto standard for text, audio and image
  datasets in deep learning, with regular and streaming loading, processing and saving.
- **Web scraping (c04):** extracting data from web pages with `requests` and BeautifulSoup - politely
  (robots.txt, rate limiting, retries) and with an eye on the legal and ethical side.
- **Scheduling (c05):** turning a one-off download into a recurring, robust ingestion job with APScheduler
  (and cron), including logging, error handling and idempotent storage.
- **APIs (c06):** consuming REST APIs: sessions, timeouts, retries, pagination, rate limits, caching and
  turning nested JSON into tidy DataFrames.

## Common principles in this part
A few principles appear in every chapter and are worth keeping in mind from the start:

1. **Keep the raw data.** Store what you received unchanged (and record when and from where), then derive
   cleaned versions from it. If a processing step turns out to be wrong you can always re-run it.
2. **Make ingestion idempotent.** Running the same job twice must not create duplicates. Natural keys,
   content hashes, upserts and partition overwrites are the standard tools.
3. **Expect failure.** Networks time out, servers return 503, APIs change. Use timeouts, retries with
   back-off, validation and clear error messages - and log what happened.
4. **Respect the source.** Read the licence and terms of use, honour rate limits and robots.txt, never put
   credentials into code, and be careful with personal data (GDPR).
5. **Prefer typed, columnar formats.** Parquet preserves data types and is fast and compact; CSV is fine for
   exchange with humans but loses information.

All examples are designed to run on a laptop in a minute or two. Examples that depend on external services
(Kaggle, OpenML, Hugging Face, public APIs and web sites) fail gracefully with a clear message when the
service cannot be reached, and several of them offer an offline mode (a local sandbox web site, a built-in
dataset or synthetic data) so that the techniques can be practised without internet access.
The conda environment for the whole part is `conda_env_gzh_data0.yml`.
