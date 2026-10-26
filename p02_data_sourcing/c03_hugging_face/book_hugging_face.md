## Hugging Face Datasets

The Hugging Face Hub (https://huggingface.co/datasets) hosts a very large number of datasets for text,
audio, images, video and tabular data, next to models and demo apps. Each dataset is a git repository with
data files (usually Parquet, but also CSV, JSON lines, images, audio, ...) and a *dataset card* (`README.md`)
that documents content, licence, intended use and limitations in a YAML header plus free text. Two libraries
are used to access it:

- **`datasets`** - loads, processes and saves datasets efficiently on top of Apache Arrow,
- **`huggingface_hub`** - the client for the Hub itself: search, metadata, single-file downloads, uploads.

The repository already used `load_dataset("gbharti/finance-alpaca")` for the mini-GPT in part p07; this
chapter explains what happens behind that call.

### Regular versus streaming mode
`load_dataset(repo_id)` in **regular mode** downloads the data files, converts them to Arrow files in a local
cache (`~/.cache/huggingface` by default, `cache_dir=` to change it) and *memory-maps* them. The result is a
`DatasetDict` with one `Dataset` per split. Memory mapping means that a dataset much larger than RAM can be
used: only the pages that are accessed are loaded. You get `len()`, random access (`ds[123]`), column access
(`ds["label"]`), fast slicing and a cache that is re-used next time. Split slicing selects parts of splits
directly: `split="train[:10%]+test[:10%]"`.

With `streaming=True` nothing is downloaded up front: an `IterableDataset` reads examples lazily while you
iterate, shard by shard. Transformations (`map`, `filter`) are applied on the fly, `take(n)`/`skip(n)` select
examples, and `shuffle(buffer_size=...)` shuffles approximately by sampling from a buffer (plus shuffling the
order of the shards). Streaming is the right choice for datasets that do not fit on disk (web-scale text
corpora), for a quick look at a dataset, and for training loops that consume data once. There is no `len()`
and no random access.

`main_1_load_and_stream.py` loads the small *rotten_tomatoes* sentiment dataset (about 10,000 movie-review
sentences, Pang and Lee 2005) in regular mode, inspects features (`Value`, `ClassLabel` with label names), random
access, the Arrow cache files and split slicing, and then streams the larger *imdb* dataset (50,000 reviews,
Maas et al. 2011) with `take`, `shuffle`, `filter` and `map`, comparing the time to the first examples.

### Processing and saving
`main_2_process_and_save.py` shows the typical processing chain:

1. `map(fn, batched=True)` - `fn` receives a dictionary of lists (one list per column, `batch_size` rows) and
   returns new or updated columns. Batched functions are much faster than per-example functions;
   `num_proc=` parallelises over processes for large datasets. Results are cached on disk, keyed by a
   fingerprint of the function and its inputs, so re-running the script is instant.
2. `filter(fn)` - keeps the rows for which `fn` returns `True`.
3. `train_test_split(test_size=0.1, stratify_by_column="label", seed=42)` - creates a validation split with
   the same class ratio as the training data (stratification requires a `ClassLabel` column).
4. `to_pandas()` - for quick statistics with familiar tools.
5. `save_to_disk(path)` / `load_from_disk(path)` - stores the whole `DatasetDict` as Arrow files plus
   metadata; the fastest way to persist an intermediate result for later steps.
6. `to_parquet(path)` - portable files that pandas, Spark, DuckDB and others can read. Local files are
   loaded with the same API: `load_dataset("parquet", data_files={"train": ..., "test": ...})` (also `"csv"`,
   `"json"`, `"text"`, `"imagefolder"`, ...). `datasets` stores its feature definitions (e.g. the label
   names) in the Parquet schema metadata, so they survive the round trip.

The script runs without internet access with `--offline` (or automatically if the Hub is unreachable): it then
uses a tiny built-in dataset with the same schema (`utils/offline_data.py`).

### Working with the Hub directly
`main_3_hub_files.py` uses `huggingface_hub`:

- `HfApi().list_datasets(search="sentiment", sort="downloads", limit=10)` searches the Hub,
- `HfApi().dataset_info(repo_id)` returns metadata: last commit, tags, card data (licence, task categories),
- `list_repo_files(repo_id, repo_type="dataset")` lists the files of the repository,
- `hf_hub_download(repo_id, filename, repo_type="dataset", revision=...)` downloads a single file (cached and
  resumable); `revision` pins a branch, tag or commit for reproducibility,
- `snapshot_download(repo_id, repo_type="dataset", allow_patterns=["*.parquet"])` downloads parts of a repository.

### Access, licences and good practice
- Gated datasets (the authors require you to accept conditions) and private datasets need a token: create
  one in your Hugging Face settings and provide it via the environment variable `HF_TOKEN` or with
  `hf auth login`, never in code.
- Read the dataset card: licence, how the data was collected, known biases and limitations, and personal
  data. Licences vary from permissive (CC BY, MIT) to research-only.
- Pin revisions for reproducible pipelines; datasets are repositories and can change.
- Offline use: set `HF_HUB_OFFLINE=1` to use only the local cache.
