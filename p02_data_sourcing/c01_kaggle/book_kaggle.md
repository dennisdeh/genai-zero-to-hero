## Kaggle

Kaggle (owned by Google) hosts machine learning competitions and a very large collection of community
datasets, notebooks and models. For learning and prototyping it is often the quickest way to obtain a
realistic dataset - from the classic Titanic and house-price competitions to large image and text corpora.
Two official Python libraries give programmatic access:

- **`kaggle`** - the official API client and command-line tool. It covers the full API: searching and
  downloading datasets, downloading competition data, submitting predictions, creating datasets and
  pushing notebooks.
- **`kagglehub`** - a newer, lightweight library focused on *consuming* resources from Python code with a
  single call (`dataset_download`, `competition_download`, `model_download`), with local caching and
  versioned handles.

### Credentials
Most of the API requires authentication. Credentials are personal secrets and must **never** be written into
code or committed to git. The `kaggle` client looks for them in this order (recent versions):

1. a new-style **API token** in the environment variable `KAGGLE_API_TOKEN` or in the file
   `~/.kaggle/access_token` (created on kaggle.com under *Settings -> API*),
2. a **legacy API key**: the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`, or the file
   `kaggle.json` (downloaded from the same settings page) in `~/.kaggle/` - on Linux also
   `~/.config/kaggle/` - or in the folder given by `KAGGLE_CONFIG_DIR`,
3. cached **OAuth credentials** from a browser-based login with `kaggle auth login`.

Files containing secrets should only be readable by you (`chmod 600 ~/.kaggle/kaggle.json`); the client warns
otherwise. Environment variables are the usual choice for CI pipelines and containers, where they are injected
from a secret store. For competitions, you must additionally accept the competition rules once on the website,
otherwise the download is refused (HTTP 403). Public datasets can often be downloaded with `kagglehub`
without credentials.

The helper `utils/credentials.py` checks which of these sources are present - printing only their *names*,
never their values - so that the scripts can explain what is missing instead of failing with a traceback.

### The `kaggle` API client
`main_1_kaggle_api.py` walks through a typical session:

1. Authenticate: `KaggleApi().authenticate()`. Without credentials the client prints its own help and calls
   `exit(1)`; the script catches this and shows how to configure credentials.
2. Search: `api.dataset_list(search="iris", sort_by="votes")` returns dataset objects with reference
   (`owner/dataset-name`), title, size, download and vote counts and a *usability rating* (Kaggle's score for
   documentation quality: licence, description, file descriptions, ...).
3. Inspect: `api.dataset_list_files("uciml/iris")` lists the files of a dataset before downloading anything.
4. Download: `api.dataset_download_files(ref, path=..., unzip=False)` stores a zip archive;
   `api.competition_download_files("titanic", path=...)` does the same for a competition.
5. Unzip safely: archives from the internet are untrusted input. A crafted archive can contain member names
   such as `../../.bashrc` that would be written outside the target folder ("zip slip"). `utils/files.py`
   resolves every target path and refuses to extract members that escape the destination.
6. Load and summarise every CSV file (and every table of a bundled SQLite database) with pandas: shape,
   dtypes, missing values, numeric summary and the target distribution.

The same operations are available on the command line, which is convenient for quick exploration:

```bash
kaggle datasets list -s iris
kaggle datasets files uciml/iris
kaggle datasets download uciml/iris -p data/ --unzip
kaggle competitions download -c titanic -p data/
```

### kagglehub
`main_2_kagglehub.py` shows the lightweight alternative:

```python
import kagglehub
path = kagglehub.dataset_download("uciml/iris")                   # cached under ~/.cache/kagglehub
path = kagglehub.dataset_download("uciml/iris/versions/2")        # pinned version -> reproducible
path = kagglehub.competition_download("titanic")                  # needs credentials + accepted rules
```

The return value is a local folder with the extracted files. The script passes `output_dir=` so that the data
lands in the chapter's `outputs/` folder instead of the global cache. `kagglehub.dataset_load(...)` with a
`KaggleDatasetAdapter` can load a single file directly into a pandas DataFrame (or a Hugging Face dataset).

### Good practice
- Read the **licence** of every dataset (shown on the dataset page and printed by the client on download):
  many community datasets are CC0 or CC BY, but some are restricted to non-commercial use, and competition
  data is usually limited to the competition.
- Pin dataset **versions** in pipelines; community datasets can change.
- Check the provenance: community datasets are often scraped or re-uploaded copies. Prefer the original
  source (e.g. the UCI repository for the Iris data) when it matters.
- Keep credentials out of notebooks that you share - Kaggle notebooks themselves offer a "secrets" feature.
