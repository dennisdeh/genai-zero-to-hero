# Data Processing
Here we discuss data processing methods to clean and prepare data for further analysis.

Data rarely arrives in the shape a model or an analysis needs. Exports from source systems contain
inconsistent spellings, numbers stored as text, mixed date formats, duplicates and impossible values;
scraped text contains HTML, invisible Unicode characters and personal data. A common estimate in
practice is that the larger part of the time in a data project goes into this preparation work - and
mistakes made here (for example leaking test information into the preprocessing) silently invalidate
everything that follows.

This part is organised by the type of data:

- **Structured data** (chapter `c00_structured_data`): tabular data in pandas. We generate a realistic,
  deliberately messy dataset of an online shop (customers and their transactions), clean it step by
  step, engineer features and assemble a leakage-free scikit-learn preprocessing pipeline
  (section `s01_preprocessing`). Section `s02_dqa` turns the expectations about the data into automated
  data quality checks - schema, completeness, uniqueness, ranges, referential integrity - and monitors
  distribution drift between data batches.
- **Unstructured data** (chapter `c01_unstructured_data`): text. We clean and normalise raw text
  (HTML, Unicode, personal data, duplicates) and look at tokenisation - the step that turns text into
  the integer sequences language models work with - by implementing byte-pair encoding (BPE) from
  scratch and comparing it with production tokenisers.

All examples run offline on a CPU in a few seconds; only the finance-alpaca tokenisation example
downloads a dataset from Hugging Face. The clean data of chapter `c00_structured_data` is re-used in
part 4 for an interactive dashboard.

The conda environment for this part is `p03_data_processing/conda_env_gzh_proc0.yml`. All scripts are
run from the repository root, e.g.

```bash
conda env create -f p03_data_processing/conda_env_gzh_proc0.yml
conda activate gzh_proc0
python -m p03_data_processing.c00_structured_data.s01_preprocessing.main_1_cleaning
```
