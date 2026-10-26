## Storing Sourced Data

Before we collect any data we should decide where and how to keep it. The choice of storage format and
layout determines how fast the data can be read, whether its types survive, how much disk space it needs,
whether a pipeline can be re-run safely and how easily other tools (pandas, Spark, DuckDB, BI tools) can
use it. This chapter compares the most common formats for tabular data and builds a small local
"data lake" that follows the same conventions as large cloud data platforms.

### File formats for tabular data
**CSV** (comma-separated values) is the lowest common denominator: plain text, readable by humans and by
every tool. Its drawbacks are equally fundamental: there is no schema, so types must be guessed when reading
(dates and categoricals come back as strings, leading zeros disappear, `"NA"` becomes missing); it is row
oriented, so reading two columns still means parsing every line; and it is large unless compressed.
Compression (`.csv.gz`) reduces size substantially but makes writing and reading slower.

**Apache Parquet** is a columnar, compressed and typed file format and the de-facto standard for analytical
data. Columns are stored contiguously and split into *row groups*; each column chunk carries statistics
(minimum, maximum, null count). This enables

- *column projection:* only the requested columns are read from disk,
- *predicate push-down:* row groups whose statistics cannot match a filter (e.g. `ts >= 2025-03-15`) are
  skipped without being decompressed - very effective when data is sorted by the filter column,
- *efficient compression:* similar values sit next to each other; codecs such as snappy (fast) and zstd
  (smaller) are typical,
- *schema preservation:* timestamps, categoricals (dictionary encoding), booleans, nullable types and even
  nested lists survive a round trip.

**Feather / Arrow IPC** stores Apache Arrow's in-memory layout directly on disk. Reading and writing is
extremely fast, which makes it ideal for short-lived intermediate files; for long-term storage Parquet is
usually preferred because it compresses better and is more widely supported.

**SQLite** is a complete relational database in a single file, included in Python's standard library.
It offers SQL, indexes, constraints (e.g. primary keys) and transactions - exactly what is needed by small
operational stores, such as a scheduled job that appends records and must never store duplicates
(see chapter c05). Its type system is loose (any column can hold any type), so typed analytics are better
served by Parquet. With SQLAlchemy the same pandas code (`to_sql`, `read_sql`) works against PostgreSQL,
MySQL and other databases.

The script `main_1_file_formats.py` writes 200,000 synthetic transactions to all of these formats.
A typical result on a laptop (exact numbers vary):

| format           | size (MB) | write (s) | read (s) | types preserved                      |
|------------------|-----------|-----------|----------|--------------------------------------|
| csv              | 12.6      | 0.5       | 0.03     | no (timestamp, category -> string)    |
| csv.gz           | 2.7       | 2.2       | 0.10     | no                                   |
| parquet (snappy) | 3.5       | 0.4       | 0.09     | yes                                  |
| parquet (zstd)   | 2.7       | 0.2       | 0.03     | yes                                  |
| feather          | 5.6       | 0.03      | 0.02     | yes                                  |
| sqlite           | 14.3      | 2.1       | 1.2      | partly (category lost)               |

The script then demonstrates selective reads: reading two columns, filtering with `filters=` on a Parquet
file with eight row groups (and printing the min/max statistics that make skipping possible), and an
aggregation with SQL in SQLite.

### A local data lake
Data platforms organise data in *zones* (also called layers; Databricks popularised the names bronze,
silver and gold in the "medallion architecture"):

- **raw (bronze):** data exactly as received, immutable, typically JSON/CSV, partitioned by ingestion date.
  If anything downstream goes wrong, the raw zone allows a full replay.
- **curated (silver):** cleaned, typed, de-duplicated data in Parquet, partitioned by a business key such
  as the event date.
- **aggregated (gold):** tables built for a specific use, e.g. daily revenue per country for a dashboard.

On disk, partitions are folders named `key=value` (*Hive-style partitioning*), e.g.
`curated/transactions/date=2025-03-01/part-0.parquet`. Readers such as pandas, pyarrow, Spark and DuckDB
recognise the scheme, turn `date` into a column and read only the folders that match a filter
(*partition pruning*). Choose partition keys with low to moderate cardinality that appear in filters;
too many tiny files are as harmful as one huge file.

`main_2_local_data_lake.py` implements this layout with pyarrow and SQLite:

1. Each incoming batch (a list of JSON records, as returned by an API) gets a **content hash** (SHA-256 of
   the canonical JSON). The **catalog** (`catalog.sqlite`) records every ingested batch with hash, source,
   ingestion date, path and row count.
2. If the hash is already in the catalog, the batch is skipped - re-deliveries, which are common when
   upstream systems retry, cannot create duplicates.
3. New batches are landed unchanged (gzip-compressed JSON) in the raw zone, then curated: schema check,
   type conversion, de-duplication on `transaction_id`, and a `date` column for partitioning.
4. Curated data is written with `pyarrow.dataset.write_dataset(..., existing_data_behavior="delete_matching")`:
   re-processing a day **replaces** that day's partition instead of appending to it. A corrected batch for
   an earlier day therefore overwrites the old data - the second pillar of idempotency.
5. The catalog entry is written only after the data has been stored successfully, so a crash in between
   leads to a re-run rather than to a silently missing batch.
6. A gold table with daily revenue is written both as Parquet and into SQLite for SQL users.

Running the script twice shows that the second run skips every batch. In the cloud the same design maps
to object storage (S3, ADLS, GCS) for the zones, a metastore or table format (Hive metastore, AWS Glue,
Delta Lake, Apache Iceberg) for the catalog, and Spark or DuckDB as the engine.

### Practical recommendations
- Store analytical data as Parquet (zstd or snappy), sorted by the most common filter column.
- Keep raw data immutable and record lineage: source, retrieval time, version or hash.
- Use natural keys or content hashes plus upserts or partition overwrites to make pipelines idempotent.
- Use SQLite (or PostgreSQL) for small operational state: run logs, catalogs, incremental "last seen" markers.
- Write files atomically (write to a temporary name, then rename) so readers never see half-written files.
