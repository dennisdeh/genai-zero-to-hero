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

The script `main_1_file_formats.py` writes 200,000 synthetic transactions to all of these formats. Both
Parquet variants use the same row group size (25,000 rows, i.e. eight row groups), so they differ only in
the compression codec. Before anything is timed, the script writes and reads the full table twice in every
binary format (*warm-up*): the first Parquet calls in a process pay one-time costs - lazy imports, codec and
thread pool start-up, growing Arrow's memory pool - that would otherwise be charged to whichever format
happens to be measured first. Each timing is then the best of three runs. A typical result on a laptop (pandas 3,
pyarrow 25; exact numbers vary, compare ratios rather than absolute values):

| format           | size (MB) | write (s) | read (s) | types preserved                       |
|------------------|-----------|-----------|----------|---------------------------------------|
| csv              | 12.6      | 0.5       | 0.02     | no (timestamp, category -> string)    |
| csv.gz           | 2.7       | 2.1       | 0.09     | no                                    |
| parquet (snappy) | 4.1       | 0.07      | 0.02     | yes                                   |
| parquet (zstd)   | 3.0       | 0.09      | 0.02     | yes                                   |
| feather          | 5.6       | 0.03      | 0.01     | yes                                   |
| sqlite           | 14.3      | 1.9       | 0.76     | partly (category lost)                |

Snappy writes slightly faster, zstd produces a file about a quarter smaller, and both read equally fast.
Gzip-compressed CSV is about as small as Parquet here, but writing it is 20-30 times slower and every type
is lost. Row groups have a price, too: written as a single row group (pyarrow's default for up to about one
million rows) the zstd file shrinks to 2.7 MB, because dictionaries and statistics are stored once instead of
eight times - but then there is nothing to skip when filtering. Without the warm-up, the first Parquet
variant measured looked several times slower than it really is - a classic benchmarking pitfall.

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
   type conversion, de-duplication on `transaction_id`, and a `date` column (the event date) for partitioning.
4. Curated data is **upserted** into the partitions: a batch is grouped by `date`, and for every partition
   it touches the existing rows are read, the new rows appended, duplicates on `transaction_id` dropped
   (`keep="last"`, so the newest version wins) and the partition file replaced atomically (write a temporary
   file, then `os.replace`). Partitions the batch does not touch are left alone. Re-applying a batch gives the
   same result - the second pillar of idempotency - and the demo shows both typical cases: a batch of
   corrections for day 1 updates 1,500 existing records (the other 500 stay), and the batch of day 4 also
   carries five *late-arriving* events from day 2, which are added to day 2's partition (2,000 -> 2,005 rows).

   A plain *partition overwrite* (e.g. `pyarrow.dataset.write_dataset(...,
   existing_data_behavior="delete_matching")` or Spark's dynamic partition overwrite) is simpler and cheaper,
   but it is only correct when every batch is a **complete snapshot** of each partition it contains - for
   example a daily export that always re-sends the whole day. When batches are partitioned by *event* date,
   as here, a batch may carry a handful of late events for an earlier day; overwriting would replace that
   day's 2,000 rows by those few records and silently lose data. Table formats such as Delta Lake and Apache
   Iceberg provide the upsert as `MERGE INTO`, with transactions on top.
5. The catalog entry is written only after the data has been stored successfully, so a crash in between
   leads to a re-run rather than to a silently missing batch.
6. A gold table with daily revenue is written both as Parquet and into SQLite for SQL users.

Running the script twice shows that the second run skips every batch and the curated zone is unchanged. In the cloud the same design maps
to object storage (S3, ADLS, GCS) for the zones, a metastore or table format (Hive metastore, AWS Glue,
Delta Lake, Apache Iceberg) for the catalog, and Spark or DuckDB as the engine.

### Practical recommendations
- Store analytical data as Parquet (zstd or snappy), sorted by the most common filter column.
- Keep raw data immutable and record lineage: source, retrieval time, version or hash.
- Use natural keys or content hashes plus upserts to make pipelines idempotent; overwrite whole partitions
  only when each batch is a complete snapshot of the partitions it writes.
- Use SQLite (or PostgreSQL) for small operational state: run logs, catalogs, incremental "last seen" markers.
- Write files atomically (write to a temporary name, then rename) so readers never see half-written files.
