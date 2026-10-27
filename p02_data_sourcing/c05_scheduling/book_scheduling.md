## Scheduling Data Collection

Many data sources are not "download once" but change continuously: prices, weather, exchange rates, sensor
readings, new documents. Collecting such data means running an ingestion job *repeatedly* - every few
minutes, hourly or nightly - for weeks or months, without anyone watching. This chapter shows how to schedule
jobs from Python with **APScheduler** and with the classic Unix tool **cron**, and how to write the job itself
so that it is robust.

### APScheduler concepts
APScheduler (Advanced Python Scheduler) runs Python callables on a schedule inside a Python process. It is
built from four components:

- **Triggers** decide *when* a job runs:
  - `date` - exactly once at a given time,
  - `interval` - every N weeks/days/hours/minutes/seconds, optionally with `jitter` (random offset to avoid
    many jobs hitting a service at the same moment) and `start_date`/`end_date`,
  - `cron` - calendar-based rules with cron semantics, e.g. `hour=2, minute=0` (nightly at 02:00),
    `day_of_week="mon-fri", hour="8-18", minute="*/15"`; also `CronTrigger.from_crontab("0 2 * * *")`.
- **Job stores** keep the job definitions: in memory (default) or persistent (SQLAlchemy, MongoDB, Redis), so
  that jobs survive a restart of the scheduler process.
- **Executors** run the jobs: a thread pool (default, good for I/O-bound jobs such as downloads) or a process
  pool (for CPU-bound work).
- **Schedulers** tie everything together. `BlockingScheduler` occupies the main thread and is the right
  choice for a dedicated scheduler process (as in our examples); `BackgroundScheduler` runs in a background
  thread next to an application (e.g. a web service); `AsyncIOScheduler` integrates with asyncio.

Important job options (set per job or as `job_defaults` for the scheduler):

- `max_instances` - how many runs of the same job may overlap; `1` prevents a slow run from being started
  again before the previous one has finished (the extra run is skipped and an event is emitted),
- `coalesce` - if several runs were missed (e.g. the machine was asleep), run once instead of catching up on all,
- `misfire_grace_time` - how late a run may start and still be executed.

Exceptions raised in a job do not stop the scheduler: they are logged and reported as `EVENT_JOB_ERROR` to
event listeners registered with `add_listener`. The examples use the APScheduler 3.x API, the stable line at
the time of writing (version 4 redesigns the API).

`main_1_apscheduler_basics.py` demonstrates all of this in 20 seconds: an interval job, a cron job
(`second="*/5"`), a slow job whose overlapping runs are skipped because of `max_instances=1`, a flaky job whose
exceptions are reported by a listener, and a one-off date job. A `threading.Timer` calls `shutdown()` after the
given duration so the demo terminates.

### A robust ingestion job
`main_2_scheduled_ingestion.py` is a template for real ingestion pipelines. Each run fetches observations
(synthetic weather data by default, or the current weather from the free Open-Meteo API with
`--source open-meteo`), validates them and stores them; a cron job exports a Parquet snapshot. The design
principles are more important than the specific source:

- **Idempotency.** The SQLite table has a natural primary key `(source, location, observed_at)` and rows are
  written with `INSERT ... ON CONFLICT DO NOTHING`. Running the job twice for the same period - after a retry,
  an overlapping schedule or a manual re-run - does not create duplicates. In the demo the synthetic source
  produces one observation per 10-second bucket while the job runs every 3 seconds, so duplicates are
  frequent and visibly ignored; Open-Meteo reports 15-minute values, with the same effect.
- **Transactions.** Each batch is inserted in one transaction: either all rows are stored or none.
- **Error handling.** Every run catches its exceptions, logs them and records them in the `runs` table. A
  failing source (simulated with `--failure-rate`) costs one run, not the whole scheduler; the next run simply
  tries again. HTTP requests use timeouts and retries with back-off.
- **Validation.** Implausible records (e.g. temperatures outside -60..60 °C) are dropped and counted before
  anything is stored.
- **Observability.** Logs go to the console and to a log file; the `runs` table holds start, end, status,
  number of fetched and inserted rows and the error message of every run - the basis for monitoring and alerts.
- **Atomic outputs.** The snapshot is written to a temporary file and renamed, so consumers never read a
  half-written file.

The `--once` flag executes exactly one ingestion run and exits - the mode used with external schedulers.

### Scheduling with cron
On Linux and macOS, **cron** runs commands on a schedule defined in a *crontab* (edit with `crontab -e`, list
with `crontab -l`). Each line has five time fields followed by the command:

```text
# minute  hour  day-of-month  month  day-of-week  command
  */15    *     *             *      *            <command>     # every 15 minutes
  0       2     *             *      *            <command>     # every night at 02:00
  30      7     *             *      1-5          <command>     # weekdays at 07:30
```

To run the ingestion every 15 minutes with the conda environment of this part:

```text
*/15 * * * * cd /path/to/genai-zero-to-hero && /path/to/miniconda3/envs/gzh_data0/bin/python -m p02_data_sourcing.c05_scheduling.main_2_scheduled_ingestion --once >> /tmp/gzh_ingestion_cron.log 2>&1
```

Things to know about cron:

- cron starts commands with a minimal environment: use absolute paths (or set `PATH`/`SHELL` at the top of the
  crontab), `cd` into the working directory and call the environment's Python directly instead of
  `conda activate`.
- Redirect stdout/stderr to a log file; otherwise output is mailed to the user or lost.
- cron does not prevent overlapping runs; wrap long jobs with `flock -n /tmp/ingest.lock <command>`.
- Times are in the system time zone; be careful around daylight-saving changes (jobs between 02:00 and 03:00
  may run twice or not at all).
- https://crontab.guru helps to read and write cron expressions.

### Choosing a scheduler
| tool                         | good for                                                                 |
|------------------------------|--------------------------------------------------------------------------|
| cron / systemd timers        | single machine, simple commands; systemd timers add logging and catch-up |
| APScheduler                  | scheduling inside a Python application or a small dedicated process      |
| Airflow, Prefect, Dagster    | many dependent jobs (DAGs), retries, back-filling, UI, team workflows     |
| cloud schedulers             | serverless jobs (e.g. Cloud Scheduler, EventBridge, Kubernetes CronJob)  |

Whatever the tool, the job itself should follow the principles above - the scheduler only decides *when* it
runs, the job must make sure that running it is always safe.
