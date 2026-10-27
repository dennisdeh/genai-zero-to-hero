"""
APScheduler basics: triggers, job options, events and a clean shutdown.

APScheduler (Advanced Python Scheduler) runs Python functions on a schedule inside your own process.
Its four building blocks are
    - triggers   : *when* a job runs (date = once, interval = every N seconds/minutes/..., cron = cron-like rules),
    - job stores : *where* jobs are kept (memory by default; SQLAlchemy/Redis/MongoDB to survive restarts),
    - executors  : *how* jobs run (a thread pool by default, or a process pool for CPU-bound work),
    - schedulers : tie everything together (BlockingScheduler blocks the main thread - ideal for a dedicated
                   scheduler process; BackgroundScheduler runs in a thread next to your application).

The script schedules
    1. an interval job (every 2 seconds),
    2. a cron job (every 5th second, i.e. second="*/5"),
    3. a slow job that takes longer than its interval - with max_instances=1 overlapping runs are skipped,
    4. a job that fails from time to time - errors are caught by APScheduler and reported to a listener,
    5. a one-off date job, which runs exactly once at a given point in time.
The scheduler is shut down after --duration seconds so that the demo terminates.

How to run (from the repository root):
    python -m p02_data_sourcing.c05_scheduling.main_1_apscheduler_basics
    python -m p02_data_sourcing.c05_scheduling.main_1_apscheduler_basics --duration 30

The script uses the APScheduler 3.x API (the stable release line).
"""

import argparse
import logging
import random
import threading
import time
from datetime import datetime, timedelta

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MAX_INSTANCES, JobExecutionEvent
from apscheduler.schedulers.blocking import BlockingScheduler

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# APScheduler logs job errors (with traceback) and skipped runs itself. In this demo our event listener
# reports them in a compact form, so we silence the library logger. In production keep these logs!
logging.getLogger("apscheduler").setLevel(logging.CRITICAL)
START = time.monotonic()


def stamp() -> str:
    """Seconds since the start of the script, for readable output."""
    return f"[t={time.monotonic() - START:5.1f}s]"


def heartbeat() -> None:
    """A trivial job: print that we are alive."""
    print(f"{stamp()} interval job: heartbeat")


def cron_job() -> None:
    """A job triggered by a cron expression."""
    print(f"{stamp()} cron job    : it is {datetime.now():%H:%M:%S} (second is a multiple of 5)")


def slow_job() -> None:
    """A job that takes longer than its interval."""
    print(f"{stamp()} slow job    : started (takes 3s, scheduled every 2s)")
    time.sleep(3)
    print(f"{stamp()} slow job    : finished")


def flaky_job() -> None:
    """A job that raises an exception half of the time."""
    if random.random() < 0.5:
        raise ConnectionError("simulated network failure")
    print(f"{stamp()} flaky job   : succeeded")


def make_listener(stats: dict):
    """Return an event listener that counts executions and reports errors and skipped runs."""

    def listener(event: JobExecutionEvent) -> None:
        if event.code == EVENT_JOB_ERROR:
            stats["errors"] += 1
            print(f"{stamp()} LISTENER    : job '{event.job_id}' failed with {event.exception!r}")
        elif event.code == EVENT_JOB_MAX_INSTANCES:
            stats["skipped"] += 1
            print(f"{stamp()} LISTENER    : run of '{event.job_id}' skipped, previous run still active")
        else:
            stats["ok"] += 1

    return listener


def main(duration_s: int = 20) -> dict:
    """Configure the scheduler, run it for duration_s seconds and return execution statistics."""
    # 1. Scheduler with sensible job defaults
    scheduler = BlockingScheduler(
        job_defaults={
            "coalesce": True,  # if several runs were missed (e.g. machine asleep), run only once
            "max_instances": 1,  # never run the same job concurrently
            "misfire_grace_time": 10,  # a run that is up to 10s late is still executed
        }
    )
    stats = {"ok": 0, "errors": 0, "skipped": 0}
    scheduler.add_listener(make_listener(stats), EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MAX_INSTANCES)

    # 2. Jobs with different triggers
    scheduler.add_job(heartbeat, "interval", seconds=2, id="heartbeat")
    scheduler.add_job(cron_job, "cron", second="*/5", id="cron")  # cron: minute="*/15", hour="2", day_of_week="mon-fri", ...
    scheduler.add_job(slow_job, "interval", seconds=2, id="slow")
    scheduler.add_job(flaky_job, "interval", seconds=3, id="flaky", jitter=1)  # jitter spreads load: each run is delayed by a random 0..+1 s
    # date trigger: run exactly once at a given point in time
    scheduler.add_job(
        lambda: print(f"{stamp()} date job    : runs exactly once"), "date",
        run_date=datetime.now() + timedelta(seconds=duration_s / 2), id="once",
    )

    # 3. Inspect the schedule (next_run_time is only known once the scheduler has started)
    print(f"Scheduled jobs (the scheduler stops after {duration_s}s):")
    for job in scheduler.get_jobs():
        print(f"  {job.id:10s} trigger={job.trigger}")

    # 4. Start: blocks until shutdown() is called. To end the demo, a timer thread calls shutdown() after
    #    duration_s (calling shutdown from *inside* a job would race with the scheduler's bookkeeping).
    stopper = threading.Timer(duration_s, scheduler.shutdown, kwargs={"wait": False})
    stopper.daemon = True
    print("\nStarting the BlockingScheduler, press Ctrl+C to stop early...\n")
    try:
        stopper.start()
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        stopper.cancel()
    print(f"\nScheduler stopped. Executions: {stats}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=int, default=20, help="seconds to run before shutting down")
    args = parser.parse_args()
    main(duration_s=args.duration)
