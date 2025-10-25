import os
from celery import Celery
from celery.schedules import crontab

celery_app = Celery("dltrader",
    broker=os.getenv("CELERY_BROKER_URL","redis://redis:6379/0"),
    backend=os.getenv("CELERY_BACKEND_URL","redis://redis:6379/1"),
)

@celery_app.task
def heartbeat():
    return "alive"

@celery_app.task
def run_daily():
    from tasks import daily_train_and_signal
    return daily_train_and_signal()

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(300.0, heartbeat.s(), name="heartbeat_5min")
    sender.add_periodic_task(
        crontab(hour=22, minute=10, day_of_week="mon-fri", timezone="Europe/Berlin"),
        run_daily.s(),
        name="daily_train_and_signal_eod"
    )
