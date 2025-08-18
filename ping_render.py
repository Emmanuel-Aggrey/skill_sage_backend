import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from decouple import config

logging.basicConfig(level=logging.INFO)

scheduler = BackgroundScheduler()
BASE_URL = config("BASE_URL")


def ping_render():
    if BASE_URL:
        try:
            response = requests.get(BASE_URL)
            logging.info(
                f"✅ Pinged {BASE_URL}, Status: {response.status_code}")
        except requests.RequestException as e:
            logging.error(f"❌ Error pinging {BASE_URL}: {e}")
    else:
        logging.warning("⚠️ BASE_URL is not set!")


scheduler.add_job(ping_render, 'interval', minutes=14)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 LIFESPAN STARTED: Starting scheduler")
    scheduler.start()

    yield

    logging.info("🛑 LIFESPAN ENDED: Stopping scheduler")
    scheduler.shutdown()
