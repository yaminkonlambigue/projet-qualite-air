"""
Collecte des données LCSQA - PM2.5 horaire via API Geod'air
Source : geodair.fr
Polluant : PM2.5 (code 39)
Statistique : moyenne horaire (code a1)
Période : 2021-2025
Zone : Île-de-France (filtrage post-téléchargement)
"""

import os
import time
import logging
from pathlib import Path
from datetime import date

import boto3
import pandas as pd
import requests
from io import StringIO
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GEODAIR_BASE  = "https://www.geodair.fr/api-ext"
CODE_POLLUANT = "39"
CODE_STAT     = "a1"

#DEPTS_IDF = ["75", "77", "78", "91", "92", "93", "94", "95"]

TRIMESTRES = [
#   (date(2021,  1,  1), date(2021,  3, 31)),
#    (date(2021,  4,  1), date(2021,  6, 30)),
#    (date(2021,  7,  1), date(2021,  9, 30)),
#    (date(2021, 10,  1), date(2021, 12, 31)),
#    (date(2022,  1,  1), date(2022,  3, 31)),
#    (date(2022,  4,  1), date(2022,  6, 30)),
#    (date(2022,  7,  1), date(2022,  9, 30)),
#    (date(2022, 10,  1), date(2022, 12, 31)),
#    (date(2023,  1,  1), date(2023,  3, 31)),
#    (date(2023,  4,  1), date(2023,  6, 30)),
#    (date(2023,  7,  1), date(2023,  9, 30)),
#    (date(2023, 10,  1), date(2023, 12, 31)),
#    (date(2024,  1,  1), date(2024,  3, 31)),
#   (date(2024,  4,  1), date(2024,  6, 30)),
#    (date(2024,  7,  1), date(2024,  9, 30)),
    (date(2024, 10,  1), date(2024, 12, 31)),
    (date(2025,  1,  1), date(2025,  3, 31)),
    (date(2025,  4,  1), date(2025,  6, 30)),
    (date(2025,  7,  1), date(2025,  9, 30)),
    (date(2025, 10,  1), date(2025, 12, 31)),
]

RAW_DIR = Path("data/raw/lcsqa")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_headers() -> dict:
    return {"apikey": os.getenv("GEODAIR_TOKEN")}


def request_file(date_debut: date, date_fin: date) -> str | None:
    params = {
        "date_debut" : date_debut.strftime("%d/%m/%Y 00:00"),
        "date_fin"   : date_fin.strftime("%d/%m/%Y 23:59"),
        "type_donnee": CODE_STAT,
        "polluant"   : CODE_POLLUANT,
    }
    r = requests.get(
        f"{GEODAIR_BASE}/statistique/export",
        params=params,
        headers=get_headers(),
        timeout=60
    )
    if r.status_code == 200 and r.text.strip():
        return r.text.strip()
    logger.error(f"Erreur generation : {r.status_code} | {r.text[:200]}")
    return None


def download_file(file_id: str, max_retries: int = 10) -> str | None:
    for attempt in range(max_retries):
        # Attendre plus longtemps au début
        wait = 10 if attempt == 0 else 5
        time.sleep(wait)
        r = requests.get(
            f"{GEODAIR_BASE}/download",
            params={"id": file_id},
            headers=get_headers(),
            timeout=120
        )
        if r.status_code == 200 and "text/csv" in r.headers.get("content-type", ""):
            return r.text
    logger.error(f"Echec telechargement apres {max_retries} tentatives")
    return None


def filter_idf(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["code site"].astype(str).str.startswith("FR04")
    return df[mask].copy()


def process_trimestre(date_debut: date, date_fin: date) -> Path | None:
    fname = f"lcsqa_pm25_{date_debut.strftime('%Y%m%d')}_{date_fin.strftime('%Y%m%d')}.csv"
    dest  = RAW_DIR / fname

    if dest.exists():
        logger.info(f"Deja traite : {fname} - skip")
        return dest

    logger.info(f"Traitement : {date_debut} -> {date_fin}")

    file_id = request_file(date_debut, date_fin)
    if not file_id:
        return None

    content = download_file(file_id)
    if not content:
        return None

    df = pd.read_csv(StringIO(content), sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip().str.replace('"', '')
    logger.info(f"  Lignes brutes : {len(df)}")

    df_idf = filter_idf(df)
    logger.info(f"  Lignes IDF : {len(df_idf)}")

    if df_idf.empty:
        logger.warning(f"  Aucune donnee IDF")
        return None

    df_idf.to_csv(dest, index=False)
    logger.info(f"  Sauvegarde : {fname} ({dest.stat().st_size / 1e6:.1f} Mo)")
    return dest


def upload_to_s3(filepath: Path) -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url          = os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id     = os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key = os.getenv("S3_SECRET_KEY"),
        aws_session_token     = os.getenv("S3_SESSION_TOKEN"),
    )
    bucket = os.getenv("S3_BUCKET")
    key    = f"projet-qualite-air/raw/lcsqa/{filepath.name}"
    s3.upload_file(str(filepath), bucket, key)
    logger.info(f"  Upload S3 : s3://{bucket}/{key}")


def collect_lcsqa() -> None:
    logger.info(f"Demarrage collecte LCSQA - {len(TRIMESTRES)} trimestres")

    for date_debut, date_fin in tqdm(TRIMESTRES, desc="Trimestres"):
        dest = process_trimestre(date_debut, date_fin)
        if dest:
            upload_to_s3(dest)
        time.sleep(2)

    logger.info("Collecte LCSQA terminee.")


if __name__ == "__main__":
    collect_lcsqa()