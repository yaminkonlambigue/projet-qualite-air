"""
Collecte des données météo horaires Météo-France
Source : meteo.data.gouv.fr / data.gouv.fr
zone : Île-de-France (75, 77, 78, 91, 92, 93, 94, 95)
Années : 2021-2025
"""

import os
import logging
from pathlib import Path

import boto3
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration 

DATASET_ID   = "6569b4473bedf2e7abad3b72"
DATAGOUV_URL = f"https://www.data.gouv.fr/api/1/datasets/{DATASET_ID}/"

DEPTS_IDF = ["75", "77", "78", "91", "92", "93", "94", "95"]
ANNEES    = ["2021", "2022", "2023", "2024", "2025"]

RAW_DIR = Path("data/raw/meteo")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# Filtrage 

def is_relevant(resource: dict) -> bool:
    """
    Retourne True si la ressource concerne un département IDF
    et une année dans notre période d'intérêt.
    """
    title = resource.get("title", "")
    url   = resource.get("url", "")
    ref   = title + url

    dept_ok  = any(f"_{d}_" in ref or f"HOR_{d}" in ref for d in DEPTS_IDF)
    annee_ok = any(a in ref for a in ANNEES)

    return dept_ok and annee_ok


#  Téléchargement 

def download_resource(resource: dict) -> Path | None:
    """
    Télécharge une ressource dans data/raw/meteo/.
    Retourne le Path du fichier téléchargé, ou None si échec.
    Skip si le fichier existe déjà.
    """
    title = resource.get("title", "").replace(" ", "_")
    fmt   = resource.get("format", "csv.gz")
    url   = resource.get("url", "")
    fname = f"{title}.{fmt}" if not title.endswith(fmt) else title
    dest  = RAW_DIR / fname

    if dest.exists():
        logger.info(f"Déjà téléchargé : {fname} — skip")
        return dest

    try:
        response = requests.get(url, timeout=300, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=fname, leave=False
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        logger.info(f"Téléchargé : {fname} ({dest.stat().st_size / 1e6:.1f} Mo)")
        return dest

    except Exception as e:
        logger.error(f"Erreur téléchargement {fname} : {e}")
        if dest.exists():
            dest.unlink()
        return None


#  Upload S3 

def upload_to_s3(filepath: Path) -> None:
    """Upload un fichier local vers S3 SSPCloud."""
    s3 = boto3.client(
        "s3",
        endpoint_url          = os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id     = os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key = os.getenv("S3_SECRET_KEY"),
        aws_session_token     = os.getenv("S3_SESSION_TOKEN"),
    )
    bucket = os.getenv("S3_BUCKET")
    key = f"projet-qualite-air/raw/meteo/{filepath.name}"

    s3.upload_file(str(filepath), bucket, key)
    logger.info(f"Uploadé sur S3 : s3://{bucket}/{key}")


# Fonction principale 

def collect_meteo() -> None:
    """
    Récupère les métadonnées du dataset météo, filtre les ressources
    IDF 2021-2025 et télécharge + uploade chaque fichier sur S3.
    """
    logger.info("Récupération des métadonnées du dataset météo...")
    r = requests.get(DATAGOUV_URL, timeout=30)
    r.raise_for_status()
    resources = r.json().get("resources", [])
    logger.info(f"Total ressources dans le dataset : {len(resources)}")

    relevant = [res for res in resources if is_relevant(res)]
    logger.info(f"Ressources IDF 2021-2025 identifiées : {len(relevant)}")

    if not relevant:
        logger.warning("Aucune ressource trouvée — affichage des 10 premiers titres pour debug :")
        for res in resources[:10]:
            logger.info(f"  {res.get('title')} | {res.get('url')}")
        return

    logger.info("Fichiers à télécharger :")
    for res in relevant:
        logger.info(f"  {res.get('title')}")

    for res in relevant:
        dest = download_resource(res)
        if dest:
            upload_to_s3(dest)

    logger.info("Collecte météo terminée.")


if __name__ == "__main__":
    collect_meteo()