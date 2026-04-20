"""
Collecte des données LCSQA - Concentrations de polluants atmosphériques
Source : data.gouv.fr / INERIS
Polluant cible : PM2.5
Période : 2021-2025 
Zone : Île-de-France (filtrage post-téléchargement)
"""

import io
import logging
from pathlib import Path

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from tqdm import tqdm
import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration 

BASE_URL = (
    "https://files.data.gouv.fr/lcsqa/"
    "concentrations-de-polluants-atmospheriques-reglementes/temps-reel"
)

STATIONS_URL = (
    "https://static.data.gouv.fr/resources/"
    "donnees-temps-reel-de-mesure-des-concentrations-de-polluants-atmospheriques-reglementes-1/"
    "20251210-084445/fr-2025-d-lcsqa-ineris-20251209.xls"
)

ANNEES = [2021, 2022, 2023, 2024, 2025]
POLLUANT = "PM2.5"

# Départements Île-de-France
DEPTS_IDF = ["75", "77", "78", "91", "92", "93", "94", "95"]

# Exclusion JO Paris 2024
JO_START = "2024-07-26"
JO_END = "2024-08-11"

RAW_DIR = Path("data/raw/lcsqa")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Fonctions de collecte 

def build_url(annee: int, mois: int, jour: int) -> str:
    return f"{BASE_URL}/{annee}/FR_E2_{annee}-{mois:02d}-{jour:02d}.csv"


def download_file(url: str, dest: Path) -> bool:
    """Télécharge un fichier CSV depuis une URL vers un chemin local."""
    if dest.exists():
        logger.info(f"Déjà téléchargé : {dest.name} — skip")
        return True
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        dest.write_bytes(response.content)
        logger.info(f"Téléchargé : {dest.name} ({len(response.content) / 1e6:.1f} Mo)")
        return True
    except requests.HTTPError as e:
        logger.error(f"Erreur HTTP {e.response.status_code} pour {url}")
        return False
    except Exception as e:
        logger.error(f"Erreur inattendue pour {url} : {e}")
        return False


def download_stations_metadata() -> pd.DataFrame:
    """
    Télécharge le fichier de métadonnées des stations (Dataset D)
    contenant les coordonnées GPS de chaque station.
    """
    dest = RAW_DIR / "stations_metadata.xls"

    if not dest.exists():
        response = requests.get(STATIONS_URL, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        logger.info("Métadonnées stations téléchargées")

    df = pd.read_excel(dest)

    # Normaliser les noms de colonnes
    df.columns = df.columns.str.strip().str.lower()

    logger.info(f"Stations chargées : {len(df)} | Colonnes : {df.columns.tolist()}")
    return df


def filter_idf(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre les stations d'Île-de-France."""
    # Le code station commence par le numéro de département
    mask = df["code_station"].astype(str).str[:2].isin(DEPTS_IDF)
    return df[mask].copy()


def exclude_jo(df: pd.DataFrame) -> pd.DataFrame:
    """Exclut la période des JO Paris 2024."""
    mask = ~df["datetime_utc"].between(JO_START, JO_END)
    n_excluded = (~mask).sum()
    if n_excluded > 0:
        logger.info(f"Exclusion JO 2024 : {n_excluded} lignes supprimées")
    return df[mask].copy()


def load_and_clean(filepath: Path) -> pd.DataFrame:
    """Charge et nettoie un fichier CSV LCSQA."""
    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig", low_memory=False)
   
    print("oui1")

    rename_map = {
        "Date de début"        : "datetime_utc",
        "code site"            : "code_station",
        "nom site"             : "nom_station",
        "type d'implantation"  : "type_implantation",
        "Polluant"             : "polluant",
        "type d'influence"     : "type_influence",
        "valeur brute"               : "valeur_ug_m3",
        "unité de mesure"      : "unite",
        "code qualité"         : "code_qualite",
        "validé"               : "valide",
        "X"                    : "lon",
        "Y"                    : "lat",
    }

    # Filtrage PM2.5 après chargement
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    print("oui2")
    df = df[df["polluant"] == "PM2.5"].copy()
    print("oui3")

    # Conversion datetime
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc"])

    # Conversion valeur numérique
    df["valeur_ug_m3"] = pd.to_numeric(
        df["valeur_ug_m3"].astype(str).str.replace(",", "."),
        errors="coerce"
    )

    # Suppression des valeurs physiquement impossibles
    # df = df[(df["valeur_ug_m3"] >= 0) & (df["valeur_ug_m3"] <= 500)]

    return df


# ── Fonction principale ──────────────────────────────────────────

def collect_lcsqa() -> pd.DataFrame:
    """
    Collecte complète LCSQA pour PM2.5, toutes années, IDF uniquement.
    Itère jour par jour, filtre PM2.5 et les stations IDF.
    Retourne un DataFrame consolidé.
    """
    all_dfs = []

    for annee in ANNEES:
        logger.info(f"Début collecte année {annee}")
        debut = datetime.date(annee, 1, 1)
        fin = datetime.date(annee, 12, 31)
        jours = [debut + datetime.timedelta(days=i) for i in range((fin - debut).days + 1)]

        for jour in tqdm(jours, desc=f"{annee}"):
            url = build_url(jour.year, jour.month, jour.day)
            dest = RAW_DIR / f"FR_E2_{jour}.csv"

            success = download_file(url, dest)
            if not success:
                logger.warning(f"Fichier manquant : {jour} — ignoré")
                continue

            try:
                df = load_and_clean(dest)
            except Exception as e:
                logger.error(f"Erreur lecture {dest.name} : {e}")
                continue

            if df.empty:
                continue

            df = filter_idf(df)
            df["annee"] = annee
            all_dfs.append(df)

            # Pause courte pour ne pas surcharger le serveur
            time.sleep(0.2)

        logger.info(f"Année {annee} terminée — {len(all_dfs)} fichiers chargés au total")

    if not all_dfs:
        raise RuntimeError("Aucune donnée collectée — vérifiez les URLs et le réseau")

    consolidated = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        f"Total consolidé : {len(consolidated)} lignes | "
        f"{consolidated['code_station'].nunique()} stations"
    )

    return consolidated


# ── Upload S3 ────────────────────────────────────────────────────

def upload_to_s3(df: pd.DataFrame, filename: str) -> None:
    """Upload un DataFrame en CSV vers S3 SSPCloud."""
    s3 = boto3.client(
        "s3",
        endpoint_url = os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id  = os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key = os.getenv("S3_SECRET_KEY"),
    )
    bucket = os.getenv("S3_BUCKET")
    key = f"raw/lcsqa/{filename}"

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    logger.info(f"Uploadé sur S3 : s3://{bucket}/{key}")


# ── Point d'entrée ───────────────────────────────────────────────

if __name__ == "__main__":
    df = collect_lcsqa()

    # Sauvegarde locale
    out = Path("data/raw/lcsqa_idf_pm25_2021_2025.csv")
    df.to_csv(out, index=False)
    logger.info(f"Sauvegardé localement : {out}")

    # Upload S3
    upload_to_s3(df, "lcsqa_idf_pm25_2021_2025.csv")