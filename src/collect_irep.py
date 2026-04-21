"""
Collecte des données IREP - Registre des émissions polluantes
Source : files.georisques.fr
Années : 2021-2024 (donnée statique annuelle)
Filtre : Île-de-France
"""

import os
import logging
import zipfile
from pathlib import Path

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────

ANNEES_IREP = [2021, 2022, 2023, 2024]
BASE_URL    = "https://files.georisques.fr/irep/{annee}.zip"
DEPTS_IDF   = ["75", "77", "78", "91", "92", "93", "94", "95"]

RAW_DIR = Path("data/raw/irep")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Téléchargement et extraction ─────────────────────────────────

def download_and_extract(url: str, annee: int) -> Path:
    """
    Télécharge le ZIP IREP pour une année donnée et extrait les fichiers.
    Retourne le dossier d'extraction. Skip si déjà extrait.
    """
    zip_dest    = RAW_DIR / f"irep_{annee}.zip"
    extract_dir = RAW_DIR / str(annee)

    if extract_dir.exists():
        logger.info(f"Déjà extrait : {extract_dir} — skip")
        return extract_dir

    if not zip_dest.exists():
        logger.info(f"Téléchargement IREP {annee}...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        zip_dest.write_bytes(r.content)
        logger.info(f"Téléchargé : {zip_dest.name} ({zip_dest.stat().st_size / 1e6:.1f} Mo)")

    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_dest, "r") as z:
        z.extractall(extract_dir)
        logger.info(f"Fichiers extraits : {z.namelist()}")

    return extract_dir





# ── Chargement et filtrage IDF ────────────────────────────────────

def load_and_filter(extract_dir: Path, annee: int) -> pd.DataFrame:
    """
    Charge emissions.csv (polluants atmosphériques) et etablissements.csv
    (coordonnées GPS), filtre sur l'IDF, corrige l'encodage et joint les deux.
    """
    sub = extract_dir / str(annee)
    if not sub.exists():
        sub = extract_dir

    emissions_path      = sub / "emissions.csv"
    etablissements_path = sub / "etablissements.csv"

    # ── Emissions ──
    df_em = pd.read_csv(emissions_path, sep=";", encoding="utf-8", low_memory=False)
    df_em.columns = df_em.columns.str.strip().str.lower()
    

    df_em_idf = df_em[
        df_em["code_departement"].astype(str).str.strip().isin(DEPTS_IDF)
    ].copy()
    logger.info(f"  emissions.csv IDF : {len(df_em_idf)} / {len(df_em)} lignes")

    # Filtre milieu AIR uniquement
    if "milieu" in df_em_idf.columns:
        df_em_idf = df_em_idf[
            df_em_idf["milieu"].astype(str).str.upper().str.contains("AIR")
        ].copy()
        logger.info(f"  Après filtre AIR : {len(df_em_idf)} lignes")

    # ── Etablissements (coordonnées GPS) ──
    df_et = pd.read_csv(etablissements_path, sep=";", encoding="utf-8", low_memory=False)
    df_et.columns = df_et.columns.str.strip().str.lower()
   

    df_et_idf = df_et[
        df_et["code_departement"].astype(str).str.strip().isin(DEPTS_IDF)
    ][["identifiant", "coordonnees_x", "coordonnees_y", "code_epsg"]].copy()
    logger.info(f"  etablissements.csv IDF : {len(df_et_idf)} établissements")

    # ── Jointure ──
    df_merged = df_em_idf.merge(df_et_idf, on="identifiant", how="left")
    df_merged["annee"] = annee
    logger.info(f"  Après jointure : {len(df_merged)} lignes")

    return df_merged


# ── Upload S3 ────────────────────────────────────────────────────

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
    key    = f"projet-qualite-air/raw/irep/{filepath.name}"
    s3.upload_file(str(filepath), bucket, key)
    logger.info(f"Uploadé sur S3 : s3://{bucket}/{key}")


# ── Fonction principale ──────────────────────────────────────────

def collect_irep() -> None:
    """
    Collecte les données IREP pour 2021-2024,
    filtre sur l'IDF et consolide en un seul fichier encodé en utf-8.
    """
    all_dfs = []

    for annee in ANNEES_IREP:
        url = BASE_URL.format(annee=annee)
        try:
            extract_dir = download_and_extract(url, annee)
            df          = load_and_filter(extract_dir, annee)
            all_dfs.append(df)
            logger.info(f"IREP {annee} : {len(df)} lignes IDF chargées")
        except Exception as e:
            logger.error(f"Erreur pour {annee} : {e}")
            continue

    if not all_dfs:
        raise RuntimeError("Aucune donnée IREP collectée")

    df_all = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total consolidé : {len(df_all)} lignes | {df_all['annee'].value_counts().to_dict()}")

    # Sauvegarde en utf-8 explicite
    out = RAW_DIR / "irep_2021_2024_idf.csv"
    df_all.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"Sauvegardé localement : {out}")

    upload_to_s3(out)
    logger.info("Collecte IREP terminée.")


if __name__ == "__main__":
    collect_irep()