"""
Chargement des données brutes depuis S3 SSPCloud.
Trois fonctions principales :
- load_lcsqa()  : concatène les 20 fichiers trimestriels PM2.5
- load_meteo()  : décompresse et concatène les 16 fichiers météo
- load_irep()   : charge le fichier IREP consolidé
"""

import io
import os
import gzip
import logging
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Client S3 ────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url          = os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id     = os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key = os.getenv("S3_SECRET_KEY"),
        aws_session_token     = os.getenv("S3_SESSION_TOKEN"),
    )

def get_bucket() -> str:
    return os.getenv("S3_BUCKET")

def list_s3_files(prefix: str) -> list[str]:
    """Liste les fichiers S3 sous un préfixe donné."""
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=get_bucket(), Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]

def read_s3_csv(key: str, **kwargs) -> pd.DataFrame:
    """Lit un fichier CSV depuis S3."""
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=get_bucket(), Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), **kwargs)

def read_s3_csv_gz(key: str, **kwargs) -> pd.DataFrame:
    """Lit un fichier CSV.gz depuis S3."""
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=get_bucket(), Key=key)
    with gzip.open(io.BytesIO(obj["Body"].read())) as f:
        return pd.read_csv(f, **kwargs)


# ── Chargement LCSQA ─────────────────────────────────────────────

def load_lcsqa() -> pd.DataFrame:
    """
    Charge et concatène les 20 fichiers trimestriels LCSQA PM2.5.
    Retourne un DataFrame avec les colonnes normalisées.
    """
    prefix = "projet-qualite-air/raw/lcsqa/"
    keys   = sorted(list_s3_files(prefix))
    logger.info(f"LCSQA : {len(keys)} fichiers trouvés")

    dfs = []
    for key in keys:
        df = read_s3_csv(key, sep=",", encoding="utf-8")
        df.columns = (df.columns
            .str.strip()
            .str.replace('"', '')
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("'", "_")
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("ascii")
)        
        dfs.append(df)
        logger.info(f"  {key.split('/')[-1]} : {len(df)} lignes")

    consolidated = pd.concat(dfs, ignore_index=True)

    # Renommage des colonnes clés
    rename_map = {
        "date_de_debut"      : "datetime_debut",
        "date_de_fin"        : "datetime_fin",
        "code_site"          : "code_station",
        "nom_site"           : "nom_station",
        "type_d_implantation": "type_station",
        "valeur"             : "pm25_valide",
        "valeur_brute"       : "pm25_brute",
        "code_qualite"       : "code_qualite",
        "validite"           : "validite",
        "latitude"           : "lat",
        "longitude"          : "lon",
    } 
    consolidated = consolidated.rename(columns={k: v for k, v in rename_map.items() if k in consolidated.columns})

    # Conversion datetime
    consolidated["datetime_debut"] = pd.to_datetime(consolidated["datetime_debut"], errors="coerce")

    consolidated["pm25_brute"] = pd.to_numeric(
    consolidated["pm25_brute"].astype(str).str.replace(",", "."),
    errors="coerce"
    )

    logger.info(f"LCSQA consolidé : {len(consolidated)} lignes | {consolidated['code_station'].nunique()} stations")
    return consolidated


# ── Chargement Météo-France ───────────────────────────────────────

# Colonnes utiles uniquement
COLS_METEO = [
    "num_poste", "nom_usuel", "lat", "lon", "alti",
    "aaaammjjhh",   # datetime
    "ff",           # vent vitesse
    "dd",           # vent direction
    "t",            # température
    "u",            # humidité
    "rr1",          # précipitations
    "pres",         # pression
]

def load_meteo() -> pd.DataFrame:
    prefix = "projet-qualite-air/raw/meteo/"
    keys   = sorted(list_s3_files(prefix))
    logger.info(f"Météo : {len(keys)} fichiers trouvés")

    dfs = []
    for key in keys:
        df = read_s3_csv_gz(key, sep=";", encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip().str.lower()

        # Garder uniquement les colonnes utiles disponibles
        cols_disponibles = [c for c in COLS_METEO if c in df.columns]
        df = df[cols_disponibles].copy()

        dfs.append(df)
        logger.info(f"  {key.split('/')[-1]} : {len(df)} lignes")

    consolidated = pd.concat(dfs, ignore_index=True)

    # Renommage
    rename_map = {
        "num_poste"  : "code_station_meteo",
        "nom_usuel"  : "nom_station_meteo",
        "lat"        : "lat_meteo",
        "lon"        : "lon_meteo",
        "alti"       : "altitude",
        "aaaammjjhh" : "datetime_meteo",
        "ff"         : "vent_vitesse_ms",
        "dd"         : "vent_direction_deg",
        "t"          : "temperature_c",
        "u"          : "humidite_pct",
        "rr1"        : "pluie_mm",
        "pres"       : "pression_hpa",
    }
    consolidated = consolidated.rename(columns={k: v for k, v in rename_map.items() if k in consolidated.columns})

    # Conversion datetime (format YYYYMMDDHH)
    consolidated["datetime_meteo"] = pd.to_datetime(
        consolidated["datetime_meteo"].astype(str),
        format="%Y%m%d%H",
        errors="coerce"
    )

    # Température : déjà en dixièmes de °C → convertir en °C
    if "temperature_c" in consolidated.columns:
        consolidated["temperature_c"] = pd.to_numeric(consolidated["temperature_c"], errors="coerce") / 10

    # Pression : déjà en dixièmes de hPa → convertir en hPa
    if "pression_hpa" in consolidated.columns:
        consolidated["pression_hpa"] = pd.to_numeric(consolidated["pression_hpa"], errors="coerce") / 10

    logger.info(f"Météo consolidée : {len(consolidated)} lignes | {consolidated['code_station_meteo'].nunique()} stations")
    return consolidated

# ── Chargement IREP ──────────────────────────────────────────────

def load_irep() -> pd.DataFrame:
    """
    Charge le fichier IREP consolidé (émissions industrielles IDF 2021-2024).
    Retourne un DataFrame avec les colonnes normalisées.
    """
    key = "projet-qualite-air/raw/irep/irep_2021_2024_idf.csv"
    df  = read_s3_csv(key, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    # Renommage colonnes clés
    rename_map = {
        "coordonnees_x" : "lon_irep",
        "coordonnees_y" : "lat_irep",
        "quantite"      : "emission_kg",
        "polluant"      : "polluant_irep",
        "milieu"        : "milieu_irep",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Conversion numérique
    df["emission_kg"] = pd.to_numeric(df["emission_kg"], errors="coerce")

    logger.info(f"IREP : {len(df)} lignes | {df['identifiant'].nunique()} établissements")
    return df