"""
Jointures spatiales entre les sources de données.
- join_meteo_to_lcsqa()  : associe chaque station LCSQA à la station météo
                           la plus proche puis merge temporel exact sur l'heure
- compute_irep_density() : calcule le nombre d'installations industrielles PM
                           dans un rayon autour de chaque station LCSQA
"""

import logging
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Distance Haversine ────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcule la distance en km entre deux points GPS (WGS84)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Extraction des métadonnées de stations ───────────────────────

def get_stations_lcsqa(df_lcsqa: pd.DataFrame) -> pd.DataFrame:
    """Extrait les métadonnées uniques des stations LCSQA."""
    return (
        df_lcsqa
        .groupby("code_station")[["lat", "lon", "nom_station", "type_station"]]
        .first()
        .reset_index()
    )


def get_stations_meteo(df_meteo: pd.DataFrame) -> pd.DataFrame:
    """Extrait les métadonnées uniques des stations météo."""
    return (
        df_meteo
        .groupby("code_station_meteo")[["lat_meteo", "lon_meteo", "nom_station_meteo"]]
        .first()
        .reset_index()
    )


# ── Mapping stations LCSQA → Météo ───────────────────────────────

def find_nearest_meteo_station(stations_lcsqa: pd.DataFrame,
                                stations_meteo: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque station LCSQA, trouve la station météo complète la plus proche
    via la distance Haversine.
    Retourne un DataFrame de mapping : code_station → code_station_meteo + distance_km.
    """
    mappings = []
    for _, row_lcsqa in stations_lcsqa.iterrows():
        distances = stations_meteo.apply(
            lambda row_meteo: haversine_km(
                row_lcsqa["lat"], row_lcsqa["lon"],
                row_meteo["lat_meteo"], row_meteo["lon_meteo"]
            ),
            axis=1
        )
        idx_min = distances.idxmin()
        nearest = stations_meteo.loc[idx_min]
        mappings.append({
            "code_station"      : row_lcsqa["code_station"],
            "code_station_meteo": nearest["code_station_meteo"],
            "nom_station_meteo" : nearest["nom_station_meteo"],
            "distance_km"       : distances[idx_min],
        })

    df_mapping = pd.DataFrame(mappings)
    logger.info("Mapping LCSQA → Météo :")
    for _, row in df_mapping.iterrows():
        logger.info(f"  {row['code_station']} → {row['nom_station_meteo']} ({row['distance_km']:.1f} km)")

    return df_mapping


# ── Jointure LCSQA ↔ Météo ────────────────────────────────────────

def join_meteo_to_lcsqa(df_lcsqa: pd.DataFrame,
                         df_meteo: pd.DataFrame) -> pd.DataFrame:
    """
    Associe les variables météo à chaque observation LCSQA via :
    1. Filtrage des stations météo complètes
       (température + vent + pluie + humidité >= 1000 mesures chacune)
    2. Mapping spatial : station LCSQA → station météo complète la plus proche
    3. Merge temporel exact sur l'heure

    Les NaN résiduels correspondent à des heures sans mesure météo
    (pannes, maintenance) — imputés dans le notebook de nettoyage.
    """
    # Étape 1 — Identifier les stations complètes sur les 4 variables clés
    def count_notna(col: str) -> pd.Index:
        return (
            df_meteo[df_meteo[col].notna()]
            .groupby("code_station_meteo")[col]
            .count()
        )

    stations_temp     = count_notna("temperature_c")
    stations_vent     = count_notna("vent_vitesse_ms")
    stations_pluie    = count_notna("pluie_mm")
    stations_humidite = count_notna("humidite_pct")

    seuil = 1000
    stations_completes = (
        stations_temp[stations_temp >= seuil].index
        .intersection(stations_vent[stations_vent >= seuil].index)
        .intersection(stations_pluie[stations_pluie >= seuil].index)
        .intersection(stations_humidite[stations_humidite >= seuil].index)
    )

    df_meteo_actif = df_meteo[
        df_meteo["code_station_meteo"].isin(stations_completes)
    ].copy()

    logger.info(
        f"Stations météo complètes (temp + vent + pluie + humidité) : "
        f"{len(stations_completes)} / {df_meteo['code_station_meteo'].nunique()}"
    )

    # Étape 2 — Mapping spatial sur stations complètes uniquement
    stations_lcsqa = get_stations_lcsqa(df_lcsqa)
    stations_meteo = get_stations_meteo(df_meteo_actif)
    mapping        = find_nearest_meteo_station(stations_lcsqa, stations_meteo)

    # Étape 3 — Enrichir LCSQA avec le code station météo associé
    df = df_lcsqa.merge(
        mapping[["code_station", "code_station_meteo", "distance_km"]],
        on="code_station",
        how="left"
    )

    # Étape 4 — Préparer météo : arrondir datetime à l'heure
    df_meteo_actif["datetime_meteo"] = df_meteo_actif["datetime_meteo"].dt.floor("h")

    # Colonnes météo à joindre
    cols_meteo = [
        "code_station_meteo", "datetime_meteo",
        "vent_vitesse_ms", "vent_direction_deg",
        "temperature_c", "humidite_pct", "pluie_mm",
    ]
    df_meteo_slim = df_meteo_actif[
        [c for c in cols_meteo if c in df_meteo_actif.columns]
    ].copy()

    # Étape 5 — Arrondir datetime LCSQA à l'heure pour le merge
    df["datetime_merge"] = df["datetime_debut"].dt.floor("h")

    # Étape 6 — Merge temporel exact sur l'heure
    df = df.merge(
        df_meteo_slim,
        left_on  = ["code_station_meteo", "datetime_merge"],
        right_on = ["code_station_meteo", "datetime_meteo"],
        how="left"
    )
    df = df.drop(columns=["datetime_merge", "datetime_meteo"], errors="ignore")

    # Taux de remplissage
    logger.info(f"Après jointure météo : {len(df)} lignes | {df['code_station'].nunique()} stations")
    for col in ["temperature_c", "vent_vitesse_ms", "humidite_pct", "pluie_mm"]:
        if col in df.columns:
            logger.info(f"  Taux remplissage {col} : {df[col].notna().mean():.1%}")

    return df


# ── Jointure LCSQA ↔ IREP ────────────────────────────────────────

def compute_irep_density(df_lcsqa: pd.DataFrame,
                          df_irep: pd.DataFrame,
                          rayon_km: float = 5.0) -> pd.DataFrame:
    """
    Pour chaque station LCSQA et chaque année, calcule le nombre
    d'installations industrielles PM dans un rayon donné.

    Note : densite_emission_pm_kg est calculée mais non utilisée car
    les émissions 2023-2024 sont toutes < seuil et 2025 non publiées.
    La variable retenue est nb_installations_5km.
    Les NaN pour 2025 sont une limite documentée des données sources.
    """
    # Filtrer IREP sur les polluants PM uniquement
    df_irep_pm = df_irep[
        df_irep["polluant_irep"].str.contains(
            "PM|Poussi|particule", case=False, na=False
        )
    ].copy()

    logger.info(
        f"IREP PM : {len(df_irep_pm)} lignes | "
        f"années : {sorted(df_irep_pm['annee'].unique())}"
    )

    stations_lcsqa = get_stations_lcsqa(df_lcsqa)

    # Calcul par station et par année
    results = []
    for _, station in stations_lcsqa.iterrows():
        for annee in df_irep_pm["annee"].unique():
            irep_annee = df_irep_pm[df_irep_pm["annee"] == annee].copy()

            # Distance de chaque installation à la station
            irep_annee["dist_km"] = irep_annee.apply(
                lambda row: haversine_km(
                    station["lat"], station["lon"],
                    row["lat_irep"], row["lon_irep"]
                ) if pd.notna(row["lat_irep"]) and pd.notna(row["lon_irep"])
                else np.inf,
                axis=1
            )

            dans_rayon = irep_annee[irep_annee["dist_km"] <= rayon_km]

            results.append({
                "code_station"          : station["code_station"],
                "annee"                 : annee,
                "densite_emission_pm_kg": dans_rayon["emission_kg"].sum(),
                "nb_installations_5km"  : dans_rayon["identifiant"].nunique(),
            })

    df_densite = pd.DataFrame(results)

    # Ajouter l'année aux observations LCSQA pour la jointure
    df_lcsqa = df_lcsqa.copy()
    df_lcsqa["annee"] = df_lcsqa["datetime_debut"].dt.year

    # Merge sur station + année (left : NaN pour 2025)
    df_lcsqa = df_lcsqa.merge(
        df_densite,
        on=["code_station", "annee"],
        how="left"
    )

    taux = df_lcsqa["nb_installations_5km"].notna().mean()
    logger.info(f"Après jointure IREP : {len(df_lcsqa)} lignes")
    logger.info(f"Taux de remplissage nb_installations_5km : {taux:.1%}")

    return df_lcsqa