"""
Construction des features pour la modélisation.
- add_temporal_features() : variables temporelles issues de datetime
- add_lags()              : lags et moyennes glissantes de PM2.5 par station
- add_target()            : variable cible (dépassement seuil 24h) par station
"""

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Seuil réglementaire PM2.5 en µg/m³
SEUIL_PM25 = 25.0


#  Variables temporelles 

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les variables temporelles depuis datetime_debut.
    Ces variables encodent les patterns cycliques connus :
    saisonnalité, effet weekend, rush hour.
    """
    df = df.copy()

    df["heure"]        = df["datetime_debut"].dt.hour
    df["jour_semaine"] = df["datetime_debut"].dt.dayofweek  # 0=lundi, 6=dimanche
    df["mois"]         = df["datetime_debut"].dt.month
    df["annee"]        = df["datetime_debut"].dt.year
    df["is_weekend"]   = (df["jour_semaine"] >= 5).astype(int)
    return df

    # Saison météorologique
    def get_saison(date) -> str:
        mois = date.month
        jour = date.day
        if (mois == 12 and jour >= 21) or (mois in [1, 2]) or (mois == 3 and jour <= 20):
            return "hiver"
        if (mois == 3 and jour >= 21) or (mois in [4, 5]) or (mois == 6 and jour <= 20):
            return "printemps"
        if (mois == 6 and jour >= 21) or (mois in [7, 8]) or (mois == 9 and jour <= 22):
            return "ete"
        return "automne"

    df["saison"] = df["datetime_debut"].apply(get_saison)


#  Lags de PM2.5 

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les lags et moyennes glissantes de PM2.5 par station.

    Les lags sont calculés dans l'ordre chronologique strict par station
    pour éviter tout mélange entre stations ou toute fuite temporelle.

    Variables créées :
    - pm25_lag1h    : concentration 1 heure avant
    - pm25_lag6h    : concentration 6 heures avant
    - pm25_lag24h   : concentration 24 heures avant
    - pm25_roll24h  : moyenne glissante sur les 24 dernières heures
    - pm25_roll72h  : moyenne glissante sur les 72 dernières heures
    """
    df = df.copy()

    # Tri chronologique strict par station — indispensable pour les lags
    df = df.sort_values(["code_station", "datetime_debut"]).reset_index(drop=True)

    # Calcul des lags par station
    grp = df.groupby("code_station")["pm25_brute"]

    df["pm25_lag1h"]   = grp.shift(1)
    df["pm25_lag6h"]   = grp.shift(6)
    df["pm25_lag24h"]  = grp.shift(24)
    df["pm25_roll24h"] = grp.transform(lambda x: x.shift(1).rolling(24, min_periods=12).mean())
    df["pm25_roll72h"] = grp.transform(lambda x: x.shift(1).rolling(72, min_periods=36).mean())

    logger.info(
        f"Lags PM2.5 ajoutés : lag1h, lag6h, lag24h, roll24h, roll72h"
    )
    logger.info(
        f"Taux de remplissage lags : "
        f"lag1h={df['pm25_lag1h'].notna().mean():.1%} | "
        f"lag24h={df['pm25_lag24h'].notna().mean():.1%} | "
        f"roll24h={df['pm25_roll24h'].notna().mean():.1%}"
    )
    return df


# Variable cible

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la variable cible : dépassement du seuil PM2.5 (25 µg/m³)
    dans les 24 heures suivantes.

    Pour chaque observation (station, heure t), on regarde la concentration
    maximale de PM2.5 entre t+1 et t+24. Si ce maximum dépasse le seuil,
    la cible vaut 1, sinon 0.

    Les dernières 24 heures de chaque station auront une cible NaN
    (pas assez d'observations futures) — elles seront exclues du modèle.
    """
    df = df.copy()

    # Tri chronologique strict par station
    df = df.sort_values(["code_station", "datetime_debut"]).reset_index(drop=True)

    # Maximum de PM2.5 sur les 24 heures suivantes (shift négatif = futur)
    df["pm25_max_24h"] = (
        df.groupby("code_station")["pm25_brute"]
        .transform(lambda x: x.shift(-1).rolling(24, min_periods=1).max())
    )

    # Binarisation
    df["depasse_seuil_24h"] = (df["pm25_max_24h"] > SEUIL_PM25).astype("Int64")

    # Mettre NaN là où pm25_max_24h est NaN (fin de série)
    df.loc[df["pm25_max_24h"].isna(), "depasse_seuil_24h"] = pd.NA

    # Statistiques
    taux_depassement = df["depasse_seuil_24h"].mean()
    logger.info(f"Variable cible ajoutée : depasse_seuil_24h (seuil = {SEUIL_PM25} µg/m³)")
    logger.info(f"Taux de dépassement global : {taux_depassement:.1%}")
    logger.info(
        f"Distribution : "
        f"0 (pas de dépassement) = {(df['depasse_seuil_24h'] == 0).sum()} | "
        f"1 (dépassement) = {(df['depasse_seuil_24h'] == 1).sum()} | "
        f"NaN = {df['depasse_seuil_24h'].isna().sum()}"
    )
    return df



def imputer_par_fenetre(df: pd.DataFrame,
                         colonnes: list[str],
                         fenetre: int = 6) -> pd.DataFrame:
    """
    Impute les valeurs manquantes par interpolation linéaire locale
    sur une fenêtre de `fenetre` observations de chaque côté, par station.
    """
    df = df.copy()
    df = df.sort_values(["code_station", "datetime_debut"]).reset_index(drop=True)

    for col in colonnes:
        if col not in df.columns:
            logger.warning(f"Colonne {col} absente — skip")
            continue

        df[col] = (
            df.groupby("code_station")[col]
            .transform(lambda x: x.interpolate(
                method="linear",
                limit=fenetre,
                limit_direction="both"
            ))
        )
        taux = df[col].isna().mean()
        logger.info(f"Imputation {col} : NaN résiduels = {taux:.1%}")

    return df