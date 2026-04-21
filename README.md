# Projet Data Science — Prédiction de la qualité de l'air (PM2.5) en Île-de-France

## Problématique

Ce projet vise à prédire le dépassement du seuil réglementaire de PM2.5 (25 µg/m³) dans les 24 prochaines heures en Île-de-France, à partir de données historiques de pollution atmosphérique, de variables météorologiques et d'informations sur les sources d'émissions industrielles.

La variable cible est binaire : 1 si la concentration maximale de PM2.5 dépasse 25 µg/m³ dans les 24 heures suivantes, 0 sinon.

## Sources de données

| Source | Description | Accès |
|---|---|---|
| LCSQA / Geod'air | Concentrations PM2.5 horaires — 18 stations IDF | API (token gratuit) |
| Météo-France | Variables météo horaires — 8 départements IDF | CSV.gz (data.gouv.fr) |
| IREP / Géorisques | Émissions industrielles PM par installation IDF | ZIP (georisques.gouv.fr) |

## Période d'étude

la période générale est de 2021 à 2025

## Installation

```bash
# Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les dépendances
uv sync

# Configurer les variables d'environnement
cp .env.example .env
# Remplir .env avec vos clés (voir section Prérequis)
```

## Prérequis

Avant d'exécuter les notebooks, configurer le fichier `.env` :

```
GEODAIR_TOKEN=        # token API Geod'air — inscription sur geodair.fr
S3_ENDPOINT_URL=      # endpoint MinIO SSPCloud
S3_ACCESS_KEY=        # access key SSPCloud
S3_SECRET_KEY=        # secret key SSPCloud
S3_SESSION_TOKEN=     # session token SSPCloud
S3_BUCKET=            # nom du bucket 
```

## Lancement

Exécuter les notebooks dans l'ordre :

```bash
# Enregistrer le kernel Jupyter avec l'environnement du projet
uv run python -m ipykernel install --user --name projet-qualite-air --display-name "Python (projet-qualite-air)"

# Lancer Jupyter
uv run jupyter notebook
```

| Notebook | Description |
|---|---|
| `01_collecte.ipynb` | Collecte des données brutes et upload S3 |
| `02_nettoyage.ipynb` | Jointures spatiales, features, nettoyage, sauvegarde |
| `03_descriptif.ipynb` | Analyse exploratoire et visualisations |
| `04_modelisation.ipynb` | Entraînement, évaluation et interprétation des modèles |

## Structure du projet

```
projet-qualite-air/
├── README.md
├── pyproject.toml              
├── .env.example                 variables d'environnement (sans valeurs)
├── .gitignore
├── data/
│   ├── raw/                     données brutes (non versionnées)
│   └── processed/               données consolidées (non versionnées)
├── notebooks/
│   ├── 01_collecte.ipynb
│   ├── 02_nettoyage.ipynb
│   ├── 03_descriptif.ipynb
│   └── 04_modelisation.ipynb
└── src/
    ├── __init__.py
    ├── collect_lcsqa.py         collecte PM2.5 via API Geod'air
    ├── collect_meteo.py         collecte météo via data.gouv.fr
    ├── collect_irep.py          collecte émissions industrielles
    ├── data_loader.py           chargement des données depuis S3
    ├── spatial.py               jointures spatiales 
    ├── features.py              features temporelles, lags, variable cible
    └── models.py                entraînement et évaluation des modèles
```

## Pipeline

```
Collecte (01)          Nettoyage (02)             Modélisation (04)
─────────────          ──────────────             ─────────────────
LCSQA   ──┐                                       Régression logistique
Météo   ──┼──► Jointures spatiales ──► Features ──► Random Forest
IREP    ──┘    + Imputation NaN       + Cible        XGBoost / LightGBM
               + Sauvegarde S3                     + SHAP values
```

## Features du dataset consolidé

| Feature | Description | Source |
|---|---|---|
| `pm25_brute` | Concentration PM2.5 horaire (µg/m³) | LCSQA |
| `pm25_lag1h` | Concentration 1 heure avant | LCSQA |
| `pm25_lag6h` | Concentration 6 heures avant | LCSQA |
| `pm25_lag24h` | Concentration 24 heures avant | LCSQA |
| `pm25_roll24h` | Moyenne glissante 24h | LCSQA |
| `pm25_roll72h` | Moyenne glissante 72h | LCSQA |
| `temperature_c` | Température (°C) | Météo-France |
| `vent_vitesse_ms` | Vitesse du vent (m/s) | Météo-France |
| `vent_direction_deg` | Direction du vent (degrés) | Météo-France |
| `humidite_pct` | Humidité relative (%) | Météo-France |
| `pluie_mm` | Précipitations (mm) | Météo-France |
| `heure` | Heure de la journée (0-23) | Temporelle |
| `jour_semaine` | Jour de la semaine (0=lundi) | Temporelle |
| `mois` | Mois (1-12) | Temporelle |
| `saison` | Saison météorologique | Temporelle |
| `is_weekend` | 1 si samedi ou dimanche | Temporelle |
| `nb_installations_5km` | Nombre d'installations industrielles PM dans 5 km | IREP |
| `depasse_seuil_24h` | **Variable cible** — dépassement 25 µg/m³ / 24h | LCSQA |

## Statistiques du dataset

- **Lignes** : 715 350 observations horaires
- **Colonnes** : 26 features + variable cible
- **Stations** : 18 stations de mesure IDF
- **Taux de dépassement** : 18.7% (déséquilibre modéré)
- **Période** : janvier 2021 — décembre 2025



## Auteurs

- BODJONA Horace
- KONLAMBIGUE Youdan-yamin
- SENOU Delphin

## Citation des données

- **LCSQA / Geod'air** : Laboratoire Central de Surveillance de la Qualité de l'Air — INERIS
- **Météo-France** : données issues de meteo.data.gouv.fr, Licence Ouverte Etalab
- **IREP** : Géorisques, Ministère de la Transition Écologique
