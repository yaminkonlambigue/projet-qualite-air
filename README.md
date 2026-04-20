# Projet Data Science — Prédiction de la qualité de l'air (PM2.5)

## Problématique
l'Objectif de ce projet est de prédire le dépassement du seuil réglementaire de PM2.5 (25 µg/m³)
dans les 24 prochaines heures en Île-de-France.

## Sources de données
| Source        | Description                          | Accès        |
|---------------|--------------------------------------|--------------|
| LCSQA/Geod'air| Concentrations polluants horaires    | CSV / API    |
| Météo-France  | Variables météo horaires             | CSV          |
| CAMS          | Prévisions PM2.5 J+1 à J+4          | API (cdsapi) |
| CEREMA        | Trafic routier                       | CSV          |
| IREP          | Émissions industrielles              | CSV          |

## Période
- Entraînement : 2021, 2022, 2023
- Test : 2024 (hors JO 26 juillet–11 août) + 2025

## Installation
# Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les dépendances
uv sync

# Configurer les variables d'environnement
cp .env.example .env
# Remplir .env avec vos clés

## Lancement
# Collecte des données
uv run python src/data_loader.py

# Notebooks dans l'ordre
uv run jupyter notebook

## Structure
projet-qualite-air/
├── README.md
├── pyproject.toml         
├── .env.example            (variables d'environnement)
├── .gitignore
├── Makefile                (commandes raccourcies)
├── data/
│   ├── raw/                
│   └── processed/          
├── notebooks/
│   ├── 01_collecte.ipynb
│   ├── 02_nettoyage.ipynb
│   ├── 03_descriptif.ipynb
│   └── 04_modelisation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py      (collecte des données)
│   ├── features.py         (construction des features)
│   ├── spatial.py          (jointures spatiales)
│   ├── models.py           (entraînement et évaluation)
│   └── utils.py            (fonctions utilitaires partagées)
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    └── test_features.py

## Reproductibilité
Toutes les cellules des notebooks doivent s'exécuter sans erreur
de haut en bas après un `uv sync` et la configuration du `.env`.

## Auteurs
- Prénom Nom
- Prénom Nom

## Citation des données
- LCSQA/Geod'air : Laboratoire Central de Surveillance de la Qualité de l'Air
- Météo-France : données issues de meteo.data.gouv.fr
- CAMS : Copernicus Atmosphere Monitoring Service, ECMWF
- CEREMA / Bison Futé : Ministère de la Transition Écologique
- IREP : Géorisques, Ministère de la Transition Écologique