import pandas as pd
import matplotlib.pyplot as plt
from great_tables import GT, loc, style
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# 1. ANALYSE DESCRITPIVE DE LA VARIABLE CIBLE (PM2.5)
# ============================================================================


def graphique_frequence_depassement(df, colonne="depassement_seuil"):
    """
    Tableau de fréquence binaire: Dépassement vs Non-dépassement du seuil.
    Affiche le nombre et le pourcentage de cas où PM2.5 > 25 µg/m³.
    Args:
        df (pd.DataFrame): DataFrame contenant la colonne 'depassement_seuil'
    Returns:
        pd.DataFrame: Tableau de fréquence
    Exemple:
        >>> freq = tableau_frequence_depassement(df)
    """
    freq = df[colonne].value_counts().sort_index()
    pct = (freq / len(df) * 100).round(1)
    # Création du diagramme
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Pas de dépassement\n(≤ 25)", "Dépassement\n(> 25)"]
    values = [freq[0], freq[1]]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for i, (bar, val, p) in enumerate(zip(bars, values, pct)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{p}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Nombre d\'observations', fontsize=12, fontweight='bold')
    ax.set_title('FRÉQUENCE DES DÉPASSEMENTS DE SEUIL PM2.5', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return plt.show()


def graphique_top_bottom_stations(df, n=5, colonne="pm25", station="station"):
    """
    Classement des stations par concentration moyenne de PM2.5.
    Affiche les 5 stations les plus polluées et les 5 moins polluées.
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'station' et 'pm25'
        n (int): Nombre de stations à afficher (défaut: 5)
    Returns:
        tuple: (top_n, bottom_n) DataFrames
    Exemple:
        >>> top, bottom = tableau_top_bottom_stations(df, n=5)
    """
    station_avg = df.groupby(station)[colonne].agg(['mean', 'std', 'count']).round(2)
    station_avg = station_avg.sort_values('mean', ascending=False)

    top_n = station_avg.head(n)
    bottom_n = station_avg.tail(n)
    combined = pd.concat([top_n, bottom_n.iloc[::-1]])
    combined = combined.reset_index()
    combined = combined.sort_values('mean', ascending=False)
    colors = ['#e74c3c'] * len(top_n) + ['#2ecc71'] * len(bottom_n)

    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(range(len(combined)), combined['mean'], color=colors, edgecolor='black', linewidth=1.5)

    # Ajouter les valeurs
    for i, (bar, val) in enumerate(zip(bars, combined['mean'])):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined[station])
    ax.set_xlabel('Concentration moyenne (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title(f'STATIONS POLLUÉES (TOP {n}) vs PROPRES (BOTTOM {n})', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return plt.show()


def graphique_histogramme_pm25(df, colonne="pm25", bins=30, figsize=(12, 6)):
    """
    Histogramme de la distribution de PM2.5.
    Visualise la distribution avec une courbe de densité superposée.
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'pm25_µg_m3'
        bins (int): Nombre de bacs (défaut: 30)
        figsize (tuple): Dimensions de la figure (défaut: 12x6)
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_histogramme_pm25(df)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[colonne], bins=bins, color='steelblue', edgecolor='black', 
            alpha=0.7, density=True, label='Fréquence')
    # Ajouter la courbe de densité
    df[colonne].plot(kind='density', ax=ax, color='red', linewidth=2, label='Densité')
    # Seuil réglementaire
    ax.axvline(x=25, color='orange', linestyle='--', linewidth=2, label='Seuil alerte (25 µg/m³)')
    ax.set_xlabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Densité', fontsize=12, fontweight='bold')
    ax.set_title('Distribution de PM2.5', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def graphique_boxplot_global(df, colonne="pm25", figsize=(12, 6)):
    """
    Boxplot (boîte à moustaches) global de PM2.5.
    Identifie visuellement les valeurs aberrantes et la distribution.
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'pm25_µg_m3'
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_boxplot_global(df)
    """
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(df[colonne], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Seuil alerte')
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Boxplot - PM2.5 (Global)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig, ax


# ============================================================================
# 2. ANALYSE TEMPORELLE (SAISONNALITÉ ET CYCLES)
# ============================================================================


def tableau_moyennes_mensuelles(df, colonne="pm25", mois="mois"):
    """
    Tableau des concentrations moyennes de PM2.5 par mois.
    Identifie les pics saisonniers (ex: hiver vs été).
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'mois' et 'pm25_µg_m3'
    Returns:
        pd.DataFrame: Moyennes par mois
    Exemple:
        >>> monthly = tableau_moyennes_mensuelles(df)
    """
    monthly = df.groupby(mois)[colonne].agg(['mean', 'std', 'min', 'max']).round(2)
    monthly.index = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                     'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'][:len(monthly)]
    monthly.columns = ['Moyenne', 'Écart-type', 'Min', 'Max']
    monthly_reset = monthly.reset_index().rename(columns={'index': 'Mois'})

    gt = (
        GT(monthly_reset)
        .tab_header(
            title="Statistiques PM2.5 par Mois",
            subtitle="Moyennes, écarts-types, minima et maxima"
        )
        .tab_options(
            column_labels_font_weight="bold",
            table_font_size="12px"
        )
        .fmt_number(
            columns=['Moyenne', 'Écart-type', 'Min', 'Max'],
            decimals=2
        )
        .cols_align(align="center", columns=['Moyenne', 'Écart-type', 'Min', 'Max'])
    )
    return gt.show()


def graphique_moyennes_mensuelles(df, colonne="pm25", mois="mois"):
    """
    Graphique en barres des concentrations moyennes de PM2.5 par mois.
    Visualise les pics saisonniers (ex: hiver vs été).
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'mois' et colonne spécifiée
        colonne (str): Colonne à analyser (défaut: "pm25")
        mois (str): Colonne contenant les mois (défaut: "mois")
    Returns:
        matplotlib.figure.Figure: Figure du graphique
    Exemple:
        >>> fig = graphique_moyennes_mensuelles(df)
        >>> plt.show()
    """
    # Calcul des moyennes
    monthly = df.groupby(mois)[colonne].agg('mean').round(2)
    # Mapping des mois
    mois_labels = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                   'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
    monthly.index = mois_labels[:len(monthly)]
    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    # Couleurs : bleu pour hiver, rouge pour été
    colors = ['#FF6B6B' if i in [0, 1, 2, 11] else '#4ECDC4' 
              for i in range(len(monthly))]
    bars = ax.bar(monthly.index, monthly.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Mise en forme
    ax.set_xlabel('Mois', fontsize=12, fontweight='bold')
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Concentrations moyennes de PM2.5 par mois', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, monthly.max() * 1.15)
    plt.tight_layout()
    return fig


def tableau_moyennes_jour_semaine(df, colonne="pm25", jour="jour_semaine"):
    """
    Tableau des concentrations moyennes par jour de la semaine.
    Compare l'effet semaine vs weekend sur la pollution.
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'jour_semaine' et 'pm25_µg_m3'
    Returns:
        pd.DataFrame: Moyennes par jour
    Exemple:
        >>> daily = tableau_moyennes_jour_semaine(df)
    """
    daily = df.groupby(jour)[colonne].agg(['mean', 'std', 'count']).round(2)
    daily.index = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][:len(daily)]
    daily.columns = ['Moyenne', 'Écart-type', 'Nombre obs']
    daily_reset = daily.reset_index().rename(columns={'index': 'Jour'})

    gt = (
        GT(daily_reset)
        .tab_header(
            title="Statistiques PM2.5 par Jour de la Semaine",
            subtitle="Moyennes, écarts-types et nombre d'observations"
        )
            .tab_options(
                column_labels_font_weight="bold",
                table_font_size="12px"
        )
        .fmt_number(
            columns=['Moyenne', 'Écart-type'],
            decimals=2
        )
        .fmt_integer(
            columns=['Nombre obs']
        )
        .cols_align(align="center")
        .opt_row_striping()
    )
    return gt.show()


def graphique_serie_temporelle(df, colonne="pm25", date="date", figsize=(14, 6)):
    """
    Série temporelle: évolution du PM2.5 sur la période complète.
    Montre les tendances et variations temporelles.
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'date' et 'pm25_µg_m3'
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_serie_temporelle(df)
    """
    fig, ax = plt.subplots(figsize=figsize)
    df_sorted = df.sort_values(date)
    ax.plot(df_sorted[date], df_sorted[colonne], linewidth=1.5, color='steelblue')
    ax.fill_between(df_sorted[date], df_sorted[colonne], alpha=0.3, color='steelblue')
    ax.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Seuil alerte')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Série Temporelle - PM2.5', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax


def graphique_cycle_diurne(df, colonne="pm25", heure="heure", figsize=(12, 6)):
    """
    Profil horaire moyen: concentration de PM2.5 par heure du jour
    Identifie les pics de pollution liés au trafic (7-9h, 17-19h).
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'heure' et 'pm25_µg_m3'
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_cycle_diurne(df)
    """
    fig, ax = plt.subplots(figsize=figsize)
    hourly = df.groupby(heure)[colonne].mean()
    ax.plot(hourly.index, hourly.values, marker='o', linewidth=2, 
            markersize=8, color='steelblue')
    ax.fill_between(hourly.index, hourly.values, alpha=0.3, color='steelblue')
    ax.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Seuil alerte')
    # Mettre en évidence les heures de pointe
    ax.axvspan(7, 9, alpha=0.1, color='orange', label='Rush matin')
    ax.axvspan(17, 19, alpha=0.1, color='orange', label='Rush soir')
    ax.set_xlabel('Heure du jour', fontsize=12, fontweight='bold')
    ax.set_ylabel('PM2.5 moyen (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Cycle Diurne - PM2.5 (Profil horaire moyen)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def graphique_boxplot_mensuel(df, colonne="pm25", mois="mois", figsize=(14, 6)):
    """
    Boxplots mensuels: variation de PM2.5 selon les mois.
    Montre la saisonnalité et la variabilité.
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        colonne (str): Colonne à analyser (défaut: "pm25_µg_m3")
        mois (str): Colonne contenant les mois (défaut: "mois")
        seuil_alerte (float): Seuil d'alerte à afficher (défaut: 25)
        figsize (tuple): Dimensions de la figure (défaut: (14, 6))
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_boxplot_mensuel(df)
        >>> plt.show()
    """
    month_labels = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                    'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
    # Récupérer les mois présents dans les données
    months_in_data = sorted(df[mois].unique())
    # Créer le graphique
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=mois, y=colonne, ax=ax, palette='Set2')
    # Ajouter le seuil d'alerte
    ax.axhline(y=25, color='red', linestyle='--', linewidth=2, 
               label=f'Seuil alerte ({25} µg/m³)')
    # Configurer les labels des mois
    ax.set_xticklabels([month_labels[m-1] for m in months_in_data])
    # Mise en forme
    ax.set_xlabel('Mois', fontsize=12, fontweight='bold')
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Boxplots Mensuels - PM2.5', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig, ax


def graphique_boxplot_jour_semaine(df, colonne="pm25", jour_semaine="jour_semaine", figsize=(12, 6)):
    """
    Boxplots par jour de la semaine.
    Compare la pollution entre semaine et weekend.
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'jour_semaine' et 'pm25_µg_m3'
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_boxplot_jour_semaine(df)
    """
    fig, ax = plt.subplots(figsize=figsize)
    day_labels = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    # Colorer différemment semaine et weekend
    colors = ['lightblue']*5 + ['lightcoral']*2
    sns.boxplot(data=df, x=jour_semaine, y=colonne, ax=ax, palette=colors)
    ax.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Seuil alerte')
    ax.set_xticklabels(day_labels)
    ax.set_xlabel('Jour de la semaine', fontsize=12, fontweight='bold')
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Boxplots par Jour de la Semaine - PM2.5', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig, ax


def graphique_heatmap_heure_jour(df, colonne="pm25", heure="heure", jour_semaine="jour_semaine", figsize=(14, 8)):
    """
    Heatmap: concentration de PM2.5 par heure et jour de la semaine.
    Identifie visuellement les moments critiques.
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'heure', 'jour_semaine', 'pm25_µg_m3'
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_heatmap_heure_jour(df)
    """
    fig, ax = plt.subplots(figsize=figsize)
    pivot_data = df.pivot_table(values=colonne, 
                                 index=jour_semaine, 
                                 columns=heure, 
                                 aggfunc='mean')
    day_labels = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    pivot_data.index = [day_labels[i] for i in pivot_data.index]
    sns.heatmap(pivot_data, cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'PM2.5 (µg/m³)'})
    ax.set_xlabel('Heure du jour', fontsize=12, fontweight='bold')
    ax.set_ylabel('Jour de la semaine', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap - PM2.5 (Heure & Jour de la semaine)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, ax


def graphique_moyennes_saisonnieres(df, colonne="pm25", col_saison="saison"):
    """
    Graphique en barres des concentrations moyennes de PM2.5 par saison.
    Args:
        df (pd.DataFrame): DataFrame avec colonne spécifiée
        colonne (str): Colonne à analyser (ex: "pm25_brute")
        col_saison (str): Colonne contenant les saisons (ex: "saison")
    """
    # 1. Calcul des moyennes par saison
    seasonal = df.groupby(col_saison)[colonne].agg('mean').round(2)
    # 2. Définition de l'ordre logique et des couleurs spécifiques
    ordre_saisons = ['hiver', 'printemps', 'ete', 'automne']
    # On ne garde que les saisons présentes dans le dataframe
    saisons_presentes = [s for s in ordre_saisons if s in seasonal.index]
    seasonal = seasonal.reindex(saisons_presentes)
    # Mapping des couleurs par saison
    color_map = {
        'Hiver': '#4A90E2',     # Bleu froid
        'Printemps': '#7ED321', # Vert bourgeon
        'Été': '#F5A623',       # Jaune/Orange soleil
        'Automne': '#8B4513'    # Marron feuilles mortes
    }
    colors = [color_map.get(s, '#808080') for s in seasonal.index]
    # 3. Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(seasonal.index, seasonal.values, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    # 4. Mise en forme
    ax.set_xlabel('Saison', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Moyenne {colonne} (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title(f'Concentrations moyennes de {colonne} par saison', 
                 fontsize=14, fontweight='bold', pad=20) 
    ax.grid(axis='y', alpha=0.3, linestyle='--') 
    # Ajuster la limite Y pour laisser de la place au texte
    if not seasonal.empty:
        ax.set_ylim(0, seasonal.max() * 1.2)    
    plt.tight_layout()
    return fig

# ============================================================================
# 3. ANALYSE SPATIALE (COMPARAISON ENTRE STATIONS)
# ============================================================================


def tableau_comparaison_stations(df, station="station", colonne="pm25", seuil="depassement_seuil", date="date", industrie="nombre_industrie"):
    """
    Tableau comparatif des stations: moyenne PM2.5, taux dépassement, densité.
    Compare les caractéristiques de pollution entre villes.
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'station', 'pm25_µg_m3', etc.
        station (str): Colonne contenant les noms de station
        colonne (str): Colonne à analyser (défaut: "pm25")
        seuil (str): Colonne du dépassement de seuil
        date (str): Colonne de date pour compter les observations
        industrie (str): Colonne du nombre d'industries
    Returns:
        GT: Objet great_tables GT
    Exemple:
        >>> gt = tableau_comparaison_stations(df)
        >>> gt.show()
    """
    station_comp = df.groupby(station).agg({
        colonne: ['mean', 'std'],
        seuil: 'mean',
        date: 'count'
    }).round(2)
    station_comp.columns = ['Moy PM2.5', 'Sd PM2.5', 'Taux dépassement', 'Nb obs']
    station_comp['Taux dépassement %'] = (station_comp['Taux dépassement'] * 100).round(1)
    station_comp = station_comp.drop('Taux dépassement', axis=1)
    station_comp = station_comp.sort_values('Moy PM2.5', ascending=False)
    station_comp_reset = station_comp.reset_index().rename(columns={station: 'Station'})
    gt = (
        GT(station_comp_reset)
        .tab_header(
            title="Tableau Comparatif par Station",
            subtitle="Moyennes PM2.5, écarts-types, taux de dépassement et nombre d'observations"
        )
        .tab_options(
            column_labels_font_weight="bold",
            table_font_size="12px"
        )
        .fmt_number(
            columns=['Moy PM2.5', 'SD PM2.5'],
            decimals=2
        )
        .fmt_number(
            columns=['Taux dépassement %'],
            decimals=1
        )
        .fmt_integer(
            columns=['Nb obs']
        )
        .cols_align(align="center", columns=['Moy PM2.5', 'SD PM2.5', 'Taux dépassement %', 'Nb obs'])
        .opt_row_striping()
        # En-tête coloré
        .tab_style(
            style=style.fill(color="#2E86AB"),
            locations=loc.header()
        )
        .tab_style(
            style=style.text(color="white", weight="bold"),
            locations=loc.header()
        )
        .tab_style(
            style=style.fill(color="#FFE5E5"),
            locations=loc.body(
                rows=[i for i, val in enumerate(station_comp_reset['Taux dépassement %']) if val > 20]
            )
        )
    )
    return gt.show()


def graphique_boxplot_stations(df, station="station", colonne="pm25", figsize=(14, 6)):
    """
    Boxplots par station: distribution de PM2.5 par ville.
    Compare visuellement les différences entre stations.
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'station' et 'pm25_µg_m3'
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_boxplot_stations(df)
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=station, y=colonne, ax=ax, palette='Set2')
    ax.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Seuil alerte')
    ax.set_xlabel('Station', fontsize=12, fontweight='bold')
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Boxplots par Station - PM2.5', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax


def graphique_bar_depassements_stations(df, station="station", seuil="depassement_seuil", figsize=(14, 6)):
    """
    Bar chart: nombre cumulé de jours de dépassement par station.
    Identifie les stations les plus problématiques.
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'station' et 'depassement_seuil'
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_bar_depassements_stations(df)
    """
    fig, ax = plt.subplots(figsize=figsize)
    exceeds = df.groupby(station)[seuil].sum().sort_values(ascending=False)
    bars = ax.bar(exceeds.index, exceeds.values, color='red', alpha=0.7, edgecolor='black')
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel('Station', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de dépassements', fontsize=12, fontweight='bold')
    ax.set_title('Nombre Cumulé de Dépassements par Station', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax


def tracer_comparaison_carto(df, annee_cible=2025, col_station="code_station", colonne="pm25", 
    col_industrie="nb_installations_5km", col_lat="lat", col_lon="lon", col_annee="annee"):
    
    # 1. Filtrage et Agrégation
    df_filtered = df[df[col_annee] == annee_cible].copy()
    df_agg = df_filtered.groupby([col_station, col_lat, col_lon]).agg({
        colonne: 'mean',
        col_industrie: 'mean'
    }).reset_index()

    # 2. Création des subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{"type": "mapbox"}, {"type": "mapbox"}]],
        subplot_titles=(
            f"Moyenne {colonne} ({annee_cible})",
            f"Densité {col_industrie} ({annee_cible})"
        )
    )

    # 3. Carte 1 : Points PM2.5 (Scattermapbox)
    fig.add_trace(
        go.Scattermapbox(
            lat=df_agg[col_lat],
            lon=df_agg[col_lon],
            mode='markers',
            marker=dict(
                size=14, # Taille du point fixe et net
                color=df_agg[colonne], # La couleur dépend de la valeur
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.45, title=colonne)
            ),
            customdata=df_agg[col_station],
            hovertemplate="<b>Station:</b> %{customdata}<br><b>Valeur:</b> %{marker.color:.2f}<extra></extra>",
        ),
        row=1, col=1
    )

    # 4. Carte 2 : Points Industries (Scattermapbox)
    fig.add_trace(
        go.Scattermapbox(
            lat=df_agg[col_lat],
            lon=df_agg[col_lon],
            mode='markers',
            marker=dict(
                size=14,
                color=df_agg[col_industrie],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(x=1.0, title="Industries")
            ),
            customdata=df_agg[col_station],
            hovertemplate="<b>Station:</b> %{customdata}<br><b>Industries:</b> %{marker.color:.0f}<extra></extra>",
        ),
        row=1, col=2
    )

    # 5. Mise en page
    center_lat = df_agg[col_lat].mean()
    center_lon = df_agg[col_lon].mean()

    fig.update_layout(
        mapbox1=dict(
            style='carto-positron',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=9
        ),
        mapbox2=dict(
            style='carto-positron',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=9
        ),
        margin={"r": 10, "t": 80, "l": 10, "b": 10},
        height=600,
        title_text=f"Analyse Spatiale : {colonne} vs {col_industrie} - {annee_cible}",
        showlegend=False
    )
    
    return fig.show()

# ============================================================================
# 4. ANALYSE DES CORRÉLATIONS
# ============================================================================


def graphique_heatmap_correlation(df, colonnes_numeriques=None, method='spearman', figsize=(12, 10)):
    """
    Heatmap de la matrice de corrélation.
    Visualise les corrélations avec code couleur.
    Args:
        df (pd.DataFrame): DataFrame
        colonnes_numeriques (list): Colonnes à corréler
        method (str): Méthode de corrélation ('pearson' ou 'spearman')
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_heatmap_correlation(df)
        >>> fig, ax = graphique_heatmap_correlation(df, method='pearson')
    """
    if colonnes_numeriques is None:
        colonnes_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=figsize)
    corr_matrix = df[colonnes_numeriques].corr(method=method)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Corrélation'})
    ax.set_title(f'Heatmap - Matrice de Corrélation ({method.capitalize()})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, ax


def graphique_scatter_pm25(df, colonne_x="vitesse_vent_ms", colonne_y="pm25", 
                            label_x=None, label_y=None, figsize=(12, 6)):
    """
    Scatter plot: PM2.5 vs une variable quelconque. 
    Args:
        df (pd.DataFrame): DataFrame
        colonne_x (str): Colonne pour l'axe X (défaut: "vitesse_vent_ms")
        colonne_y (str): Colonne pour l'axe Y (défaut: "pm25_µg_m3")
        label_x (str): Label personnalisé pour l'axe X (optionnel)
        label_y (str): Label personnalisé pour l'axe Y (optionnel)
        figsize (tuple): Dimensions de la figure 
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_scatter_pm25(df)
        >>> fig, ax = graphique_scatter_pm25(df, colonne_x='temperature', colonne_y='pm25_µg_m3')
        >>> fig, ax = graphique_scatter_pm25(df, colonne_x='humidite', colonne_y='pm25_µg_m3',
        ...                                   label_x='Humidité (%)', label_y='PM2.5 (µg/m³)')
    """
    # Vérifier que les colonnes existent
    if colonne_x not in df.columns or colonne_y not in df.columns:
        raise ValueError(f"Colonnes invalides. Colonnes disponibles: {df.columns.tolist()}")
    # Supprimer les NaN
    df_clean = df[[colonne_x, colonne_y]].dropna()
    fig, ax = plt.subplots(figsize=figsize)
    # Scatter plot
    ax.scatter(df_clean[colonne_x], df_clean[colonne_y], alpha=0.5, s=30, color='steelblue')
    # Corrélation (Pearson par défaut)
    corr = df_clean[colonne_x].corr(df_clean[colonne_y], method="spearman")
    # Labels
    xlabel = label_x if label_x else colonne_x
    ylabel = label_y if label_y else colonne_y
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'{ylabel} vs {xlabel} (r = {corr:.3f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def histogramme_densité_superposée(df, colonne, groupe='depasse_seuil_24h', figsize=(12, 6), bins=30):
    """
    Crée un seul histogramme avec les deux groupes superposés et leurs courbes de densité.
    Args:
        df (pd.DataFrame): DataFrame
        colonne (str): Colonne quantitative à analyser
        groupe (str): Colonne binaire (défaut: 'depasse_seuil_24h')
        figsize (tuple): Dimensions de la figure
        bins (int): Nombre de bins
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = histogramme_densité_superposée(df, 'pm25_brute')
    """
    df_clean = df[[colonne, groupe]].dropna()
    groupe_0 = df_clean[df_clean[groupe] == 0][colonne]
    groupe_1 = df_clean[df_clean[groupe] == 1][colonne]
    fig, ax = plt.subplots(figsize=figsize)
    # Histogrammes
    ax.hist(groupe_0, bins=bins, color='#2E86AB', alpha=0.5, 
            edgecolor='black', density=True, label='Sans dépassement (0)')
    ax.hist(groupe_1, bins=bins, color='#A23B72', alpha=0.5, 
            edgecolor='black', density=True, label='Avec dépassement (1)')
    # Courbes de densité
    groupe_0.plot(kind='density', ax=ax, color='#2E86AB', linewidth=2.5)
    groupe_1.plot(kind='density', ax=ax, color='#A23B72', linewidth=2.5)
    ax.set_xlabel(colonne, fontsize=12, fontweight='bold')
    ax.set_ylabel('Densité', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution de {colonne} par dépassement de seuil', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig, ax

# ============================================================================
# 5. ANALYSE DE PERSISTANCE (AUTO-CORRÉLATION)
# ============================================================================


def graphique_acf_pm25(df, colonne="pm25", date="date", nlags=48, figsize=(14, 6)):
    """
    ACF (Autocorrelation Function): corrélation de PM2.5 avec lui-même.
    Montre jusqu'à combien d'heures la pollution passée explique présente.
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'pm25_µg_m3'
        nlags (int): Nombre de retards à afficher (défaut: 48 heures)
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_acf_pm25(df, nlags=48)
    """
    fig, ax = plt.subplots(figsize=figsize)
    pm25_series = df.sort_values(date)[colonne].values
    plot_acf(pm25_series, lags=nlags, ax=ax)
    ax.set_xlabel('Retard (heures)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ACF', fontsize=12, fontweight='bold')
    ax.set_title('ACF - Autocorrélation de PM2.5', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def graphique_pacf_pm25(df, colonne="pm25", date="date", nlags=48, figsize=(14, 6)):
    """
    PACF (Partial Autocorrelation Function): corrélation partielle.
    Isolates l'impact direct de chaque retard sans influence des autres.
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'pm25_µg_m3'
        nlags (int): Nombre de retards à afficher (défaut: 48)
        figsize (tuple): Dimensions de la figure
    Returns:
        fig, ax: Objet figure et axes matplotlib
    Exemple:
        >>> fig, ax = graphique_pacf_pm25(df, nlags=48)
    """
    fig, ax = plt.subplots(figsize=figsize)
    pm25_series = df.sort_values(date)[colonne].values
    plot_pacf(pm25_series, lags=nlags, ax=ax)
    ax.set_xlabel('Retard (heures)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PACF', fontsize=12, fontweight='bold')
    ax.set_title('PACF - Autocorrélation Partielle de PM2.5', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax
