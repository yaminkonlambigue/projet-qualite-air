import pandas as pd
import matplotlib.pyplot as plt
from great_tables import GT, loc, style
import seaborn as sns


# ============================================================================
# 1. ANALYSE DESCRITPIVE DE LA VARIABLE CIBLE (PM2.5)
# ============================================================================


def tableau_stats_descriptives(df, colonne='pm25_µg_m3'):
    """
    Génère un tableau complet de statistiques descriptives pour PM2.5.
    Affiche : moyenne, médiane, écart-type, min, max, déciles et quartiles.
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        colonne (str): Nom de la colonne à analyser (défaut: 'pm25_µg_m3')
    Returns:
        pd.DataFrame: Tableau des statistiques
    Exemple:
        >>> stats = tableau_stats_descriptives(df)
        >>> print(stats)
    """
    stats_dict = {
        'Moyenne': df[colonne].mean(),
        'Médiane': df[colonne].median(),
        'Écart-type': df[colonne].std(),
        'Min': df[colonne].min(),
        'Q1 (25%)': df[colonne].quantile(0.25),
        'Q2 (50%)': df[colonne].quantile(0.50),
        'Q3 (75%)': df[colonne].quantile(0.75),
        'Max': df[colonne].max(),
        'D1 (10%)': df[colonne].quantile(0.10),
        'D9 (90%)': df[colonne].quantile(0.90),
    }
    stats_df = pd.DataFrame(list(stats_dict.items()),
                            columns=['Statistique', 'Valeur (µg/m³)'])
    stats_df['Valeur (µg/m³)'] = stats_df['Valeur (µg/m³)'].round(2)
    gt = (GT(stats_df).tab_header(
          title="STATISTIQUES DESCRIPTIVES - PM2.5",
          subtitle="Analyse descriptive complète"
          )
          .fmt_number(
          columns=['Valeur (µg/m³)'],
          decimals=2
          )
          .tab_options(
          container_width="600px"
          )
          )
    return gt.show()


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


