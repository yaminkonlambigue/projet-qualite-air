import pandas as pd
import matplotlib.pyplot as plt


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

