
"""Statistics tools"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def generate_chi2_samples(n_samples, degrees_of_freedom):
    """
    Génère des échantillons de distribution chi2.
    Parameters:
    -----------
    n_samples : int
        Nombre d'échantillons à générer
    degrees_of_freedom : int
        Nombre de degrés de liberté        
    Returns:
    --------
    samples : array
        Échantillons de la distribution chi2
    mean_value : float
        Moyenne théorique (= degrés de liberté)
    """
    samples = np.random.chisquare(degrees_of_freedom, n_samples)
    mean_value = degrees_of_freedom  # Moyenne théorique du chi2
    return samples, mean_value

#--------------------------------------------------------------------
def generate_lognormal_samples(n_samples, mu=0, sigma=1):
    """
    Génère des échantillons de distribution log-normale. 
    Parameters:
    -----------
    n_samples : int
        Nombre d'échantillons à générer
    mu : float
        Paramètre mu de la log-normale
    sigma : float
        Paramètre sigma de la log-normale      
    Returns:
    --------
    samples : array
        Échantillons de la distribution log-normale
    mean_value : float
        Moyenne de l'échantillon
    """
    samples = np.random.lognormal(mu, sigma, n_samples)
    mean_value = np.mean(samples)
    return samples, mean_value

#---------------------------------------------------------------
def create_log_bins(n_bins, min_bound, max_bound):
    """
    Crée des bins espacés logarithmiquement.   
    Parameters:
    -----------
    n_bins : int
        Nombre de bins
    min_bound : float
        Borne minimale
    max_bound : float
        Borne maximale
        
    Returns:
    --------
    bins : array
        Edges des bins en échelle log
    """
    return np.logspace(np.log10(min_bound), np.log10(max_bound), n_bins + 1)

#---------------------------------------------------------------------

def plot_normalized_histogram(samples, mean_value, n_bins=50, min_bound=None, max_bound=None, 
                               title="Distribution Chi2 normalisée",fillcolor="r"):
    """
    Trace l'histogramme des valeurs normalisées par la moyenne avec bins logarithmiques.   
    Parameters:
    -----------
    samples : array
        Échantillons de la distribution
    mean_value : float
        Valeur moyenne pour la normalisation
    n_bins : int
        Nombre de bins (default: 50)
    min_bound : float, optional
        Borne minimale (calculée automatiquement si None)
    max_bound : float, optional
        Borne maximale (calculée automatiquement si None)
    title : str
        Titre du graphique
    """
    # Normalisation par la moyenne
    normalized_samples = samples/mean_value
    
    # Calcul automatique des bornes si non fournies
    if min_bound is None:
        min_bound = max(normalized_samples.min() * 0.5, 1e-3)  # Éviter 0
    if max_bound is None:
        max_bound = normalized_samples.max() * 1.5
    
    # Création des bins logarithmiques
    bins = create_log_bins(n_bins, min_bound, max_bound)
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogramme
    counts, edges, patches = ax.hist(normalized_samples, bins=bins, 
                                      alpha=0.7, edgecolor='black', 
                                      facecolor=fillcolor, density=True,
                                      label=f'N échantillons = {len(samples)}')
    
    # Échelle logarithmique sur l'axe x
    ax.set_xscale('log')
    
    # Labels et titre
    ax.set_xlabel('Valeur normalisée (chi2 / moyenne)', fontsize=12)
    ax.set_ylabel('Fréquence', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

#------------------------------------------------------------------------------
def ks_test_chi2_vs_lognormal(data, verbose=True):
    """
    Effectue un test de Kolmogorov-Smirnov pour comparer les données
    à une distribution chi2 et log-normale.
    
    Parameters:
    -----------
    data : array
        Données à tester
    verbose : bool
        Si True, affiche les résultats détaillés
        
    Returns:
    --------
    results : dict
        Dictionnaire contenant les résultats des tests:
        - 'chi2': {'statistic', 'pvalue', 'df'}
        - 'lognormal': {'statistic', 'pvalue', 'mu', 'sigma'}
        - 'best_fit': 'chi2' ou 'lognormal'
    """
    results = {}
    
    # ===== Test pour Chi2 =====
    # Estimation du paramètre (degrés de liberté) par la méthode des moments
    # Pour chi2: moyenne = df
    df_estimate = np.mean(data)
    
    # Test KS pour chi2
    ks_stat_chi2, p_value_chi2 = stats.kstest(data, 
                                               lambda x: stats.chi2.cdf(x, df_estimate))
    
    results['chi2'] = {
        'statistic': ks_stat_chi2,
        'pvalue': p_value_chi2,
        'df': df_estimate
    }
    
    # ===== Test pour Log-Normale =====
    # Estimation des paramètres par maximum de vraisemblance
    # Pour log-normale: on utilise le log des données
    log_data = np.log(data[data > 0])  # Éviter les valeurs négatives ou nulles
    mu_estimate = np.mean(log_data)
    sigma_estimate = np.std(log_data, ddof=1)
    
    # Test KS pour log-normale
    ks_stat_lognorm, p_value_lognorm = stats.kstest(data,
                                                     lambda x: stats.lognorm.cdf(x, 
                                                                                  s=sigma_estimate,
                                                                                  scale=np.exp(mu_estimate)))
    
    results['lognormal'] = {
        'statistic': ks_stat_lognorm,
        'pvalue': p_value_lognorm,
        'mu': mu_estimate,
        'sigma': sigma_estimate
    }
    
    # Déterminer le meilleur ajustement
    # On privilégie la p-value la plus élevée (ou la statistique KS la plus faible)
    if p_value_chi2 > p_value_lognorm:
        results['best_fit'] = 'chi2'
    else:
        results['best_fit'] = 'lognormal'
    
    if verbose:
        print("="*70)
        print("TEST DE KOLMOGOROV-SMIRNOV")
        print("="*70)
        print(f"\nNombre de données: {len(data)}")
        print(f"Moyenne: {np.mean(data):.4f}")
        print(f"Écart-type: {np.std(data):.4f}")
        
        print("\n" + "-"*70)
        print("CHI2 DISTRIBUTION")
        print("-"*70)
        print(f"Degrés de liberté estimés: {df_estimate:.4f}")
        print(f"Statistique KS: {ks_stat_chi2:.6f}")
        print(f"P-value: {p_value_chi2:.6f}")
        if p_value_chi2 > 0.05:
            print("✓ On ne peut PAS rejeter l'hypothèse chi2 (p > 0.05)")
        else:
            print("✗ On rejette l'hypothèse chi2 (p < 0.05)")
        
        print("\n" + "-"*70)
        print("LOG-NORMALE DISTRIBUTION")
        print("-"*70)
        print(f"Paramètre mu estimé: {mu_estimate:.4f}")
        print(f"Paramètre sigma estimé: {sigma_estimate:.4f}")
        print(f"Statistique KS: {ks_stat_lognorm:.6f}")
        print(f"P-value: {p_value_lognorm:.6f}")
        if p_value_lognorm > 0.05:
            print("✓ On ne peut PAS rejeter l'hypothèse log-normale (p > 0.05)")
        else:
            print("✗ On rejette l'hypothèse log-normale (p < 0.05)")
        
        print("\n" + "="*70)
        print(f"MEILLEUR AJUSTEMENT: {results['best_fit'].upper()}")
        print("="*70)
        print("\nNote: Plus la p-value est élevée, meilleur est l'ajustement.")
        print("Une p-value > 0.05 suggère que les données sont cohérentes avec la distribution.")
    
    return results
