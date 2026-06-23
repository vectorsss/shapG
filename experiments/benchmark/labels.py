"""Display labels for plot legends.

Maps config explainer keys to nicer math-text labels.  Unknown keys fall back
to the key itself, so this is purely cosmetic.
"""

DISPLAY_LABELS = {
    # ShapG runs per graph, so its label carries a " | <graph>" suffix.
    "shapg | kendalltau": "ShapG",
    "shapg | rank_deletion_cosine": r"ShapG$_{\cos}$",
    "shapg | rank_deletion_kendalltau": r"ShapG$_{\tau}$",
    "shapg | rank_deletion_mutual_info": r"ShapG$_{MI}$",
    "cis": "CIS",
    # Compressed-sensing estimators
    "randomcs": "RandomCS",
    "qrcs": "QR-CS",
    "block_qrcs": "BlockQR-CS",
    "improved_qrcs": "ImprovedQR-CS",
    "improved_block_qrcs": "ImprovedBlockQR-CS",
    # Stratified sweep: {shapley_weighted, leverage, leverage_bernoulli}
    # x {direct estimation, CS reconstruction}
    "stratified_sw_direct": r"Strat$_{SW}^{dir}$",
    "stratified_sw_cs": r"Strat$_{SW}^{cs}$",
    "stratified_lev_direct": r"Strat$_{lev}^{dir}$",
    "stratified_lev_cs": r"Strat$_{lev}^{cs}$",
    "stratified_levb_direct": r"Strat$_{levB}^{dir}$",
    "stratified_levb_cs": r"Strat$_{levB}^{cs}$",
    # Leverage-score and multilinear estimators
    "leverage": "LeverageSHAP",
    "multilinear_naive": r"ML$_{naive}$",
    "multilinear_sw": r"ML$_{SW}$",
    "multilinear_ic": r"ML$_{IC}$",
    "multilinear_bernoulli": r"ML$_{bern}$",
    "exact": "Exact",
    # Official shap-library baselines
    "kernel_shap": "KernelSHAP",
    "sampling_shap": "SamplingSHAP",
    # Forward-KPI reference curves
    "Greedy Best": "Greedy Best",
}

# Methods to exclude from KPI / efficiency plots
SKIP_METHODS = {"Random", "Greedy Worst"}


def display(key):
    """Return the display label for a data key."""
    return DISPLAY_LABELS.get(key, key)
