"""Customer segmentation with k-means on log-transformed RFM values."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

RANDOM_STATE = 42


def build_features(rfm: pd.DataFrame) -> pd.DataFrame:
    """Log-transform recency, frequency and monetary to reduce their skew.

    Monetary can be zero or slightly negative after returns, so it is clipped
    at 0 and log1p is used. Recency and frequency are always at least 1.
    """
    return pd.DataFrame({
        "log_recency": np.log(rfm["recency"]),
        "log_frequency": np.log(rfm["frequency"]),
        "log_monetary": np.log1p(rfm["monetary"].clip(lower=0)),
    }, index=rfm.index)


def fit_kmeans(X: np.ndarray, k: int, seed: int = RANDOM_STATE) -> KMeans:
    """K-means with several initialisations, keeping the best one."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def evaluate_k(X: np.ndarray, k_values) -> pd.DataFrame:
    """Inertia and silhouette of k-means, and BIC of a Gaussian mixture, for each k."""
    rows = []
    for k in k_values:
        km = fit_kmeans(X, k)
        gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                              random_state=RANDOM_STATE).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
            "gmm_bic": gmm.bic(X),
        })
    return pd.DataFrame(rows).set_index("k")


def stability(X: np.ndarray, k: int, seeds=range(5)) -> float:
    """Lowest adjusted Rand index between the first run and runs with other seeds.

    1 means every seed finds exactly the same clusters (up to relabelling).
    """
    runs = [fit_kmeans(X, k, seed=s).labels_ for s in seeds]
    return min(adjusted_rand_score(runs[0], other) for other in runs[1:])


def order_by_value(labels: np.ndarray, monetary: pd.Series) -> np.ndarray:
    """Relabel clusters 0..k-1 from the lowest to the highest median monetary value.

    K-means numbers clusters arbitrarily; a fixed order makes it possible to
    attach business names to them.
    """
    medians = monetary.groupby(labels).median().sort_values()
    mapping = {old: new for new, old in enumerate(medians.index)}
    return np.array([mapping[label] for label in labels])


def profile_clusters(rfm: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Size, typical RFM values and revenue share of each cluster."""
    summary = (rfm
               .groupby(clusters, observed=False)
               .agg(customers=("recency", "size"),
                    median_recency=("recency", "median"),
                    median_frequency=("frequency", "median"),
                    median_monetary=("monetary", "median"),
                    revenue=("monetary", "sum")))
    summary["share_of_customers"] = summary["customers"] / summary["customers"].sum()
    summary["share_of_revenue"] = summary["revenue"] / summary["revenue"].sum()
    return summary.drop(columns="revenue")
