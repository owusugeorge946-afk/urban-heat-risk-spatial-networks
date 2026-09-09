"""
02_COLAB_NETWORK_ANALYSIS.py
Spatial Networks of Urban Surface Heat Risk
Complete post-GEE network workflow reconstructed from the manuscript.

Inputs (one per city):
    Accra_heat_network_nodes_500m_2024.geojson
    Lagos_heat_network_nodes_500m_2024.geojson
    Abidjan_heat_network_nodes_500m_2024.geojson
    Freetown_heat_network_nodes_500m_2024.geojson

Methods implemented:
- city-wise z-score standardisation of node attributes
- queen candidate adjacency
- Euclidean environmental distance
- Gaussian similarity weights, sigma_x = median candidate-pair distance
- baseline edge threshold >= 0.05
- topology and centrality
- Leiden on largest connected component, 100 initialisations
- HRI propagation alpha=0.60, tol=1e-6, max_iter=100
- alpha/edge-threshold/rook sensitivity
- fixed-topology HRI-exclusion sensitivity
"""

# In Google Colab, run once before this script:
# !pip -q install geopandas libpysal networkx python-igraph leidenalg \
#     scipy scikit-learn pyproj openpyxl

from pathlib import Path
import math, warnings, json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import coo_matrix, diags
from libpysal.weights import Queen, Rook
import igraph as ig
import leidenalg

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
USE_GOOGLE_DRIVE = True
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/URBAN_HEAT_NETWORK_2024")
else:
    ROOT = Path("/content")

OUTPUT = ROOT / "REPRODUCIBILITY_RESULTS"
OUTPUT.mkdir(parents=True, exist_ok=True)

CITIES = ["Accra", "Lagos", "Abidjan", "Freetown"]
CITY_EPSG = {
    "Accra": "EPSG:32630",
    "Lagos": "EPSG:32631",
    "Abidjan": "EPSG:32630",
    "Freetown": "EPSG:32628",
}
FEATURES_BASELINE = [
    "lst", "ndvi", "ndbi", "hri",
    "built_pct", "vegetation_pct", "water_pct", "bare_pct", "elevation",
]
FEATURES_NO_HRI = [
    "lst", "ndvi", "ndbi",
    "built_pct", "vegetation_pct", "water_pct", "bare_pct", "elevation",
]
MIN_WEIGHT = 0.05
BASELINE_ALPHA = 0.60
TOL = 1e-6
MAX_ITER = 100
EPS = 1e-12
LEIDEN_REPEATS = 100
LEIDEN_SEED_START = 1000
HIGH_RISK_THRESHOLD = 0.60
TOP_FRACTION = 0.10

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def read_nodes(city):
    path = ROOT / f"{city}_heat_network_nodes_500m_2024.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Export it from 01_GEE_PREPROCESS_CLASSIFY_EXPORT.js."
        )
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(CITY_EPSG[city]).reset_index(drop=True)

    required = FEATURES_BASELINE + ["geometry"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"{city}: missing required columns: {missing}")

    for c in FEATURES_BASELINE:
        gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
    gdf = gdf.dropna(subset=FEATURES_BASELINE + ["geometry"]).copy().reset_index(drop=True)

    # Preserve cell_id when available; otherwise use deterministic sequential IDs.
    if "cell_id" in gdf.columns:
        gdf["node_id"] = gdf["cell_id"].astype(str)
    else:
        gdf["node_id"] = [str(i) for i in range(len(gdf))]

    if "longitude" not in gdf.columns or "latitude" not in gdf.columns:
        cent = gdf.to_crs("EPSG:4326").geometry.centroid
        gdf["longitude"] = cent.x
        gdf["latitude"] = cent.y
    return gdf

def contiguity_pairs(gdf, kind="queen"):
    ids = list(range(len(gdf)))
    work = gdf.copy()
    work["_row"] = ids
    if kind.lower() == "queen":
        W = Queen.from_dataframe(work, ids=ids, silence_warnings=True)
    elif kind.lower() == "rook":
        W = Rook.from_dataframe(work, ids=ids, silence_warnings=True)
    else:
        raise ValueError("kind must be 'queen' or 'rook'")
    return sorted({
        tuple(sorted((int(i), int(j))))
        for i, neighbours in W.neighbors.items()
        for j in neighbours if int(i) != int(j)
    })

def standardised_matrix(gdf, features):
    usable = []
    for c in features:
        values = pd.to_numeric(gdf[c], errors="coerce").to_numpy(float)
        if np.isfinite(values).all() and np.nanstd(values) > 1e-10:
            usable.append(c)
    if not usable:
        raise ValueError("No nonconstant similarity attributes remain.")
    z = StandardScaler().fit_transform(gdf[usable].to_numpy(float))
    return z, usable

def candidate_distances(gdf, pairs, features):
    z, usable = standardised_matrix(gdf, features)
    delta = np.array([np.linalg.norm(z[i] - z[j]) for i, j in pairs], dtype=float)
    positive = delta[delta > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    return delta, sigma, usable

def weighted_edges(gdf, neighbourhood="queen", min_weight=MIN_WEIGHT,
                   features=FEATURES_BASELINE, fixed_pairs=None,
                   fixed_sigma=None):
    pairs = fixed_pairs if fixed_pairs is not None else contiguity_pairs(gdf, neighbourhood)
    delta, sigma_calc, usable = candidate_distances(gdf, pairs, features)
    sigma = float(fixed_sigma if fixed_sigma is not None else sigma_calc)
    if sigma <= 0:
        sigma = 1.0
    weights = np.exp(-(delta ** 2) / (2.0 * sigma ** 2))
    records = []
    for (i, j), d, w in zip(pairs, delta, weights):
        if w >= min_weight:
            records.append((i, j, float(w), float(d)))
    return records, sigma, usable, pairs

def graph_from_edges(gdf, edges):
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))
    for i, j, w, d in edges:
        G.add_edge(i, j, weight=w, distance=1.0/(w + EPS),
                   environmental_distance=d)
    return G

def topology_summary(city, G, candidate_count, retained_count, sigma):
    n = G.number_of_nodes()
    components = list(nx.connected_components(G))
    largest = max(components, key=len) if components else set()
    isolates = list(nx.isolates(G))
    return {
        "City": city,
        "Nodes": n,
        "Candidate edges": candidate_count,
        "Retained edges": retained_count,
        "Retained edges (%)": 100.0 * retained_count / candidate_count if candidate_count else np.nan,
        "Density": nx.density(G),
        "Mean degree": np.mean([d for _, d in G.degree()]) if n else np.nan,
        "Mean strength": np.mean([d for _, d in G.degree(weight="weight")]) if n else np.nan,
        "Components": len(components),
        "Largest component nodes": len(largest),
        "Largest component (%)": 100.0 * len(largest) / n if n else np.nan,
        "Isolated nodes": len(isolates),
        "Isolated nodes (%)": 100.0 * len(isolates) / n if n else np.nan,
        "sigma_x": sigma,
    }

def centralities(G):
    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))

    # Weighted shortest paths: stronger similarity -> shorter effective distance.
    # Exact betweenness on ~15k nodes can be expensive; manuscript-associated
    # analysis used a reproducible approximation for large graphs.
    n = G.number_of_nodes()
    k = min(750, n)
    bet = nx.betweenness_centrality(
        G, k=k if k < n else None, normalized=True,
        weight="distance", seed=42
    )

    # Closeness and eigenvector are evaluated on the largest connected component.
    comps = list(nx.connected_components(G))
    lcc = max(comps, key=len) if comps else set()
    L = G.subgraph(lcc).copy()

    if len(L) > 0:
        close_lcc = nx.closeness_centrality(L, distance="distance")
        try:
            eig_lcc = nx.eigenvector_centrality_numpy(L, weight="weight")
        except Exception:
            eig_lcc = nx.eigenvector_centrality(
                L, weight="weight", max_iter=5000, tol=1e-10
            )
    else:
        close_lcc, eig_lcc = {}, {}

    closeness = {i: close_lcc.get(i, np.nan) for i in G.nodes()}
    eigenvector = {i: eig_lcc.get(i, np.nan) for i in G.nodes()}

    return degree, strength, bet, closeness, eigenvector, lcc

def exact_top_decile(frame, value_col, tie_secondary=None):
    n_select = int(math.ceil(len(frame) * TOP_FRACTION))
    sort_cols = [value_col]
    ascending = [False]
    if tie_secondary and tie_secondary in frame.columns:
        sort_cols.append(tie_secondary)
        ascending.append(False)
    sort_cols.append("_row")
    ascending.append(True)
    ranked = frame.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    selected = set(ranked.head(n_select)["_row"])
    return frame["_row"].isin(selected)

def leiden_best(G, lcc_nodes, repeats=LEIDEN_REPEATS):
    nodes = sorted(lcc_nodes)
    idx = {node: k for k, node in enumerate(nodes)}
    L = G.subgraph(nodes)
    edges = [(idx[u], idx[v]) for u, v in L.edges()]
    graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    graph.es["weight"] = [float(L[u][v]["weight"]) for u, v in L.edges()]

    best_quality = -np.inf
    best_membership = None
    best_seed = None

    for run in range(repeats):
        seed = LEIDEN_SEED_START + run
        partition = leidenalg.find_partition(
            graph,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            seed=seed,
        )
        q = float(partition.quality())
        if q > best_quality:
            best_quality = q
            best_membership = list(partition.membership)
            best_seed = seed

    membership = {node: int(best_membership[idx[node]]) for node in nodes}
    return membership, best_quality, best_seed

def transition_matrix_sparse(G):
    """Build a row-normalised sparse transition matrix without dense NxN arrays."""
    nodes = sorted(G.nodes())
    position = {node: i for i, node in enumerate(nodes)}
    rows, cols, data = [], [], []

    for u, v, attrs in G.edges(data=True):
        i, j = position[u], position[v]
        w = float(attrs.get("weight", 1.0))
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])

    n = len(nodes)
    if data:
        matrix = coo_matrix(
            (np.asarray(data, dtype=float),
             (np.asarray(rows, dtype=np.int32),
              np.asarray(cols, dtype=np.int32))),
            shape=(n, n)
        ).tocsr()
    else:
        matrix = coo_matrix((n, n), dtype=float).tocsr()

    strength = np.asarray(matrix.sum(axis=1)).ravel()
    isolated = strength <= EPS
    inverse_strength = np.divide(
        1.0, strength,
        out=np.zeros_like(strength, dtype=float),
        where=strength > EPS
    )
    transition = diags(inverse_strength) @ matrix
    return nodes, transition.tocsr(), isolated


def propagate(G, h0, alpha=BASELINE_ALPHA, tol=TOL, max_iter=MAX_ITER):
    nodes, P, isolated = transition_matrix_sparse(G)
    original = np.array([h0[i] for i in nodes], dtype=float)
    current = original.copy()
    residual = np.nan
    converged = False

    for iteration in range(1, max_iter + 1):
        updated = alpha * original + (1.0-alpha) * P.dot(current)
        updated[isolated] = original[isolated]
        residual = float(np.linalg.norm(updated-current, ord=2))
        current = updated
        if residual < tol:
            converged = True
            break

    return dict(zip(nodes, current)), iteration, residual, converged

def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a-b)**2)))

def jaccard_percent(a, b):
    a, b = set(a), set(b)
    union = a | b
    return 100.0 * len(a & b) / len(union) if union else 100.0

def top_decile_set(values):
    s = pd.Series(values)
    n = int(math.ceil(len(s) * TOP_FRACTION))
    return set(s.sort_values(ascending=False, kind="mergesort").head(n).index)

def compare_vectors(reference, candidate):
    reference = pd.Series(reference).sort_index()
    candidate = pd.Series(candidate).reindex(reference.index)
    ref = reference.to_numpy(float)
    alt = candidate.to_numpy(float)
    high_ref = ref >= HIGH_RISK_THRESHOLD
    high_alt = alt >= HIGH_RISK_THRESHOLD
    high_agreement = 100.0 * np.mean(high_ref == high_alt)
    top_ref = top_decile_set(reference)
    top_alt = top_decile_set(candidate)
    return {
        "RMSE": rmse(ref, alt),
        "Pearson r": float(pearsonr(ref, alt).statistic),
        "Spearman rho": float(spearmanr(ref, alt).statistic),
        "High-risk agreement (%)": high_agreement,
        "Top-decile Jaccard (%)": jaccard_percent(top_ref, top_alt),
    }

# -----------------------------------------------------------------------------
# BASELINE ANALYSIS
# -----------------------------------------------------------------------------
all_city = {}
topology_rows = []
community_summary_rows = []
propagation_rows = []

for city in CITIES:
    print("\n" + "="*80)
    print("PROCESSING", city.upper())
    print("="*80)

    gdf = read_nodes(city)
    gdf["_row"] = np.arange(len(gdf), dtype=int)

    edges, sigma, used_features, candidate_pairs = weighted_edges(gdf)
    G = graph_from_edges(gdf, edges)

    topology_rows.append(
        topology_summary(city, G, len(candidate_pairs), len(edges), sigma)
    )

    degree, strength, bet, close, eig, lcc = centralities(G)
    gdf["degree"] = gdf["_row"].map(degree)
    gdf["strength"] = gdf["_row"].map(strength)
    gdf["betweenness"] = gdf["_row"].map(bet)
    gdf["closeness"] = gdf["_row"].map(close)
    gdf["eigenvector"] = gdf["_row"].map(eig)

    # Exact top-decile selections. Degree ties resolved with strength.
    gdf["top10_degree"] = exact_top_decile(gdf, "degree", "strength")
    for col in ["strength", "betweenness", "closeness", "eigenvector"]:
        eligible = gdf.dropna(subset=[col]).copy()
        mask = exact_top_decile(eligible, col)
        gdf["top10_" + col] = False
        gdf.loc[eligible.index, "top10_" + col] = mask.values

    membership, modularity, best_seed = leiden_best(G, lcc)
    gdf["community"] = gdf["_row"].map(membership)

    counts = pd.Series(membership).value_counts().sort_index()
    for comm, count in counts.items():
        subset = gdf[gdf["community"] == comm]
        community_summary_rows.append({
            "City": city,
            "Community": int(comm),
            "Nodes": int(count),
            "Mean HRI": subset["hri"].mean(),
            "Mean LST": subset["lst"].mean(),
            "Mean NDVI": subset["ndvi"].mean(),
            "Mean NDBI": subset["ndbi"].mean(),
            "Mean built_pct": subset["built_pct"].mean(),
            "Mean vegetation_pct": subset["vegetation_pct"].mean(),
            "Mean water_pct": subset["water_pct"].mean(),
            "Mean bare_pct": subset["bare_pct"].mean(),
            "Mean degree": subset["degree"].mean(),
            "Mean strength": subset["strength"].mean(),
            "Mean betweenness": subset["betweenness"].mean(),
        })

    h0 = dict(zip(gdf["_row"], gdf["hri"].astype(float)))
    propagated, iterations, residual, converged = propagate(G, h0)
    gdf["propagated_hri"] = gdf["_row"].map(propagated)
    gdf["propagation_change"] = gdf["propagated_hri"] - gdf["hri"]

    base_cmp = compare_vectors(h0, propagated)
    propagation_rows.append({
        "City": city,
        "Original mean HRI": gdf["hri"].mean(),
        "Original SD": gdf["hri"].std(ddof=0),
        "Propagated mean HRI": gdf["propagated_hri"].mean(),
        "Propagated SD": gdf["propagated_hri"].std(ddof=0),
        "Mean change": gdf["propagation_change"].mean(),
        "Mean absolute change": gdf["propagation_change"].abs().mean(),
        "RMSE": base_cmp["RMSE"],
        "Pearson r": base_cmp["Pearson r"],
        "Spearman rho": base_cmp["Spearman rho"],
        "Iterations": iterations,
        "Final residual": residual,
        "Converged": converged,
        "Leiden communities": len(counts),
        "Modularity": modularity,
        "Best Leiden seed": best_seed,
        "LCC nodes": len(lcc),
    })

    edge_df = pd.DataFrame(edges, columns=[
        "source_row","target_row","weight","environmental_distance"
    ])
    gdf.to_file(OUTPUT / f"{city}_network_nodes_with_results.gpkg",
                layer="nodes", driver="GPKG")
    gdf.drop(columns="geometry").to_csv(
        OUTPUT / f"{city}_network_nodes_with_results.csv", index=False
    )
    edge_df.to_csv(OUTPUT / f"{city}_network_edges.csv", index=False)

    all_city[city] = {
        "gdf": gdf, "G": G, "sigma": sigma,
        "candidate_pairs": candidate_pairs,
        "baseline_edges": edges,
        "baseline_propagated": propagated,
        "h0": h0,
        "lcc": lcc,
    }

pd.DataFrame(topology_rows).to_csv(OUTPUT/"Table_1_network_topology.csv", index=False)
pd.DataFrame(propagation_rows).to_csv(OUTPUT/"Table_3_propagation_summary.csv", index=False)
pd.DataFrame(community_summary_rows).to_csv(
    OUTPUT/"Table_S3_community_characteristics.csv", index=False
)

# -----------------------------------------------------------------------------
# SENSITIVITY: alpha, threshold, neighbourhood
# -----------------------------------------------------------------------------
scenarios = [
    ("A040_Q_W005", 0.40, "queen", 0.05),
    ("BASELINE",    0.60, "queen", 0.05),
    ("A080_Q_W005", 0.80, "queen", 0.05),
    ("A060_Q_W010", 0.60, "queen", 0.10),
    ("A060_Q_W020", 0.60, "queen", 0.20),
    ("A060_R_W005", 0.60, "rook",  0.05),
]

sensitivity_rows = []
for city in CITIES:
    obj = all_city[city]
    gdf, h0 = obj["gdf"], obj["h0"]
    baseline = obj["baseline_propagated"]

    for scenario_id, alpha, neighbourhood, threshold in scenarios:
        edges, sigma, used, pairs = weighted_edges(
            gdf,
            neighbourhood=neighbourhood,
            min_weight=threshold,
            features=FEATURES_BASELINE,
        )
        Gs = graph_from_edges(gdf, edges)
        prop, iters, residual, converged = propagate(Gs, h0, alpha=alpha)
        cmp = compare_vectors(baseline, prop)
        sensitivity_rows.append({
            "City": city,
            "Scenario ID": scenario_id,
            "Alpha": alpha,
            "Neighbourhood": neighbourhood.title(),
            "Minimum weight": threshold,
            "Edges": len(edges),
            "Isolates": nx.number_of_isolates(Gs),
            "RMSE from baseline": cmp["RMSE"],
            "Spearman rho with baseline": cmp["Spearman rho"],
            "High-risk agreement (%)": cmp["High-risk agreement (%)"],
            "Top-decile Jaccard (%)": cmp["Top-decile Jaccard (%)"],
            "Iterations": iters,
            "Converged": converged,
        })

pd.DataFrame(sensitivity_rows).to_csv(
    OUTPUT/"Table_S4_sensitivity.csv", index=False
)

# -----------------------------------------------------------------------------
# FIXED-TOPOLOGY HRI-EXCLUSION SENSITIVITY
# -----------------------------------------------------------------------------
hri_exclusion_rows = []
for city in CITIES:
    obj = all_city[city]
    gdf = obj["gdf"]
    baseline_edges = obj["baseline_edges"]
    baseline_pairs = [(i,j) for i,j,_,_ in baseline_edges]
    fixed_sigma = obj["sigma"]

    # Recalculate weights using the exact retained baseline edge set, while
    # holding the baseline city-specific bandwidth constant.
    alt_edges, _, used, _ = weighted_edges(
        gdf,
        neighbourhood="queen",
        min_weight=0.0,
        features=FEATURES_NO_HRI,
        fixed_pairs=baseline_pairs,
        fixed_sigma=fixed_sigma,
    )
    # Topology remains fixed: do not re-threshold.
    Galt = graph_from_edges(gdf, alt_edges)
    alt_prop, iters, residual, converged = propagate(
        Galt, obj["h0"], alpha=BASELINE_ALPHA
    )
    cmp = compare_vectors(obj["baseline_propagated"], alt_prop)

    base_w = np.array([w for _,_,w,_ in baseline_edges], float)
    alt_w = np.array([w for _,_,w,_ in alt_edges], float)
    hri_exclusion_rows.append({
        "City": city,
        "Fixed edges": len(baseline_edges),
        "Edge-weight MAE": float(np.mean(np.abs(alt_w-base_w))),
        "Edge-weight RMSE": rmse(base_w, alt_w),
        "Edge-weight Spearman rho": float(spearmanr(base_w, alt_w).statistic),
        "Propagated-HRI MAE": float(np.mean(np.abs(
            pd.Series(alt_prop).sort_index().to_numpy() -
            pd.Series(obj["baseline_propagated"]).sort_index().to_numpy()
        ))),
        "Propagated-HRI RMSE": cmp["RMSE"],
        "Propagated-HRI Pearson r": cmp["Pearson r"],
        "Propagated-HRI Spearman rho": cmp["Spearman rho"],
        "High-risk agreement (%)": cmp["High-risk agreement (%)"],
        "Top-decile Jaccard (%)": cmp["Top-decile Jaccard (%)"],
        "Iterations": iters,
        "Converged": converged,
    })

pd.DataFrame(hri_exclusion_rows).to_csv(
    OUTPUT/"Table_S6_HRI_excluded_fixed_topology.csv", index=False
)

# -----------------------------------------------------------------------------
# MANUSCRIPT BENCHMARK CHECK
# -----------------------------------------------------------------------------
expected = {
    "Accra":    {"nodes":14841, "candidate":58153, "retained":50160},
    "Lagos":    {"nodes":15112, "candidate":59064, "retained":47848},
    "Abidjan":  {"nodes":8702,  "candidate":34046, "retained":28142},
    "Freetown": {"nodes":393,   "candidate":1396,  "retained":1262},
}

audit = []
topology_df = pd.DataFrame(topology_rows).set_index("City")
for city in CITIES:
    row = topology_df.loc[city]
    exp = expected[city]
    audit.append({
        "City": city,
        "Observed nodes": int(row["Nodes"]),
        "Expected nodes": exp["nodes"],
        "Nodes match": int(row["Nodes"]) == exp["nodes"],
        "Observed candidate edges": int(row["Candidate edges"]),
        "Expected candidate edges": exp["candidate"],
        "Candidate edges match": int(row["Candidate edges"]) == exp["candidate"],
        "Observed retained edges": int(row["Retained edges"]),
        "Expected retained edges": exp["retained"],
        "Retained edges match": int(row["Retained edges"]) == exp["retained"],
    })

pd.DataFrame(audit).to_csv(OUTPUT/"MANUSCRIPT_BENCHMARK_AUDIT.csv", index=False)

print("\nDONE. Results written to:", OUTPUT)
print(pd.DataFrame(audit).to_string(index=False))
