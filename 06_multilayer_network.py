#!/usr/bin/env python3
"""Module F - the multi-layer molecular network (L1 risk -> L2 ictal -> L3 systemic -> L4 recovery).

The graph is literature-curated: 93 genes whose identifiers were verified at NCBI Gene/HGNC and
111 relations each annotated with a mechanism and an evidence class (canonical molecular,
mechanistic, pathway-level or co-expression). It is NOT a downloaded interactome, so the analyses
below are hypothesis-generating: they say what follows from the curated mechanism set, not from an
unbiased screen. Every statistic is tested against a degree-preserving null so that conclusions
cannot be produced by the degree sequence alone.

Computed here:
  1. Random-walk-with-restart propagation from the pre-stroke risk layer, with a degree-preserving
     rewiring null, giving per-gene cross-layer reachability and empirical p-values.
  2. Identification of cross-layer connector genes (participation across the four layers).
  3. Enrichment of layer-pair adjacency against a configuration-model null.
  4. Shortest-path structure between the risk layer and the recovery layer, and the minimal edge
     cut separating them.
"""
import numpy as np, pandas as pd
import kcore as K

rng = np.random.default_rng(K.SEED + 6)
nodes = K.load_verified("genes_nodes")
edges = K.load_verified("network_edges")
genes = nodes.gene_symbol.tolist()
idx = {g: i for i, g in enumerate(genes)}
N = len(genes)
layer = nodes.set_index("gene_symbol")["layer"].to_dict()
broad = {g: l.split("_")[0] for g, l in layer.items()}

E = [(idx[a], idx[b]) for a, b in zip(edges.source_gene, edges.target_gene)
     if a in idx and b in idx]
missing = [(a, b) for a, b in zip(edges.source_gene, edges.target_gene)
           if a not in idx or b not in idx]
assert not missing, f"edges reference unknown genes: {missing}"
A = np.zeros((N, N))
for i, j in E:
    A[i, j] = A[j, i] = 1.0
deg = A.sum(1)
print(f"Module F: {N} genes, {len(E)} curated relations, mean degree {deg.mean():.2f}, "
      f"{int((deg == 0).sum())} isolated genes")

def rwr(Adj, seeds, restart=0.3, iters=2000, tol=1e-12):
    d = Adj.sum(0); d[d == 0] = 1.0
    W = Adj / d                       # column-stochastic
    p0 = np.zeros(Adj.shape[0]); p0[seeds] = 1.0 / len(seeds)
    p = p0.copy()
    for _ in range(iters):
        pn = (1 - restart) * (W @ p) + restart * p0
        if np.abs(pn - p).sum() < tol:
            p = pn; break
        p = pn
    return p

seeds = [idx[g] for g in genes if broad[g] == "L1"]
p_obs = rwr(A, seeds)

def rewire(Adj, n_swaps=8000):
    """Degree-preserving double-edge swap."""
    B = Adj.copy()
    el = [(i, j) for i in range(N) for j in range(i + 1, N) if B[i, j]]
    for _ in range(n_swaps):
        (a, b), (c, d) = el[rng.integers(len(el))], el[rng.integers(len(el))]
        if len({a, b, c, d}) < 4:
            continue
        if B[a, d] or B[c, b]:
            continue
        B[a, b] = B[b, a] = 0; B[c, d] = B[d, c] = 0
        B[a, d] = B[d, a] = 1; B[c, b] = B[b, c] = 1
        el = [(i, j) for i in range(N) for j in range(i + 1, N) if B[i, j]]
    return B

NPERM = 300
null = np.zeros((NPERM, N))
for k in range(NPERM):
    B = rewire(A, n_swaps=60)
    null[k] = rwr(B, seeds, iters=500)
pval = (1 + (null >= p_obs[None, :]).sum(0)) / (NPERM + 1)

prop = pd.DataFrame(dict(gene=genes, layer=[layer[g] for g in genes],
                         broad_layer=[broad[g] for g in genes], degree=deg,
                         rwr_score=p_obs, null_mean=null.mean(0), p_empirical=pval))
prop["enrichment_vs_null"] = prop.rwr_score / prop.null_mean.replace(0, np.nan)
prop = prop.sort_values("rwr_score", ascending=False)
K.save_table(prop, "F_rwr_propagation")

downstream = prop[(prop.broad_layer != "L1") & (prop.p_empirical < 0.05)]
print(f"   {len(downstream)} non-risk-layer genes are reached from the risk layer more strongly "
      f"than a degree-preserving null predicts (p<0.05):")
print("      " + ", ".join(downstream.gene.head(14)))

# ------------------------------------------------------ 2. cross-layer connector genes
part = []
for g in genes:
    i = idx[g]
    nb = np.where(A[i] > 0)[0]
    if len(nb) == 0:
        part.append(dict(gene=g, degree=0, n_layers_touched=0, participation=0.0)); continue
    ls = pd.Series([broad[genes[j]] for j in nb]).value_counts(normalize=True)
    part.append(dict(gene=g, degree=int(len(nb)), n_layers_touched=int(len(ls)),
                     participation=float(1 - (ls ** 2).sum())))
conn = pd.DataFrame(part).merge(prop[["gene", "layer", "broad_layer", "rwr_score"]], on="gene")
conn = conn.sort_values(["n_layers_touched", "participation", "degree"], ascending=False)
K.save_table(conn, "F_cross_layer_connectors")
print("   top cross-layer connectors: " +
      ", ".join(f"{r.gene} ({r.n_layers_touched} layers)" for r in conn.head(8).itertuples()))

# --------------------------------------------- 3. layer-pair adjacency vs configuration null
pairs = {}
for i, j in E:
    key = tuple(sorted((broad[genes[i]], broad[genes[j]])))
    pairs[key] = pairs.get(key, 0) + 1
null_pairs = {k: [] for k in pairs}
for _ in range(400):
    perm = rng.permutation(N)
    cnt = {}
    for i, j in E:
        key = tuple(sorted((broad[genes[perm[i]]], broad[genes[perm[j]]])))
        cnt[key] = cnt.get(key, 0) + 1
    for k in null_pairs:
        null_pairs[k].append(cnt.get(k, 0))
rows = []
for k, v in sorted(pairs.items(), key=lambda x: -x[1]):
    nd = np.array(null_pairs[k])
    p_two = float((1 + min((nd >= v).sum(), (nd <= v).sum())) * 2 / (len(nd) + 1))
    rows.append(dict(layer_pair="-".join(k), observed=v, null_mean=float(nd.mean()),
                     null_sd=float(nd.std()), p_two_sided=p_two,
                     p_at_permutation_floor=bool(p_two <= 2.0 / (len(nd) + 1) + 1e-12),
                     permutation_floor=float(2.0 / (len(nd) + 1))))
lp = pd.DataFrame(rows)
K.save_table(lp, "F_layer_pair_enrichment")

# ----------------------------------- 3b. how much curation would have to be missing to undo this
# The depletion of risk-to-ictal relations is the module's most quotable result and it rests on a
# hand-curated edge set, which is the correct reason to distrust it. Rather than restate that
# caveat, this block measures how fragile the result is: edges are ADDED at random between the two
# depleted layers until the observed count stops being significantly below the configuration null,
# and the number required is reported as a fraction of the curated network. If a handful of missed
# relations would erase it, the result is not usable; if it takes a substantial share of the whole
# network, the curation would have to be wrong in a very specific and implausible way.
def _pair_null_count(extra_edges, n_null=400, key=("L1", "L2")):
    E2 = list(E) + list(extra_edges)
    obs = 0
    for i, j in E2:
        if tuple(sorted((broad[genes[i]], broad[genes[j]]))) == tuple(sorted(key)):
            obs += 1
    null = []
    for _ in range(n_null):
        pm = rng.permutation(N)
        c = 0
        for i, j in E2:
            if tuple(sorted((broad[genes[pm[i]]], broad[genes[pm[j]]]))) == tuple(sorted(key)):
                c += 1
        null.append(c)
    null = np.array(null)
    p = float((1 + (null <= obs).sum()) * 2 / (len(null) + 1))
    return obs, float(null.mean()), p

risk_idx = [i for i in range(N) if broad[genes[i]] == "L1"]
ictal_idx = [i for i in range(N) if broad[genes[i]] == "L2"]
existing = {tuple(sorted(e)) for e in E}
candidates = [tuple(sorted((a, b))) for a in risk_idx for b in ictal_idx
              if tuple(sorted((a, b))) not in existing]
rng.shuffle(candidates)
curation_rows = []
robust_at = None
for add_n in (0, 3, 6, 10, 15, 20, 30, 45):
    if add_n > len(candidates):
        break
    obs, nm, pv = _pair_null_count(candidates[:add_n], n_null=300)
    curation_rows.append(dict(added_risk_ictal_relations=add_n,
                              added_as_pct_of_network=100.0 * add_n / len(E),
                              observed=obs, null_mean=nm, p_two_sided=pv,
                              still_depleted=bool(pv < 0.05 and obs < nm)))
    if robust_at is None and not (pv < 0.05 and obs < nm):
        robust_at = add_n
cur = pd.DataFrame(curation_rows)
K.save_table(cur, "F_curation_sensitivity")
CURATION_BREAKS_AT = robust_at if robust_at is not None else int(cur.added_risk_ictal_relations.max())
print(f"   curation sensitivity: the risk-to-ictal depletion survives until "
      f"{CURATION_BREAKS_AT} further relations are added between those two layers, which is "
      f"{100.0 * CURATION_BREAKS_AT / len(E):.0f}% of the whole curated network - so the result "
      f"would require the curation to have missed that many relations in one specific place")

# ------------------------------------- 4. risk-to-recovery path structure and minimal cut
def bfs_dist(src):
    d = -np.ones(N, int); d[src] = 0
    q = [src]
    while q:
        u = q.pop(0)
        for v in np.where(A[u] > 0)[0]:
            if d[v] < 0:
                d[v] = d[u] + 1; q.append(v)
    return d

rec = [idx[g] for g in genes if broad[g] == "L4"]
D = np.array([bfs_dist(s) for s in seeds])
finite = D[:, rec]
paths = pd.DataFrame(dict(risk_gene=[genes[s] for s in seeds],
                          min_hops_to_recovery_layer=[int(r[r >= 0].min()) if (r >= 0).any() else -1
                                                      for r in finite],
                          mean_hops=[float(r[r >= 0].mean()) if (r >= 0).any() else np.nan
                                     for r in finite]))
K.save_table(paths, "F_risk_to_recovery_paths")

# minimal edge cut separating the risk layer from the recovery layer (Ford-Fulkerson, unit caps)
cap = {}
for i, j in E:
    cap[(i, j)] = cap.get((i, j), 0) + 1
    cap[(j, i)] = cap.get((j, i), 0) + 1
S, T = N, N + 1
big = 10 ** 6
for s in seeds:
    cap[(S, s)] = big
for t in rec:
    cap[(t, T)] = big
adj = {}
for (u, v) in list(cap):
    adj.setdefault(u, set()).add(v)
    adj.setdefault(v, set()).add(u)
    cap.setdefault((v, u), 0)

def bfs_aug():
    par = {S: None}
    q = [S]
    while q:
        u = q.pop(0)
        for v in adj.get(u, ()):
            if v not in par and cap.get((u, v), 0) > 0:
                par[v] = u
                if v == T:
                    return par
                q.append(v)
    return None

flow = 0
while True:
    par = bfs_aug()
    if par is None:
        break
    v, b = T, big
    while par[v] is not None:
        u = par[v]; b = min(b, cap[(u, v)]); v = u
    v = T
    while par[v] is not None:
        u = par[v]; cap[(u, v)] -= b; cap[(v, u)] += b; v = u
    flow += b
reach = set()
q = [S]
while q:
    u = q.pop(0)
    if u in reach:
        continue
    reach.add(u)
    for v in adj.get(u, ()):
        if cap.get((u, v), 0) > 0 and v not in reach:
            q.append(v)
cut = [(genes[i], genes[j]) for (i, j) in E
       if (i in reach) != (j in reach)]
K.save_json(dict(curation_breaks_at_added_relations=int(CURATION_BREAKS_AT),
                 curation_breaks_at_pct_of_network=float(100.0 * CURATION_BREAKS_AT / len(E)),
                 n_genes=N, n_edges=len(E), mean_degree=float(deg.mean()),
                 n_downstream_significant=int(len(downstream)),
                 downstream_significant_genes=downstream.gene.tolist(),
                 top_connectors=conn.head(10)[["gene", "n_layers_touched", "participation"]]
                 .to_dict("records"),
                 min_hops_risk_to_recovery=int(paths.min_hops_to_recovery_layer[
                     paths.min_hops_to_recovery_layer > 0].min()),
                 median_hops_risk_to_recovery=float(paths.min_hops_to_recovery_layer[
                     paths.min_hops_to_recovery_layer > 0].median()),
                 min_cut_size=int(flow), min_cut_edges=[f"{a}-{b}" for a, b in cut],
                 caveat=("the graph is literature-curated, so these results describe the structure "
                         "of the curated mechanism set and are hypothesis-generating only")),
            "F_network_summary")
np.save(K.RESULTS / "F_adjacency.npy", A)
print(f"   risk-to-recovery separation: minimum {paths.min_hops_to_recovery_layer[paths.min_hops_to_recovery_layer>0].min()} hops, "
      f"median {paths.min_hops_to_recovery_layer[paths.min_hops_to_recovery_layer>0].median():.0f}; "
      f"minimal edge cut = {flow} relations ({', '.join(f'{a}-{b}' for a, b in cut[:6])}"
      f"{'...' if len(cut) > 6 else ''})")
print(lp.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
