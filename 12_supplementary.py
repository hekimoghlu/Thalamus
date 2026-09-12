#!/usr/bin/env python3
"""Assemble the supplementary tables and their index."""
import json, shutil
import pandas as pd
import kcore as K

SUP = K.ROOT / "supplementary"
SUP.mkdir(exist_ok=True)

TABLES = [
    ("S1", "datasets_inventory", "data", "Public dataset inventory (Data.gov / CDC / NCHS)"),
    ("S2", "gwas_loci", "data", "Stroke GWAS loci used in the polygenic model, with provenance"),
    ("S3", "genes_nodes", "data", "Genes in the multi-layer network, with NCBI Gene identifiers"),
    ("S4", "network_edges", "data", "Curated molecular relations and their evidence class"),
    ("S5", "monogenic_genes", "data", "Monogenic cerebrovascular disease genes"),
    ("S6", "thrombophilia", "data", "Thrombophilia and haemostasis variants"),
    ("S7", "cascade_params", "data", "Verified physiological parameters of the ictal cascade"),
    ("S8", "geo_series", "data", "Public transcriptomic series relevant to the continuum"),
    ("S9", "blood_signatures", "data", "Published peripheral-blood expression signatures"),
    ("S10", "epidemiology", "data", "US stroke epidemiology and risk-factor burden"),
    ("S11", "nhanes_labs", "data", "NHANES blood-measure availability, ranges and thresholds"),
    ("S12", "rehab_benchmarks", "data", "Neurorehabilitation benchmarks and measurement properties"),
    ("S13", "sensorimotor_params", "data", "Sensorimotor timing, psychophysics and feedback parameters"),
    ("S14", "compounds", "data", "Stroke-relevant compounds with PubChem identifiers"),
    ("S15", "bioassays", "data", "PubChem BioAssay records for the framework's targets"),
    ("S16", "proteins_structures", "data", "Protein-level annotation and experimental structures"),
    ("S17", "orthologs", "data", "Model-organism orthologs and focal-ischaemia model evidence"),
    ("S18", "sources", "data", "Numbered source list for every verified value"),
    ("S19", "A_locus_contributions", "results", "Per-locus contributions to the polygenic score"),
    ("S20", "A_prs_sensitivity", "results", "Polygenic-score sensitivity to effect shrinkage"),
    ("S21", "B_tissue_fate_validation", "results", "Tissue-fate model versus published values"),
    ("S22", "C_validation", "results", "Cascade model versus published values"),
    ("S23", "C_scenarios", "results", "Cascade outputs by perfusion scenario"),
    ("S24", "D_prr_identifiability_scan", "results", "Proportional-recovery identifiability scan"),
    ("S25", "D_time_course_validation", "results", "Recovery time course versus published values"),
    ("S26", "E_design_envelope", "results", "Closed-loop design envelope (latency x dose)"),
    ("S27", "E_predicted_effect_sizes", "results", "Predicted effect sizes by latency and contrast"),
    ("S28", "F_rwr_propagation", "results", "Network propagation scores and empirical p-values"),
    ("S29", "V_benchmark_master", "results", "Master benchmark table (all point comparisons)"),
    ("S30", "V_global_sensitivity", "results", "Global sensitivity coefficients"),
]

# every remaining results table is shipped too, numbered after the curated set, so that no value
# quoted in the manuscript lives only in an unshipped file
import glob as _glob
_have = {n for _, n, _, _ in TABLES}
_extra = sorted(p.stem for p in K.RESULTS.glob("*.csv") if p.stem not in _have)
for i, name in enumerate(_extra, start=32):
    TABLES.append((f"S{i}", name, "results", f"Model output: {name.replace('_', ' ')}"))

rows = []
for sid, name, where, desc in TABLES:
    src = (K.DATA if where == "data" else K.RESULTS) / f"{name}.csv"
    dst = SUP / f"Table_{sid}_{name}.csv"
    shutil.copy(src, dst)
    df = pd.read_csv(src)
    rows.append(dict(table=sid, file=dst.name, rows=len(df), columns=len(df.columns),
                     description=desc, origin=where))
idx = pd.DataFrame(rows)
idx.to_csv(SUP / "Table_index.csv", index=False)

lines = ["# Supplementary tables", "",
         "Every table is a plain CSV. Tables drawn from `data/verified/` contain only values read "
         "from a public source, each carrying a `src` key into Table S18 and a `verified` flag "
         "(`primary`, `secondary`, or `UNVERIFIED`). Tables drawn from `results/` are outputs of "
         "the shipped code and can be regenerated with `code/run_all.sh`.", "",
         "| Table | File | Rows | Cols | Contents |", "|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['table']} | `{r['file']}` | {r['rows']} | {r['columns']} | {r['description']} |")
(SUP / "README.md").write_text("\n".join(lines) + "\n")
print(f"  -> supplementary/: {len(rows)} tables + index")

# unverified-value audit
# The denominator excludes the source list (Table S18, which is itself the provenance record) and
# the curated relation table (Table S4, whose rows carry a mechanism and an evidence class rather
# than a per-row literature key - an exemption stated in the Methods).
EXCLUDE_FROM_AUDIT = {"sources", "network_edges"}
audit = []
for sid, name, where, desc in TABLES:
    if where != "data" or name in EXCLUDE_FROM_AUDIT:
        continue
    df = pd.read_csv(K.DATA / f"{name}.csv", dtype=str, keep_default_na=False)
    n_cells = df.size
    n_unv = int((df == "UNVERIFIED").sum().sum())
    audit.append(dict(table=sid, dataset=name, cells=n_cells, unverified_cells=n_unv,
                      pct_unverified=round(100 * n_unv / max(n_cells, 1), 1)))
au = pd.DataFrame(audit)
au.to_csv(SUP / "Table_S31_unverified_audit.csv", index=False)
idx = pd.concat([idx, pd.DataFrame([dict(table="S31", file="Table_S31_unverified_audit.csv",
                                         rows=len(au), columns=len(au.columns),
                                         description="Audit of unverified cells per verified data table",
                                         origin="audit")])], ignore_index=True)
idx.to_csv(SUP / "Table_index.csv", index=False)
lines.append(f"| S31 | `Table_S31_unverified_audit.csv` | {len(au)} | {len(au.columns)} | "
             f"Audit of unverified cells per verified data table |")
(SUP / "README.md").write_text("\n".join(lines) + "\n")

n_values = 0
for _, name, where, _ in TABLES:
    if where == "data" and name not in ("sources",):
        n_values += int(pd.read_csv(K.DATA / f"{name}.csv").notna().sum().sum())
print(f"  verified reference values across the evidence base: {n_values}")
print(f"  -> supplementary/Table_S31_unverified_audit.csv "
      f"(overall {au.unverified_cells.sum()}/{au.cells.sum()} cells = "
      f"{100*au.unverified_cells.sum()/au.cells.sum():.1f}% explicitly unverified)")
