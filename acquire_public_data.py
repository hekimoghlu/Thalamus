#!/usr/bin/env python3
"""Fetch the primary public datasets this framework is calibrated against.

The analyses in this repository are calibrated against published summary values, each recorded
with its source in `data/verified/`. This script downloads the underlying primary data so that a
reader with unrestricted network access can extend the framework from summaries to record-level
reanalysis. It was NOT run to produce the results in the manuscript: the environment in which the
framework was built has no route to the NCBI or Data.gov hosts (the egress proxy answers 403 to
CONNECT for eutils.ncbi.nlm.nih.gov, ftp.ncbi.nlm.nih.gov, catalog.data.gov and data.cdc.gov), a
limitation stated in the Methods.

Usage:  python3 acquire_public_data.py --out ../data/raw [--what geo|cdc|nhanes|all]
"""
import argparse, sys, urllib.request, gzip, io
from pathlib import Path
import pandas as pd

GEO = {  # accessions whose identity, platform and sample counts are verified in Table S8
    "GSE16561": "peripheral whole blood, 39 ischaemic stroke vs 24 controls (GPL6883)",
    "GSE22255": "PBMC, 20 ischaemic stroke vs 20 matched controls (GPL570)",
    "GSE58294": "whole blood, cardioembolic stroke at <3 h, 5 h, 24 h vs 23 controls (GPL570)",
    "GSE37587": "whole blood, 34 patients at 0-24 h and 24-48 h (GPL6883)",
}
GEO_MATRIX = ("https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}nnn/{acc}/matrix/"
              "{acc}_series_matrix.txt.gz")
CDC = {
    "stroke_mortality_35plus_2021_2023": "https://data.cdc.gov/resource/cpdh-8cna.csv?$limit=500000",
    "rates_trends_heart_stroke_county": "https://data.cdc.gov/resource/7b9s-s8ck.csv?$limit=500000",
    "places_census_tract_2025": "https://data.cdc.gov/resource/cwsq-ngmh.csv?$limit=500000&measureid=STROKE",
}
NHANES = {  # August 2021 - August 2023 cycle unless noted
    "CBC_L": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt",
    "GHB_L": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.xpt",
    "GLU_L": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.xpt",
    "HSCRP_L": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HSCRP_L.xpt",
    "TCHOL_L": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.xpt",
    "TRIGLY_L": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TRIGLY_L.xpt",
    "P_BIOPRO": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BIOPRO.xpt",
    "L06MH_C": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L06MH_C.xpt",
}


def fetch(url, dest: Path, decompress=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  have {dest.name}")
        return dest
    print(f"  GET {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            blob = r.read()
    except Exception as e:                                   # noqa: BLE001
        print(f"  !! {type(e).__name__}: {e}\n     (no route to this host from this environment?)")
        return None
    if decompress:
        blob = gzip.decompress(blob)
    dest.write_bytes(blob)
    print(f"  -> {dest} ({len(blob)/1e6:.1f} MB)")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/raw")
    ap.add_argument("--what", default="all", choices=["geo", "cdc", "nhanes", "all"])
    a = ap.parse_args()
    out = Path(a.out)
    if a.what in ("geo", "all"):
        print("GEO series matrices")
        for acc, desc in GEO.items():
            stub = acc[:-3]
            fetch(GEO_MATRIX.format(stub=stub, acc=acc), out / "geo" / f"{acc}_series_matrix.txt",
                  decompress=True)
    if a.what in ("cdc", "all"):
        print("CDC / Data.gov tables")
        for name, url in CDC.items():
            fetch(url, out / "cdc" / f"{name}.csv")
    if a.what in ("nhanes", "all"):
        print("NHANES laboratory files")
        for name, url in NHANES.items():
            p = fetch(url, out / "nhanes" / f"{name}.xpt")
            if p:
                try:
                    df = pd.read_sas(p)
                    df.to_csv(p.with_suffix(".csv"), index=False)
                    print(f"     converted to CSV: {len(df)} rows")
                except Exception as e:                        # noqa: BLE001
                    print(f"     (SAS conversion failed: {e})")
    print("\nNext steps once the files are present:")
    print("  * GEO: quantile-normalise, map probes to NCBI Gene IDs, and run differential")
    print("    expression per cohort; the gene sets in Table S3 give the layer assignment.")
    print("  * CDC: re-fit the risk-architecture calibration in 01_risk_architecture.py against")
    print("    record-level age- and race-stratified rates rather than the published summaries.")
    print("  * NHANES: replace the reference intervals in Table S11 with survey-weighted")
    print("    distributions (remember the 2-year MEC weights) before re-running Module A.")


if __name__ == "__main__":
    sys.exit(main())
