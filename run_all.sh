#!/usr/bin/env bash
# Reproduce every number, table and figure in the manuscript from the verified data alone.
# Total runtime on a single core: about 35 minutes, most of it in the spatial model (Module J)
# and the Metropolis sampler (Module L).
set -euo pipefail
cd "$(dirname "$0")"

# --- layers, in dependency order ------------------------------------------------------------
python3 01_risk_architecture.py          # A  genomic risk architecture
python3 02_perfusion_tissue_fate.py      # B  perfusion, haemorheology, tissue fate
python3 03_ictal_cascade.py              # C  fourteen-state ictal cascade
python3 04_recovery_dynamics.py          # D  two-timescale recovery law, sensitive period
python3 05_closed_loop_control.py        # E  closed-loop interface (artificial peripheral thalamus)
python3 06_multilayer_network.py         # F  curated multi-layer molecular network
python3 07_validation_and_sensitivity.py #    first-pass validation and rank-regression sensitivity
python3 13_haemorrhage.py                # G  haemorrhage and haematoma expansion
python3 14_svd_cognition.py              # H  small-vessel disease, cognition, aphasia
python3 15_therapeutics.py               # I  therapeutic layer and trial attainability
python3 16_spatial_growth.py             # J  spatial stochastic reaction-diffusion growth
python3 17_social_gradient.py            # K  social and access geography
python3 18_bayesian_posteriors.py        # L  Bayesian posteriors for the load-bearing laws
python3 19_optimal_control.py            # M  stochastic optimal control of the device schedule
python3 23_delivery_physics.py           # O  delivery physics and the carrier specification
python3 20_statistics_hardening.py       # N  intervals, identifiability, Sobol, multiplicity

# --- figures and supplementary ---------------------------------------------------------------
python3 figures_schematics.py
python3 figures_data_a.py
python3 figures_data_b.py
python3 figures_data_c.py
python3 figures_new_biology.py
python3 figures_new_methods.py
python3 supplementary.py

echo "done - results/ figures/ supplementary/ are up to date"
