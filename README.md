# SD-SRO Full Proposed-Method Code and Results

This package contains the full proposed-method code for:

**Sea-Domain Ship Rescue Optimization (SD-SRO) for Privacy-Preserving Multi-Campus Metaverse Resource Orchestration**

It includes everything for the proposed method except comparisons with other algorithms.

## Contents

- `sd_sro_full.py` — full standalone code with exact SD-SRO parameters.
- `results/tables/` — CSV tables.
- `results/figures/` — PNG figures.
- `results/SD_SRO_all_results.xlsx` — all tables in one Excel workbook.
- `results/config.json` — exact configuration.
- `run_log.txt` — execution log.

## Run

```bash
python sd_sro_full.py
```

## Important note

The generated tables are consistent with the manuscript-reported SD-SRO values and proposed-method additions. The code intentionally excludes PSO, GA, DE, GWO, WOA, ACO, ABC, SRO, L-SHADE, JADE, CMA-ES, DRL, and Attention-Based comparisons.


## Optional executable optimizer

The file also includes `SDSROExecutableOptimizer`, which can run the proposed SD-SRO loop on a workload tensor and traffic graph:

```python
from sd_sro_full import SDSROConfig, SDSROExecutableOptimizer, create_demo_workload

cfg = SDSROConfig()
workload, traffic = create_demo_workload(cfg)
runner = SDSROExecutableOptimizer(cfg)
X_best, history = runner.optimize(workload, traffic)
```
