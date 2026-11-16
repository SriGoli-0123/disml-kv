#!/usr/bin/env python
"""Write the SLURM array manifest for the v2 live experiments.

One line per (model, KV budget) cell; each array task starts one vLLM server
and runs every workload / load / policy for that cell (scripts/run_cell.sh).

    python scripts/make_manifest.py            # -> configs/gpu_manifest.tsv
    sbatch --array=0-$(( $(wc -l < configs/gpu_manifest.tsv) - 1 )) scripts/sol_array.sbatch
"""
import json
from pathlib import Path

cfg = json.load(open("configs/v2.json"))
g = cfg["gpu"]
lines = []
for model in g["models"]:
    for kv in g["kv_gib"]:
        lines.append("\t".join([model, g["chat_formats"][model], str(kv)]))
Path("configs/gpu_manifest.tsv").write_text("\n".join(lines) + "\n")
print(f"{len(lines)} cells -> configs/gpu_manifest.tsv")
for i, l in enumerate(lines):
    print(i, l.replace("\t", "  "))
