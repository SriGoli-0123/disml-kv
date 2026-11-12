#!/usr/bin/env python
"""Write the SLURM array manifest for the v2 live experiments.

One line per (model, KV budget) cell (``gpu.cells`` in configs/v2.json); each array task starts one vLLM server
and runs every workload / load / policy for that cell (scripts/run_cell.sh).

    python scripts/make_manifest.py            # -> configs/gpu_manifest.tsv
    sbatch --array=0-$(( $(wc -l < configs/gpu_manifest.tsv) - 1 )) scripts/sol_array.sbatch
"""
import json
from pathlib import Path

cfg = json.load(open("configs/v2.json"))
g = cfg["gpu"]
lines = []
cells = g.get("cells") or [[m, kv] for m in g["models"] for kv in g["kv_gib"]]
for model, kv in cells:
    lines.append("\t".join([model, g["chat_formats"][model], f"{kv:g}"]))
Path("configs/gpu_manifest.tsv").write_text("\n".join(lines) + "\n")
print(f"{len(lines)} cells -> configs/gpu_manifest.tsv")
for i, l in enumerate(lines):
    print(i, l.replace("\t", "  "))
