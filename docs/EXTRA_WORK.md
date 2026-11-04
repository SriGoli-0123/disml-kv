# Extra work (local branch `chay/semantic-qa-and-seeds`)

Three follow-ups to the Oct 27 status doc. Nothing here changes the policies, the
simulator, the vLLM plugin or any committed result.

## 1. Semantic QA metric (second quality check)

EM/F1 treat a correct paraphrase as wrong. `aclkv/semantic.py` adds cosine similarity
between the cleaned prediction and the gold answer. Yes/no mismatches are forced to 0
(embeddings put "yes" and "no" close together).

```bash
pip install -e ".[semantic]"
python scripts/semantic_eval.py                   # neural: all-MiniLM-L6-v2 (needs one model download)
python scripts/semantic_eval.py --backend char    # offline stand-in, NOT neural
```

Reads the 192 saved `results/bench/*.json` (no GPU). Writes `results/semantic_qa_<backend>.{csv,md}`
and `_examples.csv`. The `.md` has a per-policy table (EM, F1, sem_sim, sem_match, rescued) and a
paired difference against B3 with a 95% bootstrap CI (resampling whole questions).

**Check the threshold (default 0.8) against `_examples.csv` before quoting `sem_match`.** With the
char stand-in it accepts "Kelly Osbourne" for "Kelly Lee Osbourne" (fine) but also "French and English"
for "French, English and Spanish" (incomplete). Re-check with the neural encoder.

## 2. More seeds and users (simulator only, CPU)

```bash
bash scripts/run_extra_sim.sh                     # SEEDS="1 2 3 4" USERS="50 100" by default
```

Needs `data/prepared` (run `prepare_data.py` first). Writes `results/sim/extra/` and
`results/sim_seed_consistency.{csv,md}` (cached % as mean +/- std over seeds; exits non-zero if any
secure policy shows an unauthorized hit; UB* variants are insecure on purpose).
Note: `n_groups` stays 5 from `configs/matrix.json` even at 100 users (the proposal said 4-10).

## 3. Security probe for the policies not yet probed (needs an A100)

The probe covered UB, B3, B4, B5. B1, B2 and B6 are still unprobed. For each KV size (1, 2, 4):

```bash
bash scripts/start_vllm.sh 2                      # shell 1 (change 2 to 1 or 4)
python scripts/security_probe.py \
  --workload data/workloads/w_share0.50_mix-default_u20_s0.json \
  --policies B1 B2 B6 --out results/security_probe_B1B2B6_kv2.json    # shell 2
python scripts/summarize.py                       # picks up security_probe*.json automatically
```

Pass criteria: `no_salt` and `forged` hits must be 0/20. The `stolen` control is expected to hit 20/20.
