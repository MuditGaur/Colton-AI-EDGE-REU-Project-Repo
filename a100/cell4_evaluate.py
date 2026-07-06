# ===== CELL 4 — EVALUATE: each trained model zero-shot @ {0.3,0.7,1.2} (150 eps) + random baseline -> CSV =====
# Run after cell3 (models in /workspace/models). Uses the validated handwritten eval (reward+collision+goal).
import os, sys, subprocess, pathlib, re, ast, csv

TEAM_SIZES = [9, 12]                     # <-- only difference vs the A100 cell
TEST_RADII = [0.3, 0.7, 1.2]
SEEDS = [0, 1, 2]
POD = "a100"
MODELS = "/workspace/models"
CACHE = "/workspace/jax_cache"       # caches the eval-rollout compile per (team,test_radius)
CSV_PATH = f"/workspace/{POD}_eval_results.csv"

def set_scenario(n):
    pathlib.Path(f"/workspace/Mava/mava/configs/env/scenario/simple_spread_{n}ag.yaml").write_text(
        f"name: MPE_simple_spread_v3\ntask_name: simple_spread_{n}ag\n\n"
        f"task_config:\n  num_agents: {n}\n  num_landmarks: {n}\n  local_ratio: 0.5\n")
def set_radius(r):
    jm = pathlib.Path("/workspace/Mava/mava/wrappers/jaxmarl.py")
    s = jm.read_text(); s = re.sub(r"(visibility_radius: float = )[0-9.]+(,)", rf"\g<1>{r}\g<2>", s, count=1); jm.write_text(s)

def final_ckpt(n, train_r, seed):
    d = pathlib.Path(f"{MODELS}/{n}_{train_r}_{seed}")
    cks = list(d.glob("params_*.msgpack")) if d.exists() else []
    if not cks: return None
    return str(max(cks, key=lambda p: int(re.search(r"params_(\d+)", p.name).group(1))))

def run_eval(n, test_r, load_path):
    set_scenario(n); set_radius(test_r)
    cmd = ["uv", "run", "python", "mava/systems/ppo/anakin/rec_mappo.py",
           "network=rnn_graph", "env=mpe", f"env/scenario=simple_spread_{n}ag",
           "arch.num_envs=8", "system.update_batch_size=1", "system.num_updates=2",
           "arch.num_evaluation=1", "arch.absolute_metric=False", "system.seed=0",
           "logger.checkpointing.save_model=False"]
    env = dict(os.environ); env["MPLBACKEND"] = "Agg"; env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_COMPILATION_CACHE_DIR"] = CACHE; env["CUSTOM_EVAL"] = "1"; env["EVAL_EPISODES"] = "150"
    if load_path: env["MODEL_LOAD_PATH"] = load_path
    result = None
    proc = subprocess.Popen(cmd, cwd="/workspace/Mava", env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        if line.startswith("CUSTOM_EVAL_RESULT"):
            try: result = ast.literal_eval(line[len("CUSTOM_EVAL_RESULT"):].strip())
            except Exception: pass
        if any(k in line for k in ("Error", "Traceback", "RESOURCE_EXHAUSTED", "EVAL_FAILED", "Exception")):
            sys.stdout.write(line); sys.stdout.flush()
    proc.wait()
    return result

rows = []
FIELDS = ["team", "train_radius", "seed", "test_radius", "is_random",
          "reward", "collision_rate", "goal_completion", "mean_landmark_dist"]
def add(team, train_r, seed, test_r, is_random, res):
    if not res: print(f"  !! no result: N={team} train={train_r} seed={seed} test={test_r}", flush=True); return
    rows.append({"team": team, "train_radius": train_r, "seed": seed, "test_radius": test_r,
                 "is_random": int(is_random), "reward": res.get("reward"),
                 "collision_rate": res.get("collision_rate"), "goal_completion": res.get("goal_completion"),
                 "mean_landmark_dist": res.get("mean_landmark_dist")})
    with open(CSV_PATH, "w", newline="") as f:            # rewrite incrementally so partial results survive
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

print("evaluating (zero-shot radius shift) ...", flush=True)
for n in TEAM_SIZES:
    for train_r in TEST_RADII:            # train radii == the 3 study radii
        for seed in SEEDS:
            ck = final_ckpt(n, train_r, seed)
            if ck is None: print(f"  missing model N={n} train={train_r} seed={seed}", flush=True); continue
            for test_r in TEST_RADII:
                res = run_eval(n, test_r, ck)
                add(n, train_r, seed, test_r, False, res)
                print(f"  N={n} train={train_r} seed={seed} -> test={test_r}: "
                      f"reward={res.get('reward') if res else None}", flush=True)
    for test_r in TEST_RADII:             # random (untrained) baseline per team
        add(n, "random", -1, test_r, True, run_eval(n, test_r, None))

print(f"\n>>> wrote {CSV_PATH} ({len(rows)} rows). Download it to your Mac and run the analysis locally.", flush=True)
