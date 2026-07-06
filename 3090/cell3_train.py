# ===== CELL 3 — TRAIN each (team x radius x seed): step-0 baseline, window=10, ckpt every 200k, curves =====
# Run cell1 + cell2 first (cache present). 3090 = N=6 (9 runs). Saves models + convergence JSONs, zips -> Mac.
import os, time, subprocess, sys, pathlib, re, shutil, json

TEAM_SIZES = [6]                     # <-- only difference vs the A100 cell
RADII = [0.3, 0.7, 1.2]
SEEDS = [0, 1, 2]
POD = "3090"
NUM_EVALUATION = 200                 # window=10 -> ~2.05M steps (num_updates=2000). Same max for all teams.
CACHE = "/workspace/jax_cache"
MODELS = "/workspace/models"; CURVES = "/workspace/curves"
os.makedirs(MODELS, exist_ok=True); os.makedirs(CURVES, exist_ok=True)

def set_scenario(n):
    pathlib.Path(f"/workspace/Mava/mava/configs/env/scenario/simple_spread_{n}ag.yaml").write_text(
        f"name: MPE_simple_spread_v3\ntask_name: simple_spread_{n}ag\n\n"
        f"task_config:\n  num_agents: {n}\n  num_landmarks: {n}\n  local_ratio: 0.5\n")
def set_radius(r):
    jm = pathlib.Path("/workspace/Mava/mava/wrappers/jaxmarl.py")
    s = jm.read_text(); s = re.sub(r"(visibility_radius: float = )[0-9.]+(,)", rf"\g<1>{r}\g<2>", s, count=1); jm.write_text(s)

def train_one(n, r, seed):
    set_scenario(n); set_radius(r)
    tag = f"{n}_{r}_{seed}"
    ckdir = f"{MODELS}/{tag}"; shutil.rmtree(ckdir, ignore_errors=True)
    cmd = ["uv", "run", "python", "mava/systems/ppo/anakin/rec_mappo.py",
           "network=rnn_graph", "env=mpe", f"env/scenario=simple_spread_{n}ag",
           "arch.num_envs=8", "system.update_batch_size=1",
           f"system.num_updates={NUM_EVALUATION*10}", f"arch.num_evaluation={NUM_EVALUATION}",  # window=10
           "arch.absolute_metric=False", f"system.seed={seed}", "logger.checkpointing.save_model=False"]
    env = dict(os.environ); env["MPLBACKEND"] = "Agg"
    env["JAX_COMPILATION_CACHE_DIR"] = CACHE; env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["CKPT_DIR"] = ckdir; env["CKPT_EVERY"] = "200000"
    print(f"\n===== TRAIN N={n} radius={r} seed={seed} =====", flush=True)
    t0 = time.time(); data = []; cur_step = None; cur_train = None; win = 0; slow = False
    proc = subprocess.Popen(cmd, cwd="/workspace/Mava", env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        low = line.lower()
        if "slow_operation_alarm" in low or "very slow compile" in low or "operation took" in low: slow = True
        if line.strip().startswith("STEP0_EVAL"):
            try: data.append({"step": 0, "train": None, "eval": float(line.split()[1])})
            except Exception: pass
        m = re.search(r"Timestep:\s*(\d+)", line)
        if m: cur_step = int(m.group(1))
        if "ACTOR -" in line and "Episode return mean" in line:
            mm = re.search(r"Episode return mean:\s*(-?[0-9.]+)", line); cur_train = float(mm.group(1)) if mm else None
        if "EVALUATOR -" in line and "Episode return mean" in line:
            win += 1; mm = re.search(r"Episode return mean:\s*(-?[0-9.]+)", line)
            data.append({"step": cur_step, "train": cur_train, "eval": float(mm.group(1)) if mm else None})
            json.dump(data, open(f"{CURVES}/{tag}.json", "w"))
            if win == 1:
                el = time.time() - t0
                print(f"  [cache {'MISS -> recompiled' if (slow or el > 300) else 'HIT'}] first window at {el:.0f}s", flush=True)
            if win % 20 == 0 or win == NUM_EVALUATION:
                print(f"  win {win:>3}/{NUM_EVALUATION} | {cur_step:>9,} steps | train {cur_train} eval {data[-1]['eval']} | {(time.time()-t0)/60:.1f}m", flush=True)
        if any(k in low for k in ("error", "resource_exhausted", "traceback", "param save failed")):
            sys.stdout.write(line); sys.stdout.flush()
    proc.wait()
    nck = len(list(pathlib.Path(ckdir).glob("*.msgpack"))) if pathlib.Path(ckdir).exists() else 0
    print(f"  done {tag} in {(time.time()-t0)/60:.1f}m (exit {proc.returncode}); {len(data)} curve pts, {nck} checkpoints", flush=True)
    return proc.returncode == 0 and nck > 0

ok = True
for n in TEAM_SIZES:
    for r in RADII:
        for seed in SEEDS:
            ok = train_one(n, r, seed) and ok

print("\n============ TRAINING DONE ============", flush=True)
# zip ONLY models + curves (not the whole /workspace)
import zipfile
zp = f"/workspace/{POD}_train_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for base in (MODELS, CURVES):
        for f in pathlib.Path(base).rglob("*"):
            if f.is_file(): z.write(f, f.relative_to("/workspace"))
print(f">>> zipped models + curves -> {zp} ({os.path.getsize(zp)/1e6:.1f} MB). Download to your Mac.", flush=True)
print(f">>> (all {'OK' if ok else 'NOT all runs OK — check above'}.)", flush=True)
