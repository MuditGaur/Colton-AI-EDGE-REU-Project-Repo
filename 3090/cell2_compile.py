# ===== CELL 2 — COMPILE each (team x radius) config @ window=10, confirm VRAM fit, zip cache -> Mac =====
# Run cell1 first. 3090 handles N=6; A100 version sets TEAM_SIZES=[9,12].
import os, time, subprocess, sys, pathlib, re, shutil

TEAM_SIZES = [6]                     # <-- only difference vs the A100 cell
RADII = [0.3, 0.7, 1.2]
POD = "3090"
CACHE = "/workspace/jax_cache"
LOG = "/workspace/compile.log"; logf = open(LOG, "a")
def out(m): sys.stdout.write(m); sys.stdout.flush(); logf.write(m); logf.flush()

def set_scenario(n):
    pathlib.Path(f"/workspace/Mava/mava/configs/env/scenario/simple_spread_{n}ag.yaml").write_text(
        f"name: MPE_simple_spread_v3\ntask_name: simple_spread_{n}ag\n\n"
        f"task_config:\n  num_agents: {n}\n  num_landmarks: {n}\n  local_ratio: 0.5\n")

def set_radius(r):
    jm = pathlib.Path("/workspace/Mava/mava/wrappers/jaxmarl.py")
    s = jm.read_text(); s = re.sub(r"(visibility_radius: float = )[0-9.]+(,)", rf"\g<1>{r}\g<2>", s, count=1); jm.write_text(s)

def compile_one(n, r):
    set_scenario(n); set_radius(r)
    cmd = ["uv", "run", "python", "mava/systems/ppo/anakin/rec_mappo.py",
           "network=rnn_graph", "env=mpe", f"env/scenario=simple_spread_{n}ag",
           "arch.num_envs=8", "system.update_batch_size=1",
           "system.num_updates=10", "arch.num_evaluation=1",     # window=10
           "arch.absolute_metric=False", "system.seed=0", "logger.checkpointing.save_model=False"]
    env = dict(os.environ); env["MPLBACKEND"] = "Agg"
    env["JAX_COMPILATION_CACHE_DIR"] = CACHE; env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    out(f"\n##### N={n} radius={r} #####\n"); t0 = time.time(); verdict = None
    proc = subprocess.Popen(cmd, cwd="/workspace/Mava", env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        out(line); low = line.lower()
        if verdict is None and ("resource_exhausted" in low or "out of memory" in low): verdict = "OOM"
        if verdict is None and "ACTOR" in line and "Steps per second" in line: verdict = "OK"; break
    try: proc.kill()
    except Exception: pass
    proc.wait()
    out(f"\n>>> N={n} r={r}: {verdict or 'UNKNOWN'} in {(time.time()-t0)/60:.1f} min\n")
    return verdict

out(f"\n===== {POD} COMPILE: teams={TEAM_SIZES} radii={RADII} =====\n")
res = {}
for n in TEAM_SIZES:
    for r in RADII:
        res[(n, r)] = compile_one(n, r); time.sleep(4)

out("\n============ SUMMARY ============\n")
for (n, r), v in res.items(): out(f"  N={n:>2} radius={r}: {v}\n")
if all(v == "OK" for v in res.values()):
    shutil.make_archive(f"/workspace/{POD}_cache", "zip", CACHE)
    mb = os.path.getsize(f"/workspace/{POD}_cache.zip") / 1e6
    out(f"\nALL OK. cache zipped -> /workspace/{POD}_cache.zip ({mb:.0f} MB). Download it to your Mac.\n")
else:
    out("\nSOME FAILED (OOM/unknown) — fix before zipping.\n")
logf.close()
