# ===== CELL 1 — SETUP (run once per pod): clone Mava, install, write eval module, patch rec_mappo =====
# Applies all pipeline hooks to rec_mappo, controlled by env vars set in cells 2-4:
#   MODEL_LOAD_PATH -> load trained params        CUSTOM_EVAL -> run handwritten eval + exit
#   CKPT_DIR/CKPT_EVERY -> periodic+final checkpoints    (step-0 baseline prints automatically in train mode)
# Same file for the 3090 and A100 pods.

get_ipython().run_line_magic("cd", "/workspace")
get_ipython().system("test -d Mava || git clone https://github.com/instadeepai/Mava.git")
get_ipython().run_line_magic("cd", "/workspace/Mava")
get_ipython().system("pip install -q uv")
get_ipython().system("uv sync --extra cuda12")
get_ipython().system('uv run python -c "import jax; print(\'JAX devices:\', jax.devices())"')

import pathlib

# ---------- validated handwritten eval (reward + collision_rate + goal_completion@0.20) ----------
pathlib.Path("/workspace/Mava/custom_eval.py").write_text(r'''
from typing import Any, Dict
import jax, jax.numpy as jnp
from mava.networks import ScannedRNN

def _unwrap(obj: Any, attr: str, max_depth: int = 10):
    s = obj
    for _ in range(max_depth):
        if hasattr(s, attr):
            return getattr(s, attr)
        for nxt in ("env_state", "state", "_env"):
            if hasattr(s, nxt):
                s = getattr(s, nxt); break
        else:
            break
    raise AttributeError(f"{attr} not found within {type(obj)}")

def run(eval_env, eval_act_fn, actor_params, config, key, num_episodes=150,
        collision_dist=0.30, goal_dist=0.20) -> Dict[str, float]:
    num_agents = int(config.system.num_agents); num_landmarks = num_agents
    hidden_dim = int(config.network.hidden_state_dim)
    max_steps = int(_unwrap(eval_env, "max_steps"))
    reset_keys = jax.random.split(key, num_episodes)
    env_states, ts = jax.vmap(eval_env.reset)(reset_keys)
    hidden = ScannedRNN.initialize_carry((num_episodes, num_agents), hidden_dim)
    def step(carry, _):
        env_states, ts, hidden, key = carry
        key, ak = jax.random.split(key)
        action, astate = eval_act_fn(actor_params, ts, ak, {"hidden_state": hidden})
        env_states, ts = jax.vmap(eval_env.step)(env_states, action)
        return (env_states, ts, astate["hidden_state"], key), (ts.reward, _unwrap(env_states, "p_pos"))
    _, (rewards, ppos) = jax.lax.scan(step, (env_states, ts, hidden, key), None, max_steps)
    ep_reward = rewards.sum(axis=0).mean(axis=-1)
    ap = ppos[:, :, :num_agents, :]
    d = jnp.linalg.norm(ap[:, :, :, None, :] - ap[:, :, None, :, :], axis=-1)
    collide = (d < collision_dist) & (~jnp.eye(num_agents, dtype=bool))[None, None]
    ep_col = (collide.sum(axis=(-1, -2)) / 2.0).sum(axis=0)
    final = ppos[-1]; a_pos = final[:, :num_agents, :]; l_pos = final[:, num_agents:num_agents+num_landmarks, :]
    nearest = jnp.linalg.norm(l_pos[:, :, None, :] - a_pos[:, None, :, :], axis=-1).min(axis=-1)
    lm_motion = float(ppos[:, :, num_agents:num_agents+num_landmarks, :].std(axis=0).mean())
    return {"reward": float(ep_reward.mean()),
            "collision_rate": float(ep_col.mean()),
            "goal_completion": float((nearest < goal_dist).mean()),
            "mean_landmark_dist": float(nearest.mean()),
            "landmarks_motion": lm_motion,   # sanity: should be ~0
            "num_episodes": num_episodes, "max_steps": max_steps}
''')
print("wrote custom_eval.py", flush=True)

# ---------- patch rec_mappo.py (idempotent, env-var controlled) ----------
rp = pathlib.Path("/workspace/Mava/mava/systems/ppo/anakin/rec_mappo.py")
src = rp.read_text()
if "PIPELINE_PATCHES" not in src:
    load_block = (
        "    import os as _os2\n"
        "    if _os2.environ.get('MODEL_LOAD_PATH'):\n"
        "        import flax.serialization as _fs2, jax as _jax2, jax.numpy as _jnp2\n"
        "        _tmpl = unreplicate_n_dims(learner_state.params)\n"
        "        with open(_os2.environ['MODEL_LOAD_PATH'], 'rb') as _f2:\n"
        "            _plain = _fs2.from_bytes(_tmpl, _f2.read())\n"
        "        _bc = lambda x: _jnp2.broadcast_to(x, (n_devices, config.system.update_batch_size, *x.shape))\n"
        "        learner_state = learner_state._replace(params=_jax2.tree_util.tree_map(_bc, _plain))\n")
    assert "    # Setup evaluator." in src
    src = src.replace("    # Setup evaluator.", load_block + "    # Setup evaluator.", 1)

    ev_anchor = "    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)"
    ce_block = ("\n    if _os2.environ.get('CUSTOM_EVAL'):\n"
        "        import sys as _sysp; _sysp.path.insert(0, '/workspace/Mava')\n"
        "        import custom_eval as _ce\n"
        "        _ap = unreplicate_n_dims(learner_state.params.actor_params)\n"
        "        _res = _ce.run(eval_env, eval_act_fn, _ap, config, key_e,\n"
        "                       num_episodes=int(_os2.environ.get('EVAL_EPISODES', '150')))\n"
        "        print('CUSTOM_EVAL_RESULT', _res, flush=True)\n"
        "        import sys as _sysx; _sysx.exit(0)\n")
    assert ev_anchor in src
    src = src.replace(ev_anchor, ev_anchor + ce_block, 1)

    step0_block = ("    if not _os2.environ.get('CUSTOM_EVAL'):\n"
        "        try:\n"
        "            _p0 = unreplicate_batch_dim(learner_state.params.actor_params)\n"
        "            key_e, *_ek0 = jax.random.split(key_e, n_devices + 1)\n"
        "            _ek0 = jnp.stack(_ek0).reshape(n_devices, -1)\n"
        "            _m0 = evaluator(_p0, _ek0, {'hidden_state': eval_hs})\n"
        "            print('STEP0_EVAL', float(jnp.mean(_m0['episode_return'])), flush=True)\n"
        "        except Exception as _e0:\n"
        "            print('STEP0_EVAL_FAILED', repr(_e0), flush=True)\n")
    assert "    for eval_step in range(config.arch.num_evaluation):" in src
    src = src.replace("    for eval_step in range(config.arch.num_evaluation):",
                      step0_block + "    for eval_step in range(config.arch.num_evaluation):", 1)

    ckpt_anchor = ("        # Update runner state to continue training.\n"
                   "        learner_state = learner_output.learner_state")
    ckpt_block = ("\n        if _os2.environ.get('CKPT_DIR'):\n"
        "            import flax.serialization as _fsc, os as _osc\n"
        "            _cd = _os2.environ['CKPT_DIR']; _ev = int(_os2.environ.get('CKPT_EVERY', '200000'))\n"
        "            if (t // _ev != (t - steps_per_rollout) // _ev) or (eval_step == config.arch.num_evaluation - 1):\n"
        "                _osc.makedirs(_cd, exist_ok=True)\n"
        "                _fpp = unreplicate_n_dims(learner_output.learner_state.params)\n"
        "                with open(_osc.path.join(_cd, 'params_%d.msgpack' % t), 'wb') as _fcc:\n"
        "                    _fcc.write(_fsc.to_bytes(_fpp))\n"
        "                print('CKPT_SAVED', t, flush=True)\n")
    assert ckpt_anchor in src
    src = src.replace(ckpt_anchor, ckpt_anchor + ckpt_block, 1)

    src = "# PIPELINE_PATCHES\n" + src
    rp.write_text(src)
print("pipeline patches applied:", "PIPELINE_PATCHES" in rp.read_text(), flush=True)
