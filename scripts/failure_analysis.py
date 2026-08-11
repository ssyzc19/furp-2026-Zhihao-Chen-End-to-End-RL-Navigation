#!/usr/bin/env python3
"""
Failure analysis for HM3D PointNav eval.

Usage:
    python failure_analysis.py <ckpt_path> [--episodes 200] [--out results.json]

Runs HM3D val eval and classifies failures into:
  - near_miss:   failed, DTG in [0.20, 0.35m]  ← stop-aware 主要针对这个
  - bad_stop:    failed, DTG in [0.35, 1.00m]  ← 停在较远处
  - lost:        failed, DTG > 1.00m           ← 完全迷路/超时

Output: JSON with per-episode records + aggregate stats.
"""

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_path", help="Path to .pth checkpoint file")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", default=None, help="Output JSON path (default: <ckpt_dir>/failure_analysis.json)")
    return p.parse_args()


def run_eval(ckpt_path, num_episodes):
    """Run habitat eval and capture per-episode stats via monkey-patch."""
    import habitat_baselines.rl.ppo.habitat_evaluator as he

    per_episode_records = []
    original_eval = he.HabitatEvaluator.evaluate_agent

    def patched_eval(self, agent, envs, config, checkpoint_index, writer, *args, **kwargs):
        original_eval(self, agent, envs, config, checkpoint_index, writer, *args, **kwargs)
        # stats_episodes is a dict keyed by (scene_id, episode_id)
        for (scene_id, ep_id), stats in self._last_stats_episodes.items():
            per_episode_records.append({
                "scene_id": str(scene_id),
                "episode_id": str(ep_id),
                "success": float(stats.get("success", 0)),
                "spl": float(stats.get("spl", 0)),
                "distance_to_goal": float(stats.get("distance_to_goal", -1)),
                "reward": float(stats.get("reward", 0)),
            })

    # Instead of monkey-patching, we directly call habitat and parse stdout
    # More robust: run as subprocess and parse log output
    pass

    return per_episode_records


def run_eval_subprocess(ckpt_path, num_episodes):
    """Run eval as subprocess, parse per-episode stats from a saved JSON."""
    import subprocess
    import sys

    # Copy ckpt to tmpdir so Habitat only evals this one
    tmpdir = tempfile.mkdtemp()
    tmp_ckpt = os.path.join(tmpdir, "ckpt.0.pth")
    shutil.copy2(ckpt_path, tmp_ckpt)

    # Save stats to a JSON file via custom evaluator hook
    stats_path = os.path.join(tmpdir, "episode_stats.json")

    # Patch habitat_evaluator to dump stats
    patch_code = f"""
import habitat_baselines.rl.ppo.habitat_evaluator as _he
import json as _json

_orig = _he.HabitatEvaluator.evaluate_agent.__wrapped__ if hasattr(_he.HabitatEvaluator.evaluate_agent, '__wrapped__') else _he.HabitatEvaluator.evaluate_agent

def _patched(self, *a, **kw):
    result = _orig(self, *a, **kw)
    records = []
    for (scene_id, ep_id), stats in getattr(self, '_stats_episodes', {{}}).items():
        records.append({{
            'scene_id': str(scene_id),
            'episode_id': str(ep_id),
            'success': float(stats.get('success', 0)),
            'spl': float(stats.get('spl', 0)),
            'distance_to_goal': float(stats.get('distance_to_goal', -1)),
            'reward': float(stats.get('reward', 0)),
        }})
    with open('{stats_path}', 'w') as f:
        _json.dump(records, f)
    return result

_he.HabitatEvaluator.evaluate_agent = _patched
"""

    # Write patch to file
    patch_file = os.path.join(tmpdir, "eval_patch.pth")

    env = os.environ.copy()
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

    cmd = [
        sys.executable, "-u", "-m", "habitat_baselines.run",
        "--config-name=pointnav/ppo_pointnav.yaml",
        "benchmark/nav/pointnav=pointnav_hm3d",
        "habitat.dataset.split=val",
        "habitat_baselines.evaluate=True",
        f"habitat_baselines.eval_ckpt_path_dir={tmpdir}",
        "habitat_baselines.num_environments=10",
        f"habitat_baselines.test_episode_count={num_episodes}",
        "habitat_baselines.num_updates=-1",
        "habitat_baselines.load_resume_state_config=False",
        "~habitat.task.measurements.top_down_map",
        f"habitat_baselines.video_dir={tmpdir}/videos",
    ]

    print(f"Running eval on {ckpt_path} ({num_episodes} episodes)...")
    result = subprocess.run(cmd, env=env, cwd="/root/habitat-lab",
                           capture_output=False, text=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    return stats_path if os.path.exists(stats_path) else None


def classify_episode(ep):
    """Classify a single episode."""
    success = ep["success"] > 0.5
    dtg = ep["distance_to_goal"]

    if success:
        return "success"
    elif dtg < 0:
        return "unknown"
    elif dtg <= 0.35:
        return "near_miss"    # 停在 0.20-0.35m，差一点点
    elif dtg <= 1.00:
        return "bad_stop"     # 停在 0.35-1.00m
    else:
        return "lost"         # DTG > 1m，完全迷路或超时


def analyze_from_log(log_lines):
    """
    Parse per-episode stats directly from habitat eval stdout.
    Habitat prints aggregate stats but not per-episode — so we hook
    into the evaluator differently.
    """
    pass


def analyze_records(records):
    """Compute failure breakdown from per-episode records."""
    categories = defaultdict(list)
    for ep in records:
        cat = classify_episode(ep)
        categories[cat].append(ep)

    total = len(records)
    n_success = len(categories["success"])
    n_fail = total - n_success

    print(f"\n{'='*50}")
    print(f"FAILURE ANALYSIS  (n={total} episodes)")
    print(f"{'='*50}")
    print(f"Success:   {n_success:3d} / {total}  ({100*n_success/total:.1f}%)")
    print(f"Failed:    {n_fail:3d} / {total}  ({100*n_fail/total:.1f}%)")
    print(f"")
    print(f"Failure breakdown (of {n_fail} failures):")
    for cat in ["near_miss", "bad_stop", "lost"]:
        n = len(categories[cat])
        pct_of_fail = 100 * n / n_fail if n_fail > 0 else 0
        pct_of_total = 100 * n / total if total > 0 else 0
        avg_dtg = np.mean([e["distance_to_goal"] for e in categories[cat]]) if n > 0 else 0
        print(f"  {cat:12s}: {n:3d}  ({pct_of_fail:.1f}% of failures, {pct_of_total:.1f}% total)  avg_DTG={avg_dtg:.3f}m")

    # Near-miss DTG distribution
    if categories["near_miss"]:
        dtgs = [e["distance_to_goal"] for e in categories["near_miss"]]
        print(f"\nNear-miss DTG stats: min={min(dtgs):.3f}  max={max(dtgs):.3f}  mean={np.mean(dtgs):.3f}  median={np.median(dtgs):.3f}")

    return {
        "total": total,
        "success": n_success,
        "success_rate": n_success / total,
        "failure_breakdown": {
            cat: {
                "count": len(episodes),
                "pct_of_failures": len(episodes) / n_fail if n_fail > 0 else 0,
                "avg_dtg": float(np.mean([e["distance_to_goal"] for e in episodes])) if episodes else 0,
                "episodes": episodes,
            }
            for cat, episodes in categories.items()
        }
    }


def main():
    args = parse_args()

    if not os.path.isfile(args.ckpt_path):
        print(f"Error: {args.ckpt_path} is not a file")
        return

    # Determine output path
    out_path = args.out or os.path.join(
        os.path.dirname(args.ckpt_path), "failure_analysis.json"
    )

    # ── Strategy: patch habitat_evaluator at import time to dump stats ──
    # We inject a sitecustomize-style patch by modifying the evaluator source
    # temporarily. Simpler: just add _stats_episodes attribute in the evaluator.

    # Patch habitat_evaluator to save stats_episodes after eval
    evaluator_path = "/root/habitat-lab/habitat-baselines/habitat_baselines/rl/ppo/habitat_evaluator.py"

    with open(evaluator_path, "r") as f:
        src = f.read()

    # Insert dump after stats_episodes is populated (after the eval loop)
    dump_snippet = '''
        # [failure_analysis] dump per-episode stats to JSON
        import json as _json, os as _os
        _dump_path = _os.environ.get("HABITAT_STATS_DUMP_PATH", "")
        if _dump_path:
            _records = []
            for (_sid, _eid), _st in stats_episodes.items():
                _records.append({
                    "scene_id": str(_sid), "episode_id": str(_eid),
                    "success": float(_st.get("success", 0)),
                    "spl": float(_st.get("spl", 0)),
                    "distance_to_goal": float(_st.get("distance_to_goal", -1)),
                    "reward": float(_st.get("reward", 0)),
                })
            with open(_dump_path, "w") as _f:
                _json.dump(_records, _f)
            print(f"[failure_analysis] Saved {len(_records)} episode records to {_dump_path}")
        # [/failure_analysis]
'''

    marker = "aggregated_stats = {}"
    if "[failure_analysis]" not in src:
        patched_src = src.replace(marker, dump_snippet + "\n        " + marker)
        with open(evaluator_path, "w") as f:
            f.write(patched_src)
        print("[failure_analysis] Patched habitat_evaluator.py")

    # Run eval with dump path set
    tmpdir = tempfile.mkdtemp()
    tmp_ckpt = os.path.join(tmpdir, "ckpt.0.pth")
    shutil.copy2(args.ckpt_path, tmp_ckpt)
    stats_path = os.path.join(tmpdir, "episode_stats.json")

    import subprocess, sys
    env = os.environ.copy()
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    env["HABITAT_STATS_DUMP_PATH"] = stats_path

    cmd = [
        sys.executable, "-u", "-m", "habitat_baselines.run",
        "--config-name=pointnav/ppo_pointnav.yaml",
        "benchmark/nav/pointnav=pointnav_hm3d",
        "habitat.dataset.split=val",
        "habitat_baselines.evaluate=True",
        f"habitat_baselines.eval_ckpt_path_dir={tmpdir}",
        "habitat_baselines.num_environments=10",
        f"habitat_baselines.test_episode_count={args.episodes}",
        "habitat_baselines.num_updates=-1",
        "habitat_baselines.load_resume_state_config=False",
        "~habitat.task.measurements.top_down_map",
        f"habitat_baselines.video_dir={tmpdir}/videos",
    ]

    print(f"Evaluating: {args.ckpt_path}")
    print(f"Episodes:   {args.episodes}")
    subprocess.run(cmd, env=env, cwd="/root/habitat-lab")

    # Load and analyze
    if not os.path.exists(stats_path):
        print("ERROR: episode stats not saved. Check if patch worked.")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    with open(stats_path) as f:
        records = json.load(f)

    summary = analyze_records(records)

    # Save full results
    output = {
        "ckpt_path": args.ckpt_path,
        "num_episodes": args.episodes,
        "summary": {k: v for k, v in summary.items() if k != "failure_breakdown"},
        "failure_breakdown": {
            cat: {k: v for k, v in data.items() if k != "episodes"}
            for cat, data in summary["failure_breakdown"].items()
        },
        "episodes": records,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {out_path}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
