"""
Stop-Aware Reward Shaping for PointNav PPO
==========================================

Targets the near-miss failure mode found in the test-scenes baseline: 75% of
failures were the agent stopping 0.21-0.25 m from a 0.20 m success threshold
(a STOP-decision problem, not a path-finding problem).

DESIGN (important — read before changing):
    The dominant, reward-hacking-safe signal is a ONE-TIME terminal shaping on
    the STOP action:
        * agent calls STOP while inside success_distance   -> +correct_stop_bonus
        * agent calls STOP while outside  success_distance -> +wrong_stop_penalty
    This is only applied on the step STOP is actually called, so it cannot be
    farmed.

    The optional per-step proximity bonus is OFF by default (scale = 0.0).
    A dense per-step bonus for "being near the goal" is easily reward-hacked:
    it can exceed the slack penalty, so the agent learns to loiter near the
    goal WITHOUT stopping — which makes the near-miss problem worse, not
    better. distance_to_goal_reward already rewards approaching the goal, so
    the proximity term is largely redundant. Leave it at 0.0 for the clean
    ablation; only raise it if you specifically want to study its effect.

USAGE (runtime monkey-patch, no habitat-lab source edit needed):
    Train the stop-aware variant via train_stop_aware_wrapper.py, which calls
    patch_distance_to_goal_reward() BEFORE habitat_baselines starts. The
    baseline variant runs habitat_baselines.run unmodified, so the ONLY
    difference between the two arms is this patch — a clean controlled ablation.

    Tune values with env vars (so the same wrapper serves every seed/config):
        STOP_CORRECT_BONUS   (default  2.0)
        STOP_WRONG_PENALTY   (default -1.0)
        STOP_PROXIMITY_SCALE (default  0.0)   # keep 0.0 unless ablating it
        STOP_PROXIMITY_THRESH(default  0.3)
"""

import os


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def patch_distance_to_goal_reward():
    """
    Monkey-patch DistanceToGoalReward.update_metric to add stop-aware shaping.
    Call this ONCE, before habitat_baselines.run starts training.

    Correctness notes vs. a naive implementation:
      * STOP detection uses ``task.is_stop_called`` — the same flag Habitat's
        own Success measure uses. ``episode.is_episode_active`` is NOT a valid
        Habitat attribute and must not be used here.
      * The terminal stop bonus/penalty fires only on the STOP step, so it is
        not farmable.
    """
    from habitat.tasks.nav.nav import DistanceToGoalReward, DistanceToGoal

    correct_stop_bonus = _env_float("STOP_CORRECT_BONUS", 2.0)
    wrong_stop_penalty = _env_float("STOP_WRONG_PENALTY", -1.0)
    proximity_scale = _env_float("STOP_PROXIMITY_SCALE", 0.0)
    proximity_thresh = _env_float("STOP_PROXIMITY_THRESH", 0.3)
    # success_distance is read from the task config at runtime; fall back to 0.2
    success_distance_default = _env_float("STOP_SUCCESS_DISTANCE", 0.2)

    _original_update = DistanceToGoalReward.update_metric

    def _stop_aware_update(self, episode, task, *args, **kwargs):
        # 1. base reward (distance-decrease + any existing shaping)
        _original_update(self, episode, task, *args, **kwargs)

        # 2. current geodesic distance to goal (already computed this step)
        current_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if current_distance is None:
            return

        # success radius: prefer the task/measure config, else default 0.2
        success_distance = getattr(
            self._config, "success_distance", success_distance_default
        )

        # 3. optional per-step proximity shaping (OFF by default — see header)
        if proximity_scale > 0.0 and current_distance < proximity_thresh:
            self._metric += proximity_scale * (
                (proximity_thresh - current_distance) / proximity_thresh
            )

        # 4. terminal stop shaping — the main signal. Fires only on the step
        #    the agent actually calls STOP (task.is_stop_called), so it is not
        #    farmable.
        if getattr(task, "is_stop_called", False):
            if current_distance < success_distance:
                self._metric += correct_stop_bonus
            else:
                self._metric += wrong_stop_penalty

    DistanceToGoalReward.update_metric = _stop_aware_update
    print(
        "[PATCH] stop-aware reward ENABLED | "
        f"correct_stop=+{correct_stop_bonus} wrong_stop={wrong_stop_penalty} "
        f"proximity_scale={proximity_scale} (0=off) "
        f"proximity_thresh={proximity_thresh}"
    )


if __name__ == "__main__":
    print(__doc__)
    print("Import and call patch_distance_to_goal_reward() before training.")
