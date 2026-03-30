"""
balans_tuner.py — Optuna-based Balans hyperparameter tuner.

Searches over the Balans configuration space for a given folder of MIP
instances.  The tunable dimensions are:

  1. **Operator portfolio**  – which destroy-operator families to include and at
     what intensity (few / moderate / spread / dense).  Crossover is always
     included.  Operator variants and presets are derived from the top-
     performing configurations in the ParBalans benchmark study.
  2. **Learning policy**     – ThompsonSampling, EpsilonGreedy(ε), or Softmax(τ).
  3. **Score vector**        – reward weights [best, better, accept, reject] for
     the MAB selector, drawn from vectors observed in top configs.
  4. **Acceptance criterion** – HillClimbing or SimulatedAnnealing (with
     temperature parameters).

For efficiency each trial runs Balans for a *short* budget (default 120 s) on a
small representative subset of instances (default: 3 smallest by file size).
Initial solutions are cached across trials.

Usage
-----
  # Tune on all instances in a folder:
  python balans_tuner.py --instance_dir data/mipfeas

  # Tune on a subset listed in a file (one filename per line):
  python balans_tuner.py --instance_dir data/mipfeas --instances_to_run instances.txt

  # Custom budget / instance count:
  python balans_tuner.py --instance_dir data/mipfeas --n_trials 100 --balans_time_limit 180 --n_representative 5

  # Use Gurobi backend:
  python balans_tuner.py --instance_dir data/mipfeas --mip_solver gurobi

  # Dry-run – print search space, do nothing:
  python balans_tuner.py --dry_run

Outputs (in --output_dir, default: results_tuner/)
---------------------------------------------------
  best_config.json       Best configuration found
  optuna.pkl              Optuna object for analysis
"""

import os
import sys
import argparse
import json
import pickle
import time as _time
from pathlib import Path

import numpy as np

# ── Balans imports (installed package) ────────────────────────────
from alns.select import MABSelector, RandomSelect
from alns.accept import HillClimbing, SimulatedAnnealing
from alns.stop import MaxRuntime
from mabwiser.mab import LearningPolicy

from balans.solver import Balans, DestroyOperators, RepairOperators

import optuna
# Silence Optuna's verbose logging during trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results_tuner"

# Supported MIP file extensions
MIP_EXTENSIONS = (".mps", ".lp") # ".mps.gz", ".lp.gz",


# ═══════════════════════════════════════════════════════════════════
# Operator catalogue  (data-driven from ParBalans top configs)
# ═══════════════════════════════════════════════════════════════════
# Each family maps an "intensity preset" to a list of operator attribute names
# on DestroyOperators.  Presets are derived from the operator combinations
# observed in the top-10 configurations across all benchmark datasets
# (overall, miplib-hard, dist-miplib, real).
#
# Crossover appears in 100 % of top configs and is always included.
# Dins and Random_Objective appear in 0 % of top configs and are excluded.

OPERATOR_FAMILIES = {
    "mutation": {
        "few":      ["Mutation_50"],
        "moderate": ["Mutation_30", "Mutation_50"],
        "spread":   ["Mutation_10", "Mutation_30", "Mutation_50"],
        "dense":    ["Mutation_10", "Mutation_20", "Mutation_30",
                     "Mutation_40", "Mutation_50"],
    },
    "local_branching": {
        "few":      ["Local_Branching_20", "Local_Branching_50"],
        "moderate": ["Local_Branching_10", "Local_Branching_30",
                     "Local_Branching_50"],
        "spread":   ["Local_Branching_10", "Local_Branching_20",
                     "Local_Branching_40", "Local_Branching_50"],
        "dense":    ["Local_Branching_10", "Local_Branching_20",
                     "Local_Branching_30", "Local_Branching_40",
                     "Local_Branching_50"],
    },
    "proximity": {
        "few":      ["Proximity_010", "Proximity_030"],
        "moderate": ["Proximity_005", "Proximity_015", "Proximity_030"],
        "spread":   ["Proximity_005", "Proximity_010", "Proximity_020",
                     "Proximity_030"],
        "dense":    ["Proximity_005", "Proximity_010", "Proximity_015",
                     "Proximity_020", "Proximity_030"],
    },
    "rens": {
        "few":      ["Rens_10", "Rens_40"],
        "moderate": ["Rens_20", "Rens_40", "Rens_50"],
        "spread":   ["Rens_10", "Rens_30", "Rens_50"],
        "dense":    ["Rens_10", "Rens_20", "Rens_30", "Rens_40", "Rens_50"],
    },
    "rins": {
        "few":      ["Rins_30", "Rins_40"],
        "moderate": ["Rins_10", "Rins_30", "Rins_50"],
        "spread":   ["Rins_10", "Rins_20", "Rins_40", "Rins_50"],
        "dense":    ["Rins_10", "Rins_20", "Rins_30", "Rins_40", "Rins_50"],
    },
}

# Score presets  [best, better, accept, reject]
# Vectors observed in top-performing configurations.
SCORE_PRESETS = {
    "5_4_2_0":  [5, 4, 2, 0],    # best overall (factor importance)
    "8_4_2_1":  [8, 4, 2, 1],    # overall rank-1 (18 instance wins)
    "5_2_1_0":  [5, 2, 1, 0],    # common across datasets
    "3_2_1_0":  [3, 2, 1, 0],    # common across datasets
    "8_3_1_0":  [8, 3, 1, 0],    # common across datasets
    "16_4_2_1": [16, 4, 2, 1],   # strong on dist-miplib
}
SCORE_TS_PRESETS = {
    # Thompson Sampling needs 0/1 rewards
    "1100": [1, 1, 0, 0],
    "1110": [1, 1, 1, 0],
}

# Fallback operators when too few are selected
_FALLBACK_OPS = ["Mutation_30", "Mutation_50",
                 "Local_Branching_10", "Local_Branching_20"]


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def _instance_name(path: str) -> str:
    """Extract instance name (without extension) from a file path."""
    base = os.path.basename(str(path))
    for ext in MIP_EXTENSIONS:
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def collect_instances(instance_dir, instances_to_run=None):
    """Scan *instance_dir* for MIP files and return [(name, path), ...].

    Parameters
    ----------
    instance_dir : str
        Directory containing MIP instance files.
    instances_to_run : str or None
        Optional path to a text file listing filenames (one per line) to
        include.  If given, only files whose basename appears in this list
        are returned.

    Returns
    -------
    list of (name, path) tuples, sorted by name.
    """
    # Build whitelist from instances_to_run file if provided
    whitelist = None
    if instances_to_run and os.path.isfile(instances_to_run):
        with open(instances_to_run, "r") as f:
            whitelist = set()
            for line in f:
                entry = line.strip()
                if entry:
                    whitelist.add(entry)
                    # Also add without extension so matching is flexible
                    for ext in MIP_EXTENSIONS:
                        if entry.endswith(ext):
                            whitelist.add(entry[: -len(ext)])

    instances = []
    for fname in os.listdir(instance_dir):
        fpath = os.path.join(instance_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # Check it's a MIP file
        is_mip = any(fname.endswith(ext) for ext in MIP_EXTENSIONS)
        if not is_mip:
            continue

        # Apply whitelist filter
        if whitelist is not None:
            name = _instance_name(fpath)
            if fname not in whitelist and name not in whitelist:
                continue

        instances.append((_instance_name(fpath), fpath))

    instances.sort(key=lambda x: x[0])
    return instances


def select_representatives(instances, n=3):
    """Pick the *n* smallest (by file size) instances from a list.

    Parameters
    ----------
    instances : list of (name, path)
    n : int

    Returns
    -------
    list of (name, path), length min(n, len(instances)).
    """
    sized = [(os.path.getsize(path), name, path) for name, path in instances]
    sized.sort()
    return [(name, path) for _, name, path in sized[:n]]


# ═══════════════════════════════════════════════════════════════════
# Initial-solution cache
# ═══════════════════════════════════════════════════════════════════
def _get_initial_solution_scip(instance_path, balans_initial_sol_time_limit=20):
    from pyscipopt import Model
    model = Model("scip")
    model.readProblem(instance_path)
    sense = model.getObjectiveSense()          # "minimize" or "maximize"
    model.setParam("limits/time", balans_initial_sol_time_limit)
    start = _time.time()
    model.optimize()
    elapsed = _time.time() - start
    if (model.getNSols() < 2
            or model.getStatus() == "infeasible"
            or model.getStage() not in (9, 10)):
        return None, None, elapsed, sense
    idx_to_val = {v.getIndex(): model.getVal(v) for v in model.getVars()}
    return idx_to_val, model.getObjVal(), elapsed, sense


def _get_initial_solution_gurobi(instance_path, balans_initial_sol_time_limit=20):
    import gurobipy as grb
    from gurobipy import GRB
    model = grb.read(instance_path)
    model.Params.TimeLimit = balans_initial_sol_time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    sense = "minimize" if model.ModelSense == GRB.MINIMIZE else "maximize"
    start = _time.time()
    model.optimize()
    elapsed = _time.time() - start
    if model.SolCount < 1 or model.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
        return None, None, elapsed, sense
    idx_to_val = {v.index: v.X for v in model.getVars()}
    return idx_to_val, model.ObjVal, elapsed, sense


def get_initial_solution(instance_path, balans_initial_sol_time_limit=20, mip_solver="scip"):
    if mip_solver == "gurobi":
        return _get_initial_solution_gurobi(instance_path, balans_initial_sol_time_limit)
    return _get_initial_solution_scip(instance_path, balans_initial_sol_time_limit)


# ═══════════════════════════════════════════════════════════════════
# Trial → Balans configuration
# ═══════════════════════════════════════════════════════════════════
def _build_destroy_ops(trial):
    """Sample a destroy-operator portfolio from the Optuna *trial*.

    Crossover is always included (present in 100 % of top configs).
    Each operator family can be toggled on/off; when on, an intensity preset
    selects the specific variants.
    """
    ops = [DestroyOperators.Crossover]
    op_names_used = ["Crossover"]

    # Main families
    for family, presets in OPERATOR_FAMILIES.items():
        include = trial.suggest_categorical(f"include_{family}", [True, False])
        if include:
            intensity = trial.suggest_categorical(
                f"intensity_{family}", list(presets.keys()))
            for op_name in presets[intensity]:
                ops.append(getattr(DestroyOperators, op_name))
                op_names_used.append(op_name)

    # Guarantee at least 2 operators (MABSelector needs ≥ 2)
    if len(ops) < 2:
        for fb_name in _FALLBACK_OPS:
            fb_op = getattr(DestroyOperators, fb_name)
            if fb_op not in ops:
                ops.append(fb_op)
                op_names_used.append(fb_name)
            if len(ops) >= 2:
                break

    return ops, op_names_used


def _build_selector(trial, num_destroy):
    """Sample the MAB selector (policy + scores) from *trial*."""
    policy_name = trial.suggest_categorical(
        "learning_policy", ["thompson_sampling", "epsilon_greedy", "softmax"])

    if policy_name == "thompson_sampling":
        lp = LearningPolicy.ThompsonSampling()
        scores_key = trial.suggest_categorical("scores_ts",
                                               list(SCORE_TS_PRESETS.keys()))
        scores = SCORE_TS_PRESETS[scores_key]

    elif policy_name == "epsilon_greedy":
        eps = trial.suggest_float("epsilon", 0.02, 0.5)
        lp = LearningPolicy.EpsilonGreedy(epsilon=eps)
        scores_key = trial.suggest_categorical("scores_eg",
                                               list(SCORE_PRESETS.keys()))
        scores = SCORE_PRESETS[scores_key]

    else:  # softmax
        tau = trial.suggest_float("tau", 1.0, 3.0)
        lp = LearningPolicy.Softmax(tau=tau)
        scores_key = trial.suggest_categorical("scores_sm",
                                               list(SCORE_PRESETS.keys()))
        scores = SCORE_PRESETS[scores_key]

    if num_destroy == 1:
        return RandomSelect(num_destroy=1, num_repair=1), scores_key, scores

    selector = MABSelector(scores=scores,
                           num_destroy=num_destroy,
                           num_repair=1,
                           learning_policy=lp)
    return selector, scores_key, scores


def _build_acceptance(trial):
    """Sample the acceptance criterion from *trial*."""
    accept_name = trial.suggest_categorical(
        "acceptance", ["hill_climbing", "simulated_annealing"])

    if accept_name == "simulated_annealing":
        start_t = trial.suggest_float("sa_start_temp", 5.0, 50.0)
        end_t = trial.suggest_float("sa_end_temp", 0.1, 5.0)
        step = trial.suggest_float("sa_step", 0.05, 0.9)
        return SimulatedAnnealing(start_temperature=start_t,
                                  end_temperature=end_t,
                                  step=step), accept_name
    return HillClimbing(), accept_name


# ═══════════════════════════════════════════════════════════════════
# Objective
# ═══════════════════════════════════════════════════════════════════
def _objective(trial, *,
               representatives,      # [(name, path), ...]
               init_cache,            # {name: (idx_to_val, obj, sense)}
               balans_time_limit,
               balans_initial_sol_time_limit,
               seed,
               mip_solver):
    """Optuna objective – returns average relative improvement (maximise)."""

    destroy_ops, op_names = _build_destroy_ops(trial)
    selector, scores_key, scores = _build_selector(trial, len(destroy_ops))
    acceptance, accept_name = _build_acceptance(trial)

    balans = Balans(
        destroy_ops=destroy_ops,
        repair_ops=[RepairOperators.Repair],
        selector=selector,
        accept=acceptance,
        stop=MaxRuntime(balans_time_limit),
        seed=seed,
        mip_solver=mip_solver,
    )

    improvements = []
    for inst_name, inst_path in representatives:
        # Get / cache initial solution
        if inst_name not in init_cache:
            idx_to_val, obj, _, sense = get_initial_solution(
                inst_path, balans_initial_sol_time_limit, mip_solver)
            if idx_to_val is None:
                print(f"  [skip] No feasible init for {inst_name}")
                continue
            init_cache[inst_name] = (idx_to_val, obj, sense)

        idx_to_val, init_obj, sense = init_cache[inst_name]

        try:
            result = balans.solve(inst_path, index_to_val=idx_to_val)
        except Exception as exc:
            print(f"  [trial {trial.number}] error on {inst_name}: {exc}")
            continue

        if result is None:
            continue

        objs = list(result.statistics.objectives)
        if sense == "minimize":
            best_obj = min(objs)
            imp = (init_obj - best_obj) / max(abs(init_obj), 1e-10)
        else:
            best_obj = max(objs)
            imp = (best_obj - init_obj) / max(abs(init_obj), 1e-10)

        improvements.append(imp)

    if not improvements:
        return float("-inf")

    avg_imp = float(np.mean(improvements))
    return avg_imp


# ═══════════════════════════════════════════════════════════════════
# Tuning driver
# ═══════════════════════════════════════════════════════════════════
def tune_instances(instances,
                   output_dir,
                   n_trials=50,
                   balans_time_limit=120,
                   balans_initial_sol_time_limit=20,
                   seed=1283,
                   mip_solver="scip",
                   n_representative=3):
    """Run an Optuna study over the given instances and persist the best config.

    Parameters
    ----------
    instances : list of (name, path)
        All instances to consider.  A subset of *n_representative* smallest
        will be used for evaluation during each trial.
    output_dir : str
        Directory for output files (best_config.json, optuna.pkl).
    n_trials : int
        Number of Optuna trials.
    balans_time_limit : int
        Balans time limit per trial per instance (seconds).
    balans_initial_sol_time_limit : int
        Time for Balans to use when computing a warm-up initial solution
        (passed to SCIP/Gurobi limits/time or TimeLimit) in seconds.
    seed : int
        Random seed.
    mip_solver : str
        MIP solver backend for Balans ("scip" or "gurobi").
    n_representative : int
        Number of representative instances to evaluate per trial.

    Returns
    -------
    config : dict or None
        Best configuration found, or None if no solvable instances.
    """
    reps = select_representatives(instances, n=n_representative)
    if not reps:
        print("No solvable instances found -- skipping.")
        return None

    print(f"\n{'=' * 60}")
    print(f"Tuning Balans  "
          f"({len(instances)} total, {len(reps)} representative)")
    print(f"  Representatives: {[r[0] for r in reps]}")
    print(f"  Trials: {n_trials}  |  Time limit/trial: {balans_time_limit}s  |  "
          f"Solver: {mip_solver}")
    print("=" * 60)

    init_cache = {}  # shared across trials

    study = optuna.create_study(
        direction="maximize",
        study_name="balans_tuner",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )

    study.optimize(
        lambda trial: _objective(
            trial,
            representatives=reps,
            init_cache=init_cache,
            balans_time_limit=balans_time_limit,
            balans_initial_sol_time_limit=balans_initial_sol_time_limit,
            seed=seed,
            mip_solver=mip_solver,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n  Best trial #{best.number}  value={best.value:.6f}")
    print(f"  Params: {json.dumps(best.params, indent=4, default=str)}")

    # ── Persist ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    config = _trial_params_to_config(best.params,
                                     n_instances=len(instances),
                                     best_value=best.value,
                                     instance_names=[n for n, _ in instances])

    config_path = os.path.join(output_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"  Config saved -> {config_path}")

    study_path = os.path.join(output_dir, "optuna.pkl")
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"  Optuna study  saved -> {study_path}")

    return config


def _trial_params_to_config(params, n_instances, best_value, instance_names=None):
    """Convert raw Optuna params dict into a human-readable config dict."""

    # Reconstruct operator list — Crossover is always included
    op_names = ["Crossover"]
    for family, presets in OPERATOR_FAMILIES.items():
        if params.get(f"include_{family}"):
            intensity = params.get(f"intensity_{family}", "moderate")
            op_names.extend(presets.get(intensity, []))
    if len(op_names) < 2:
        for fb in _FALLBACK_OPS:
            if fb not in op_names:
                op_names.append(fb)
            if len(op_names) >= 2:
                break

    # Learning policy
    lp = params.get("learning_policy", "thompson_sampling")
    lp_detail = {}
    if lp == "epsilon_greedy":
        lp_detail["epsilon"] = params.get("epsilon")
    elif lp == "softmax":
        lp_detail["tau"] = params.get("tau")

    # Scores
    scores_key = (params.get("scores_ts")
                  or params.get("scores_eg")
                  or params.get("scores_sm")
                  or "1100")
    all_presets = {**SCORE_PRESETS, **SCORE_TS_PRESETS}
    scores = all_presets.get(scores_key, [1, 1, 0, 0])

    # Acceptance
    accept = params.get("acceptance", "hill_climbing")
    accept_detail = {}
    if accept == "simulated_annealing":
        accept_detail["start_temperature"] = params.get("sa_start_temp")
        accept_detail["end_temperature"] = params.get("sa_end_temp")
        accept_detail["step"] = params.get("sa_step")

    config = {
        "n_instances": n_instances,
        "best_trial_value": round(best_value, 6),
        "destroy_operators": op_names,
        "learning_policy": lp,
        "learning_policy_params": lp_detail,
        "scores": scores,
        "acceptance": accept,
        "acceptance_params": accept_detail,
        "raw_optuna_params": {k: (v if not isinstance(v, float)
                                  else round(v, 6))
                              for k, v in params.items()},
    }

    if instance_names is not None:
        config["instance_names"] = instance_names

    return config


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Optuna-based Balans configuration tuner.")
    parser.add_argument("--instance_dir", type=str,
                        default=str(DATA_DIR),
                        help="Directory containing MIP instance files")
    parser.add_argument("--instances_to_run", type=str, default=None,
                        help="Text file listing instance filenames to include "
                             "(one per line). If omitted, all MIP files in "
                             "--instance_dir are used.")
    parser.add_argument("--output_dir", type=str,
                        default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for tuned config and optuna study object")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Optuna trials (default: 50)")
    parser.add_argument("--balans_time_limit", type=int, default=120,
                        help="Balans time limit per trial in seconds (default: 120)")
    parser.add_argument("--balans_initial_sol_time_limit", type=int, default=20,
                        help="SCIP/Gurobi warm-up for initial solution (default: 20)")
    parser.add_argument("--n_representative", type=int, default=3,
                        help="Representative instances to evaluate per trial "
                             "(default: 3, smallest by file size)")
    parser.add_argument("--mip_solver", type=str, default="scip",
                        choices=["scip", "gurobi"],
                        help="MIP solver backend for Balans (default: scip)")
    parser.add_argument("--seed", type=int, default=1283,
                        help="Random seed (default: 1283)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print configuration space and exit")
    args = parser.parse_args()

    # ── Collect instances from folder ─────────────────────────────
    if not os.path.isdir(args.instance_dir):
        print(f"ERROR: Instance directory not found: {args.instance_dir}")
        sys.exit(1)

    instances = collect_instances(args.instance_dir, args.instances_to_run)

    if not instances:
        print(f"ERROR: No MIP files found in {args.instance_dir}")
        if args.instances_to_run:
            print(f"  (filtered by {args.instances_to_run})")
        sys.exit(1)

    print(f"Found {len(instances)} instances in {args.instance_dir}")
    if args.instances_to_run:
        print(f"  (filtered by {args.instances_to_run})")
    for name, path in instances:
        print(f"  {name}")

    # ── Dry run ──────────────────────────────────────────────────
    if args.dry_run:
        print("\n--- SEARCH SPACE ---")
        print("\nCrossover: always included")
        print("\nOperator families (include/exclude + intensity preset):")
        for fam, presets in OPERATOR_FAMILIES.items():
            print(f"  {fam}: {list(presets.keys())}")
            for preset_name, ops in presets.items():
                print(f"    {preset_name}: {ops}")
        print(f"\nLearning policies: thompson_sampling, epsilon_greedy, softmax")
        print(f"  epsilon_greedy.epsilon: [0.02, 0.5]")
        print(f"  softmax.tau: [1.0, 3.0]")
        print(f"\nScore presets (non-TS):")
        for name, s in SCORE_PRESETS.items():
            print(f"  {name}: {s}")
        print(f"\nScore presets (Thompson Sampling):")
        for name, s in SCORE_TS_PRESETS.items():
            print(f"  {name}: {s}")
        print(f"\nAcceptance: hill_climbing, simulated_annealing")
        print(f"  SA start_temp: [5, 50], end_temp: [0.1, 5], step: [0.05, 0.9]")
        print(f"\n--- END ---")
        return

    # ── Tune ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    config = tune_instances(
        instances=instances,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        balans_time_limit=args.balans_time_limit,
        balans_initial_sol_time_limit=args.balans_initial_sol_time_limit,
        seed=args.seed,
        mip_solver=args.mip_solver,
        n_representative=args.n_representative,
    )

    if config is not None:
        print("\nDone! Best configuration:")
        print(json.dumps(config, indent=2, default=str))
    else:
        print("\nNo configuration found (no solvable instances).")


if __name__ == "__main__":
    main()
