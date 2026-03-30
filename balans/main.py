import sys
import numpy as np
from balans.solver import Balans


def main():
    """Console script entry point: balans <instance_path> [config_path]"""
    if len(sys.argv) < 2:
        print("Usage: balans <instance_path> [path_to_config.json]")
        print("Example: balans problem.mps")
        print("Example: balans problem.mps config.json")
        sys.exit(1)

    instance_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    balans = Balans(config=config_path)
    result = balans.solve(instance_path)

    # print("Best solution:", result.best_state.solution())
    print("Best solution objective:", result.best_state.objective())
    print("Objective trace:", result.statistics.objectives)
    print("Runtime trace:", result.statistics.runtimes)
    print("Runtime trace (cumulative):", np.cumsum(result.statistics.runtimes))
    print("Operator counts:", result.statistics.destroy_operator_counts)

if __name__ == "__main__":
    main()

