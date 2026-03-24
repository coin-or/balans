import sys

from balans.solver import Balans


def main():
    """Console script entry point: balans <instance_path>"""
    if len(sys.argv) < 2:
        print("Usage: balans <instance_path>")
        print("Example: balans problem.mps")
        sys.exit(1)

    instance_path = sys.argv[1]

    balans = Balans()
    result = balans.solve(instance_path)

    # print("Best solution:", result.best_state.solution())
    print("Best solution objective:", result.best_state.objective())


if __name__ == "__main__":
    main()

