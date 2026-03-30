from balans.solver import ParBalans

# > **Note:** The `if __name__ == '__main__':` guard is **required** on Windows
# (and any platform that uses the *spawn* start method for multiprocessing).
# Without it, each worker process re-imports the script, creating an infinite loop of new processes.
if __name__ == '__main__':

    # ParBalans to run different Balans configs in parallel and save results
    parbalans = ParBalans(n_jobs=2,           # Outer-level: parallel Balans configurations
                          n_mip_jobs=1,       # Inner-level: parallel BnB search. Only supported by Gurobi solver
                          mip_solver="scip",
                          output_dir="parbalans_results/",
                          balans_generator=ParBalans.TOP_CONFIGS)

    # Run a mip instance to retrieve several results
    instance_path = "tests/data/pk1.mps"
    best_solution, best_objective = parbalans.run(instance_path)

    # Results of the best found solution and the objective
    print("Best solution:", best_solution)
    print("Best solution objective:", best_objective)
