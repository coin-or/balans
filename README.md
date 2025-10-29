# Balans: Bandit-based Adaptive Large Neighborhood Search
**Balans** ([IJCAI'25](https://www.ijcai.org/proceedings/2025/286)) is an online-learning meta-solver designed to tackle Mixed-Integer Programming problems (MIPs) through multi-armed bandit-based adaptive large neighborhood search strategy, ALNS(MIP).

The framework integrates several powerful components into a highly configurable, modular, extensible, solver-agnostic, open-source software: 
* [MABWiser](https://github.com/fidelity/mabwiser/) for contextual multi-armed bandits
* [ALNS](https://github.com/N-Wouda/ALNS/) for adaptive large neighborhood search
* [SCIP](https://scipopt.org/) and [Gurobi](https://www.gurobi.com/) for solving mixed-integer linear programming problems. 

**ParBalans** ([Arxiv'25](https://arxiv.org/abs/2508.06736)) extends this framework with parallelization strategies at both the outer configuration level and the inner branch-and-bound level to exploit modern multi-core architectures. 

More broadly, Balans is an integration technology at the intersection of adaptive search, meta-heuristics, multi-armed bandits, and mixed integer programming. When configured with a single neighborhood, it generalizes and subsumes prior work on Large Neighborhood Search for MIP, LNS(MIP).

Balans is developed collaboratively by the AI Center of Excellence at Fidelity Investments, the University of Southern California.

## Quick Start
```python
# ALNS for adaptive large neigborhood search
from alns.select import MABSelector
from alns.accept import HillClimbing, SimulatedAnnealing
from alns.stop import MaxIterations, MaxRuntime

# MABWiser for contextual multi-armed bandits
from mabwiser.mab import LearningPolicy

# Balans meta-solver for solving mixed integer programming problems
from balans.solver import Balans, DestroyOperators, RepairOperators

# Destroy operators
destroy_ops = [DestroyOperators.Crossover,
               DestroyOperators.Dins,
               DestroyOperators.Mutation_25,
               DestroyOperators.Local_Branching_10,
               DestroyOperators.Rins_25,
               DestroyOperators.Proximity_05,
               DestroyOperators.Rens_25,
               DestroyOperators.Random_Objective]

# Repair operators
repair_ops = [RepairOperators.Repair]

# Rewards for online learning feedback loop
best, better, accept, reject = 4, 3, 2, 1

# Bandit selector
selector = MABSelector(scores=[best, better, accept, reject],
                       num_destroy=len(destroy_ops),
                       num_repair=len(repair_ops),
                       learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50))

# Acceptance criterion
# accept = HillClimbing() # for pure exploitation 
accept = SimulatedAnnealing(start_temperature=20, end_temperature=1, step=0.1)

# Stopping condition
# stop = MaxRuntime(100)
stop = MaxIterations(10)

# Balans
balans = Balans(destroy_ops=destroy_ops,
                repair_ops=repair_ops,
                selector=selector,
                accept=accept,
                stop=stop,
                mip_solver="scip") # "gurobi"

# Run a mip instance to retrieve results 
instance_path = "data/miplib/noswot.mps"
result = balans.solve(instance_path)

# Results of the best found solution and the objective
print("Best solution:", result.best_state.solution())
print("Best solution objective:", result.best_state.objective())
```

## Quick Start - ParBalans
```python
# Parallel version of Balans, that runs several configurations parallely
from balans.solver import ParBalans
from alns.stop import MaxIterations

# ParBalans to run different Balans configs in parallel and save results
parbalans = ParBalans(n_jobs=2,                 # Outer-level: parallel Balans configurations
                      n_mip_jobs=1,             # Inner-level: parallel BnB search. Only supported by Gurobi solver
                      mip_solver="scip",
                      output_dir="results/", 
                      stop=MaxIterations(10))   # Stop criteria per each run

# Run a mip instance to retrieve several results 
instance_path = "data/miplib/noswot.mps"
best_solution, best_objective = parbalans.run(instance_path)

# Results of the best found solution and the objective
print("Best solution:", best_solution)
print("Best solution objective:", best_objective)
```

## Available Destroy Operators
* Dins[^1] 
[^1]: S. Ghosh. DINS, a MIP Improvement Heuristic. Integer Programming and Combinatorial Optimization: IPCO, 2007.
* Local Branching[^2]
[^2]: M. Fischetti and A. Lodi. Local branching. Mathematical Programming, 2003.
* Mutation[^3]
[^3]: Rothberg. An Evolutionary Algorithm for Polishing Mixed Integer Programming Solutions. INFORMS Journal on Computing, 2007.
* Rens[^4]
[^4]: Berthold. RENS–the optimal rounding. Mathematical Programming Computation, 2014.
* Rins[^5]
[^5]: E. Danna, E. Rothberg, and C. L. Pape. Exploring relaxation induced neighborhoods to improve MIP solutions. Mathematical Programming, 2005.
* Random Objective[^6]
[^6]: Random Objective.
* Proximity Search[^7]
[^7]: M. Fischetti and M. Monaci. Proximity search for 0-1 mixed-integer convex programming. Journal of Heuristics, 20(6):709–731, Dec 2014.
* Crossover[^8]
[^8]: E. Rothberg. An Evolutionary Algorithm for Polishing Mixed Integer Programming Solutions. INFORMS Journal on Computing, 19(4):534–541, 2007.

## Available Repair Operators
* Repair MIP

## Installation
Balans requires Python 3.10+ can be installed from PyPI via `pip install balans`. 

## Test Your Setup
```
$ cd balans
$ python -m unittest discover tests
```

## Citation
If you use Balans in a publication, please cite it as:

```bibtex
  @inproceedings{balans,
    title     = {Balans: Multi-Armed Bandits-based Adaptive Large Neighborhood Search for Mixed-Integer Programming Problems},
    author    = {Cai, Junyang and Kadıoğlu, Serdar and Dilkina, Bistra},
    booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
    publisher = {International Joint Conferences on Artificial Intelligence Organization},
    editor    = {James Kwok},
    pages     = {2566--2574},
    year      = {2025},
    month     = {8},
    note      = {Main Track},
    doi       = {10.24963/ijcai.2025/286},
    url       = {https://doi.org/10.24963/ijcai.2025/286},
  }

  @misc{parbalans,
        title={ParBalans: Parallel Multi-Armed Bandits-based Adaptive Large Neighborhood Search}, 
        author={Alican Yilmaz and Junyang Cai and Serdar Kadıoğlu and Bistra Dilkina},
        year={2025},
        eprint={2508.06736},
        archivePrefix={arXiv},
        primaryClass={cs.AI},
        url={https://arxiv.org/abs/2508.06736}, 
  }
```

## License

Balans is licensed under the [Apache License 2.0](LICENSE).

<br>
