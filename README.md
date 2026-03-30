[![A COIN-OR Project](https://coin-or.github.io/coin-or-badge.png)](https://www.coin-or.org) [![ci](https://github.com/skadio/balans/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/skadio/balans/actions/workflows/ci.yml) [![PyPI version fury.io](https://badge.fury.io/py/balans.svg)](https://pypi.python.org/pypi/balans/) [![PyPI license](https://img.shields.io/pypi/l/balans.svg)](https://pypi.python.org/pypi/balans/) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Downloads](https://static.pepy.tech/personalized-badge/balans?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/balans)

# Balans: Bandit-based Adaptive Large Neighborhood Search
**Balans** ([IJCAI'25](https://www.ijcai.org/proceedings/2025/286)) is an online-learning meta-solver designed to tackle Mixed-Integer Programming problems (MIPs) through multi-armed bandit-based adaptive large neighborhood search strategy, ALNS(MIP).

The framework integrates several powerful components into a highly configurable, modular, extensible, solver-agnostic, open-source software: 
* [MABWiser](https://github.com/fidelity/mabwiser/) for contextual multi-armed bandits
* [ALNS](https://github.com/N-Wouda/ALNS/) for adaptive large neighborhood search
* [SCIP](https://scipopt.org/) and [Gurobi](https://www.gurobi.com/) for solving mixed-integer linear programming problems. 

**ParBalans** ([Arxiv'25](https://arxiv.org/abs/2508.06736)) extends this framework with parallelization strategies at both the outer configuration level and the inner branch-and-bound level to exploit modern multicore architectures. 

More broadly, Balans is an integration technology at the intersection of adaptive search, meta-heuristics, multi-armed bandits, and mixed integer programming. When configured with a single neighborhood, it generalizes and subsumes prior work on Large Neighborhood Search for MIP, LNS(MIP). A detailed description of the framework, algorithms, and experimental results can be found in our [IJCAI'25 presentation](https://nbviewer.org/github/skadio/skadio.github.io/blob/master/files/2025_IJCAI_Balans_Kadioglu.pdf).

Balans is a collaborative effort between academia and industry, developed by Brown University, the University of Southern California, and the AI Center of Excellence at Fidelity Investments. The project is contributed to the [COIN-OR Foundation](https://www.coin-or.org/).

## Quick Start - Balans
```python
# Balans meta-solver for solving mixed-integer programming problems
from balans.solver import Balans

# Balans with default configuration
balans = Balans()

# Solve
result = balans.solve("mip_instance.mps")

# Results
print("Best solution:", result.best_state.solution())
print("Best solution objective:", result.best_state.objective())
```

To supply a custom JSON configuration file (e.g., [`default.json`](https://github.com/coin-or/balans/blob/main/balans/configs/default.json))
```python
from balans.solver import Balans
balans = Balans(config="/path/to/config.json")
```

To run directly from the command line after `pip install balans`:
```bash
> balans mip_instance.mps
> balans mip_instance.mps config.json
```

To run programmatically with a custom configuration, see [`main_balans.py`](https://github.com/coin-or/balans/blob/main/main_balans.py).

## How does Balans work?
Balans is a meta-solver that sits on top of a MIP solver (e.g., SCIP, Gurobi) and iteratively improves the solution to a MIP instance by applying a sequence of destroy and repair operators. The selection of these operators is guided by a multi-armed bandit algorithm that learns which operators are most effective at improving the solution over time.

### Intended Use Cases
Balans, as a meta-heuristic, _cannot provide optimality guarantees_. Balans is intended for finding good solutions for **extremely challenging instances** that cannot be solved to optimality with MIP solvers or cannot be solved quickly with heuristics to obtain good solutions.

If you have a challenging MIP instance that _cannot be solved to optimality with a MIP solver_, or _does not admit high-quality heuristic solutions in short amount of time_, then Balans is your friend! 🤗 

### 1) Create a Balans solver
You can create a Balans solver using 
1. Default configuration as `balans = Balans()`.
1. JSON configuration file as `balans = Balans(config="/path/to/config.json")`.
2. Constructor settings as `balans = Balans(destroy_ops, repair_ops, accept, stop, ...)`). 
 
The parameters specify which the destroy and repair operators to consider, the rewards and learning policy for the multi-armed bandit algorithm, the acceptance criteria for neighborhood exploration, the stopping condition, and other settings such as time limits and the mip solver to use.

### 2) Run the Balans solver
When you run `balans.solve("mip_instance.mps")` Balans reads the MIP instance from the specified path using the built-in reading capability of the underlying MIP solver. Then, it attempts to (1) find an initial solution, and (2), improve it until stop. 

#### Finding an initial solution 
First, Balans tries to find an initial solution running the mip solver with `timelimit_first_solution` seconds. Notice that this might find more than one solution until hitting the time limit and the best found solution becomes the initial solution. 

You can skip this step by providing an initial solution dictionary to Balans via `balans.solve("mip_instance.mps", index_to_val)`. This sets variable values according to `index_to_val` before starting the search. This can even be _partial_ to guide the MIP solver in its search for the initial solution. 

In case finding an initial solution fails, Balans will restart the MIP solver to find a _single feasible solution_ within the remaining the time limit. If that also fails within the timelimit, Balans will terminate without any solution. 😢

#### Improving the solution
After obtaining an initial solution, Balans will enter the main ALSN search loop and iteratively apply destroy and repair operators to improve the solution. The multi-armed bandit algorithm will update its beliefs about which operators are most effective based on the observed improvements in the solution. 

This loop runs until the specified stop condition is met (e.g., maximum number of iterations, time limit, or no improvement). 

The runtime for each ALNS iteration is limited by `timelimit_alns_iteration` seconds, except for local branching operator, which is a costlier operator than others, is limited by `timelimit_local_branching_iteration` seconds. The crossover operator requires finding a random solution to crossover with the current solution, and this is limited by `timelimit_crossover_random_feasible` seconds. The proximity operator, which alters the original objective function, uses `big_m` to avoid infeasibility. 

## Quick Start - ParBalans
```python
# Parallel version of Balans, that runs several configurations parallely
from balans.solver import ParBalans

if __name__ == '__main__':

    # ParBalans to run different Balans configs in parallel and save results
    parbalans = ParBalans(n_jobs=2,           # Outer-level: parallel Balans configurations
                          n_mip_jobs=1,       # Inner-level: parallel BnB search. Only supported by Gurobi solver
                          mip_solver="scip",
                          output_dir="parbalans_results/",
                          balans_generator=ParBalans.TOP_CONFIGS)

    # Run a mip instance to retrieve several results 
    instance_path = "mip_instance.mps"
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
**Balans** requires Python 3.10+ can be installed from PyPI via `pip install balans`.
once installed, it can be used from the command line via `balans /path/to/problem.mps`
More details in [INSTALL](INSTALL). 

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
