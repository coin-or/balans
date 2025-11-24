# ALNS for adaptive large neigborhood
from alns.accept import SimulatedAnnealing
from alns.select import MABSelector
from alns.stop import MaxIterations
# MABWiser for contextual multi-armed bandits
from mabwiser.mab import LearningPolicy

# Balans meta-solver for solving mixed integer programming problems
from balans.solver import Balans, DestroyOperators, RepairOperators

# Destroy operators
destroy_ops = [DestroyOperators.Crossover,
               # DestroyOperators.Dins,
               DestroyOperators.Mutation_25,
               DestroyOperators.Mutation_50,
               DestroyOperators.Mutation_75,
               DestroyOperators.Local_Branching_10,
               DestroyOperators.Local_Branching_25,
               DestroyOperators.Local_Branching_50,
               DestroyOperators.Proximity_005,
               DestroyOperators.Proximity_015,
               DestroyOperators.Proximity_030,
               # DestroyOperators.Random_Objective
               DestroyOperators.Rens_25,
               DestroyOperators.Rens_50,
               DestroyOperators.Rens_75,
               DestroyOperators.Rins_25,
               DestroyOperators.Rins_50,
               DestroyOperators.Rins_75]

# Repair operators
repair_ops = [RepairOperators.Repair]

# Rewards
best, better, accept, reject = 1, 1, 0, 0

# Bandit selector
selector = MABSelector(scores=[best, better, accept, reject],
                       num_destroy=len(destroy_ops),
                       num_repair=len(repair_ops),
                       learning_policy=LearningPolicy.ThompsonSampling())

# Acceptance criterion
# accept = HillClimbing()
accept = SimulatedAnnealing(start_temperature=20, end_temperature=1, step=0.1)

# Stopping condition
# stop = MaxRuntime(100)
stop = MaxIterations(10)

# Balans
balans = Balans(destroy_ops=destroy_ops,
                repair_ops=repair_ops,
                selector=selector,
                accept=accept,
                stop=stop)

# Run
instance_path = "tests/data/noswot.mps"
result = balans.solve(instance_path)

print("Best solution:", result.best_state.solution())
print("Best solution objective:", result.best_state.objective())

# Check for optimality using SCIP
# from pyscipopt import Model
# model = Model("scip")
# model.readProblem(instance_path)
# model.optimize()
# solution = model.getBestSol()
# print(solution)
# print("Optimal value:", model.getObjVal())


# Check for optimality using Gurobi
# import gurobipy as grb
# # from gurobi_onboarder import init_gurobi
# # gurobi_venv, GUROBI_FOUND = init_gurobi.initialize_gurobi()
# # gurobi_venv.setParam("OutputFlag", 1)
# # gurobi_venv.setParam("LogToConsole", 0)
# # gurobi_venv.setParam("LogFile", "asd.log")
# model = grb.read(f'{instance_path}',env=gurobi_venv)
# # model = grb.Model(env=gurobi_venv)
## model.optimize()
## print('Best answers: ')
# # for v in m.getVars():
#     # print('%s %g' % (v.varName, v.x))
## print(model.objVal)

