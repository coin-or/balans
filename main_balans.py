# ALNS for adaptive large neigborhood
from alns.accept import SimulatedAnnealing
from alns.select import MABSelector
from alns.stop import MaxIterations, MaxRuntime

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
reward_best, reward_better, reward_accept, reward_reject = 1, 1, 0, 0

# Bandit selector
selector = MABSelector(scores=[reward_best, reward_better, reward_accept, reward_reject],
                       num_destroy=len(destroy_ops),
                       num_repair=len(repair_ops),
                       learning_policy=LearningPolicy.ThompsonSampling())

# Acceptance criterion
# accept = HillClimbing()
accept = SimulatedAnnealing(start_temperature=20, end_temperature=1, step=0.1)

# Stopping condition
stop = MaxRuntime(600)
# stop = MaxIterations(10)

# Balans
balans = Balans(destroy_ops=destroy_ops,
                repair_ops=repair_ops,
                selector=selector,
                accept=accept,
                stop=stop)

# Run
instance_path = "tests/data/noswot.mps"
result = balans.solve(instance_path)

# print("Best solution:", result.best_state.solution())
print("Best objective:", result.best_state.objective())

# The installed entry point lives in balans/main.py.
# from balans.main import main
# if __name__ == "__main__":
#     main()