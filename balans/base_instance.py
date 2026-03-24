from typing import Tuple, Dict, Any

from balans.base_mip import _BaseMIP
from balans.utils import Constants, timestamp


class _Instance:
    """
    Instance from a given MIP file with solve operations on top, subject to operator
    """

    def __init__(self, mip: _BaseMIP, seed=Constants.default_seed):
        # MIP Model initialized from the original mip instance
        self.mip: _BaseMIP = mip
        self.seed = seed

        # Static, set once and for all
        self.discrete_indexes = None    # all discrete: binary + integer
        self.binary_indexes = None      # discrete and binary
        self.integer_indexes = None     # discrete but not binary
        self.lp_index_to_val = None
        self.lp_obj_val = None
        self.lp_floating_discrete_indexes = None

    def initial_solve(self, index_to_val=None) -> Tuple[Dict[Any, float], float]:

        # Extracts static instance features (sense must be minimization now, after adjustment)
        self.discrete_indexes, self.binary_indexes, self.integer_indexes = self.mip.extract_indexes()

        # Extract static lp features for root instance
        self.lp_index_to_val, self.lp_obj_val, self.lp_floating_discrete_indexes = self.mip.extract_lp(self.discrete_indexes)

        # If a solution is given fix it. It can be partial (denoted by None value). Appends mip.constraints
        self.mip.fix_vars(index_to_val)

        # Solve with some time limit to get an initial solution
        index_to_val, obj_val = self.mip.solve_and_undo(time_limit_in_sc=Constants.timelimit_first_solution)

        # If no feasible initial solution found within time limit
        if len(index_to_val) == 0:
            # Solve for the first feasible solution without time limit and without fixing the given solution
            index_to_val, obj_val = self.mip.solve_and_undo(solution_limit=1)

        # Return solution
        return index_to_val, obj_val

    def solve(self,
              index_to_val=None,
              obj_val=None,
              destroy_set=None,
              dins_set=None,
              rens_float_set=None,
              has_random_obj=False,
              local_branching_size=0,
              proximity_delta=0) -> Tuple[Dict[Any, float], float]:

        print(f"{timestamp()} \t Solve")

        # has_destroy to identify if any constraint added or objective function changed
        # If has_destroy = True, optimize the problem and get new sol and obj
        # If has_destroy = False, return the current sol and obj, do not optimize
        has_destroy = False

        # Starting solution and objective value
        starting_index_to_val = index_to_val
        starting_obj_val = obj_val

        # DESTROY used for Crossover, Mutation, RINS
        if destroy_set:
            if len(destroy_set) > 0:
                has_destroy = True
                self.mip.fix_vars(index_to_val, skip_indexes=destroy_set)

        # DINS: Discrete Variables, where incumbent and lp relaxation have distance more than 0.5
        if dins_set:
            if len(dins_set) > 0:
                has_destroy = True
                self.mip.dins(index_to_val, dins_set, self.lp_index_to_val)

        # Local Branching: Binary variables, flip a limited subset (can come from DINS with delta)
        if local_branching_size > 0:
            has_destroy = True
            self.mip.local_branching(index_to_val, local_branching_size, self.binary_indexes)

        # Proximity: Binary variables, modify objective, add new constraint
        if proximity_delta > 0:
            has_destroy = True
            # print("proximity_delta: ", proximity_delta)
            self.mip.proximity(index_to_val, obj_val, proximity_delta, self.binary_indexes)

        # RENS: Discrete variables, where the lp relaxation is not integral
        if rens_float_set:
            if len(rens_float_set) > 0:
                has_destroy = True
                self.mip.rens(index_to_val, rens_float_set, self.lp_index_to_val)

        # Random Objective
        if has_random_obj:
            has_destroy = True
            self.mip.random_objective()

        # If no destroy, don't solve, quit with previous objective
        # e.g. when destroy set is empty.
        if not has_destroy:
            print(f"{timestamp()} \t No destroy to apply, don't call optimize()")
            print(f"{timestamp()} \t Current Obj: {starting_obj_val}")
            # print("\t starting_index_to_val: ", starting_index_to_val)
            return starting_index_to_val, starting_obj_val

        # Solve mip and undo with some timelimit
        # Set time limit per alns iteration or overwrite with local branching iteration
        time_limit = Constants.timelimit_alns_iteration
        if local_branching_size:
            time_limit = Constants.timelimit_local_branching_iteration
        index_to_val, obj_val = self.mip.solve_and_undo(time_limit_in_sc=time_limit)

        # If no solution found, go back
        if len(index_to_val) == 0:
            print(f"{timestamp()} \t No solution found, go back to previous state")
            # print("\t Current Obj:", starting_obj_val)
            return starting_index_to_val, starting_obj_val

        # Solution found but for transformed objectives (random_obj and proximity), find the original obj value
        if proximity_delta > 0 or has_random_obj:
            # Objective value of the solution found in transformed
            print(f"{timestamp()} \t Transformed obj: {obj_val}")
            obj_val = self.mip.get_obj_value(index_to_val)

        print(f"{timestamp()} \t Solve DONE! {obj_val}")
        return index_to_val, obj_val
