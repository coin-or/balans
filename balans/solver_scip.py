import math
import random
from typing import Dict, List, Tuple, Any

import pyscipopt as scip
from pyscipopt import quicksum, Expr, SCIP_PARAMEMPHASIS

from balans.base_mip import _BaseMIP
from balans.utils import Constants


class _SCIP(_BaseMIP):

    def __init__(self, instance_path: str, n_mip_jobs: int, seed: int, big_m: float = Constants.M):
        super().__init__(seed, big_m)

        # Set Scip model, variables and objective
        self.model = scip.Model()
        self.model.hideOutput()
        self.model.readProblem(instance_path)
        self.model.setParam("limits/maxorigsol", 0)
        self.model.setParam("randomization/randomseedshift", self.seed)
        # self.model.setParam("parallel/maxnthreads", n_mip_jobs)

        # Set variables
        self.variables = self.model.getVars()

        # Set original objective and flag if we changed from max to min.
        # We always minimize
        self.org_objective_fn = self.model.getObjective()
        self.org_objective_sense = self.model.getObjectiveSense()
        self.is_obj_sense_changed = False
        if self.org_objective_sense == Constants.maximize:
            self.model.setObjective(-self.org_objective_fn)
            self.model.setMinimize()
            self.is_obj_sense_changed = True

        # Set constraints, proximity z, and a flag for objective transformation
        # These are used for incremental solve and undo solve
        self.constraints = []
        self.proximity_z = None
        self.is_obj_transformed = False

    def get_obj_value(self, index_to_val) -> float:
        obj_val = 0
        for key, item in self.org_objective_fn.terms.items():
            obj_val += item * index_to_val[key[0].getIndex()]

        if self.is_obj_sense_changed:
            obj_val = -obj_val

        return obj_val

    def extract_indexes(self) -> Tuple[List[Any], List[Any], List[Any]]:

        # Set indexes
        discrete_indexes = []
        binary_indexes = []
        integer_indexes = []

        for var in self.variables:
            if self.is_discrete(var.vtype()):
                discrete_indexes.append(var.getIndex())
                if self.is_binary(var.vtype()):
                    binary_indexes.append(var.getIndex())
                else:
                    integer_indexes.append(var.getIndex())

        return discrete_indexes, binary_indexes, integer_indexes

    def extract_lp(self, discrete_indexes) -> Tuple[Dict[Any, float], float, List[Any]]:
        lp_index_to_val, lp_obj_val = self.solve_lp_and_undo()
        lp_floating_discrete_indexes = [i for i in discrete_indexes if not lp_index_to_val[i].is_integer()]
        return lp_index_to_val, lp_obj_val, lp_floating_discrete_indexes

    def fix_vars(self, index_to_val, skip_indexes=None):
        # Thinking problems: Do we always fix all the continuous variables?

        # If no solution given, do nothing
        if index_to_val is None:
            return

        # Walk the variables
        for var in self.variables:
            # If a value for this variable is not given, do nothing
            if index_to_val[var.getIndex()] is None:
                continue
            # if skip list given, and the variable is in there, do nothing
            if skip_indexes and var.getIndex() in skip_indexes:
                continue

            # Variable has a value, and it's not in the skip set, FIX
            self.constraints.append(self.model.addCons(var == index_to_val[var.getIndex()]))

    def dins(self, index_to_val, dins_set, lp_index_to_val) -> None:
        for var in self.variables:
            if var.getIndex() in dins_set:
                # Add bounding constraint around initial lp solution
                index = var.getIndex()
                current_lp_diff = abs(index_to_val[index] - lp_index_to_val[index])
                self.constraints.append(self.model.addCons(abs(var - lp_index_to_val[index]) <= current_lp_diff))
            else:
                # fix the variable
                self.constraints.append(self.model.addCons(var == index_to_val[var.getIndex()]))

    def local_branching(self, index_to_val, local_branching_size, binary_indexes) -> None:
        # Only change a subset of the binary variables, keep others fixed. e.g.,
        zero_binary_vars, one_binary_vars = self.split_binary_vars(self.variables, binary_indexes, index_to_val)

        # if current binary var is 0, flip to 1 consumes 1 unit of budget
        # if current binary var is 1, flip to 0 consumes 1 unit of budget by (1-x)
        zero_expr = quicksum(zero_var for zero_var in zero_binary_vars)
        one_expr = quicksum(1 - one_var for one_var in one_binary_vars)
        self.constraints.append(self.model.addCons(zero_expr + one_expr <= local_branching_size))

    def proximity(self, index_to_val, obj_val, proximity_delta, binary_indexes) -> None:
        # Set the flag so we can undo
        self.is_obj_transformed = True

        zero_binary_vars, one_binary_vars = self.split_binary_vars(self.variables, binary_indexes, index_to_val)
        # if x_inc=0, new objective expression is x_inc.
        # if x_inc=1, new objective expression is 1 - x_inc.
        # Drop all other vars (when not in the expr it is set to 0 by default)
        zero_expr = quicksum(zero_var for zero_var in zero_binary_vars)
        one_expr = quicksum(1 - one_var for one_var in one_binary_vars)

        # add cutoff constraint depending on sense, so that next state is better quality
        # a slack variable z to prevent infeasible solution, \theta = 1
        self.proximity_z = self.model.addVar(vtype=Constants.continuous, lb=0)
        # Use obj_val - delta * |obj_val| so the cutoff always tightens
        # (moves in the improvement direction), even when obj_val < 0.
        cutoff = obj_val - proximity_delta * abs(obj_val)
        self.constraints.append(self.model.addCons(self.model.getObjective() <=
                                                   cutoff + self.proximity_z))

        # M * z is to make sure model does not use z, unless needed to avoid infeasibility
        self.model.setObjective(zero_expr + one_expr + self.big_m * self.proximity_z, Constants.minimize)

    def rens(self, index_to_val, rens_float_set, lp_index_to_val) -> None:
        for var in self.variables:
            if var.getIndex() in rens_float_set:
                # Restrict discrete vars to round up and down integer version of the lp
                # EX: If var = 3.5, the constraint is var >= 3 and var <= 4
                self.constraints.append(self.model.addCons(var >= math.floor(lp_index_to_val[var.getIndex()])))
                self.constraints.append(self.model.addCons(var <= math.ceil(lp_index_to_val[var.getIndex()])))
            else:
                # If not in the set, fix the var to the current state
                self.constraints.append(self.model.addCons(var == index_to_val[var.getIndex()]))

    # TODO: modify random_objective to be consistent with Gurobi version,
    #  have a delta value which control how much coeffs you want to keep, and zero out others
    # SK: may be we can parameterize these? So there can be a few different ways to randomize? and we don't loose code
    def random_objective(self) -> None:
        # Set the flag so we can undo
        self.is_obj_transformed = True

        objective = Expr()
        for var in self.variables:
            coeff = random.uniform(-1, 1)
            if coeff != 0:
                objective += coeff * var
        objective.normalize()
        self.model.setObjective(objective, Constants.minimize)

    def solve_and_undo(self,
                       time_limit_in_sc=None,
                       solution_limit=None,
                       is_feasibility_focus=False) -> Tuple[Dict[Any, float], float]:

        # Set limits
        if time_limit_in_sc is not None:
            self.model.setParam("limits/time", time_limit_in_sc)
        if solution_limit is not None:
            self.model.setParam("limits/bestsol", solution_limit)
        if is_feasibility_focus:
            self.model.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)
            self.model.setParam("heuristics/feaspump/freq", 1)
            self.model.setParam("heuristics/feaspump/maxloops", 10)

        # Solve
        # TODO: concurrent scip needs work
        # if self.model.getParam("parallel/maxnthreads") > 1:
        #     self.model.solveConcurrent()
        # else:
        self.model.optimize()

        # Get solution
        index_to_val, obj_val = self.get_index_to_val_and_objective()

        # Free the model
        self.model.freeTransform()

        # Undo limits
        if time_limit_in_sc is not None:
            self.model.setParam("limits/time", 1e+20)
        if solution_limit is not None:
            self.model.setParam("limits/bestsol", -1)
        if is_feasibility_focus:
            self.model.setEmphasis(SCIP_PARAMEMPHASIS.DEFAULT)
            self.model.setParam("heuristics/feaspump/freq", -1)
            self.model.setParam("heuristics/feaspump/maxloops", -1)

        # Remove constraints, and reset
        for ct in self.constraints:
            self.model.delCons(ct)
        self.constraints = []

        # Reset to original objective (random objective and proximity change objective)
        if self.is_obj_transformed:
            self.is_obj_transformed = False

            # Reset back to original minimize objective
            if self.is_obj_sense_changed:
                self.model.setObjective(-self.org_objective_fn, Constants.minimize)
            else:
                self.model.setObjective(self.org_objective_fn, Constants.minimize)

            # if proximity_z variable is added, remove it
            if self.proximity_z:  # if proximity_delta > 0
                self.model.delVar(self.proximity_z)
                self.proximity_z = None
            else:  # if random heuristic used, reset heuristics
                self.model.setHeuristics(scip.SCIP_PARAMSETTING.DEFAULT)

        # Return solution
        return index_to_val, obj_val

    def solve_random_and_undo(self,
                              time_limit_in_sc=None,
                              solution_limit=None,
                              is_feasibility_focus=False) -> Tuple[Dict[Any, float], float]:

        # Set limits
        if time_limit_in_sc is not None:
            self.model.setParam("limits/time", time_limit_in_sc)
        if solution_limit is not None:
            self.model.setParam("limits/bestsol", solution_limit)
        if is_feasibility_focus:
            self.model.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)
            self.model.setParam("heuristics/feaspump/freq", 1)
            self.model.setParam("heuristics/feaspump/maxloops", 10)

        self.random_objective()

        # Solve
        # TODO: concurrent scip needs work
        # if self.model.getParam("parallel/maxnthreads") > 1:
        #     self.model.solveConcurrent()
        # else:
        self.model.optimize()

        r1_index_to_val, r1_obj_val = self.get_index_to_val_and_objective()

        # Get back the original model
        self.model.freeTransform()

        # Undo limits
        if time_limit_in_sc is not None:
            self.model.setParam("limits/time", 1e+20)
        if solution_limit is not None:
            self.model.setParam("limits/bestsol", -1)
        if is_feasibility_focus:
            self.model.setEmphasis(SCIP_PARAMEMPHASIS.DEFAULT)
            self.model.setParam("heuristics/feaspump/freq", -1)
            self.model.setParam("heuristics/feaspump/maxloops", -1)

        # Reset to original objective
        self.is_obj_transformed = False

        # Reset back to original minimize objective
        if self.is_obj_sense_changed:
            self.model.setObjective(-self.org_objective_fn, Constants.minimize)
        else:
            self.model.setObjective(self.org_objective_fn, Constants.minimize)

        return r1_index_to_val, r1_obj_val

    def solve_lp_and_undo(self) -> Tuple[Dict[Any, float], float]:
        # Solve LP relaxation and save it
        int_index = []
        bin_index = []
        count = 0
        variables = self.model.getVars()
        for var in variables:
            if self.is_binary(var.vtype()):
                self.model.chgVarType(var, Constants.continuous)
                bin_index.append(count)
            elif self.is_discrete(var.vtype()):
                self.model.chgVarType(var, Constants.continuous)
                int_index.append(count)
            count += 1

        # Solve
        # TODO: concurrent scip needs work
        # if self.model.getParam("parallel/maxnthreads") > 1:
        #     self.model.solveConcurrent()
        # else:
        self.model.optimize()

        lp_index_to_val, lp_obj_val = self.get_index_to_val_and_objective()

        # Get back the original model
        self.model.freeTransform()
        count = 0
        for var in variables:
            if count in int_index:
                self.model.chgVarType(var, Constants.integer)
            if count in bin_index:
                self.model.chgVarType(var, Constants.binary)
            count += 1

        return lp_index_to_val, lp_obj_val

    def get_index_to_val_and_objective(self) -> Tuple[Dict[Any, float], float]:
        # we check if the optimized model has solutions, feasible, and is in the solved state
        if self.model.getNSols() == 0 or self.model.getStatus() == "infeasible" or (
                self.model.getStage() != 9 and self.model.getStage() != 10):
            return dict(), 9999999
        else:
            index_to_val = dict([(var.getIndex(), self.model.getVal(var)) for var in self.model.getVars()])
            obj_value = self.model.getObjVal()
            return index_to_val, obj_value

    @staticmethod
    def is_discrete(var_type) -> bool:
        return var_type in (Constants.binary, Constants.integer)

    @staticmethod
    def is_binary(var_type) -> bool:
        return var_type in Constants.binary

    @staticmethod
    def split_binary_vars(variables, binary_indexes, index_to_val) -> Tuple[List[Any], List[Any]]:
        zero_binary_vars = []
        one_binary_vars = []
        for var in variables:
            if var.getIndex() in binary_indexes:
                if math.isclose(index_to_val[var.getIndex()], 0.0):
                    zero_binary_vars.append(var)
                else:
                    one_binary_vars.append(var)

        return zero_binary_vars, one_binary_vars
