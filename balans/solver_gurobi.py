import math
import random
from typing import Tuple, Dict, Any, List

import gurobipy as gp
from gurobipy import GRB, quicksum

from balans.base_mip import _BaseMIP
from balans.utils import Constants


class _Gurobi(_BaseMIP):

    def __init__(self, instance_path: str, n_mip_jobs: int, seed: int, big_m: float = Constants.M):
        super().__init__(seed, big_m)

        # Create Gurobi model
        self.model = gp.read(instance_path)
        self.model.Params.OutputFlag = 0
        self.model.Params.Seed = self.seed
        self.model.Params.Threads = n_mip_jobs
        # self.model.Params.Presolve = 0 # turns of presolve

        # Set variables
        self.variables = self.model.getVars()

        # Set original objective and flag if we changed from max to min.
        # We always minimize
        self.org_objective_fn = self.model.getObjective()
        self.org_objective_sense = self.model.ModelSense  # 1 for minimize, -1 for maximize
        self.is_obj_sense_changed = False
        if self.org_objective_sense == -1:
            self.model.setObjective(-1 * self.org_objective_fn, GRB.MINIMIZE)
            self.is_obj_sense_changed = True

        # Set constraints, proximity z, and a flag for objective transformation
        # These are used for incremental solve and undo solve
        self.constraints = []
        self.proximity_z = None
        self.is_obj_transformed = False

        # Gurobi specific: this auxiliary var is needed for Gurobi to handle abs() constraint in dins operator.
        self.dins_abs_var = None

    def get_obj_value(self, index_to_val) -> float:
        expr = self.org_objective_fn
        obj_val = 0
        for i in range(expr.size()):
            var = expr.getVar(i)
            coeff = expr.getCoeff(i)
            obj_val += coeff * index_to_val[var.index]

        if self.is_obj_sense_changed:
            obj_val = -obj_val

        return obj_val

    def extract_indexes(self) -> Tuple[List[Any], List[Any], List[Any]]:
        # Set indexes
        discrete_indexes = []
        binary_indexes = []
        integer_indexes = []

        for var in self.variables:
            # Variable types naming different
            if self.is_discrete(var.VType):
                discrete_indexes.append(var.index)
                if self.is_binary(var.VType):
                    binary_indexes.append(var.index)
                else:
                    integer_indexes.append(var.index)

        return discrete_indexes, binary_indexes, integer_indexes

    def extract_lp(self, discrete_indexes) -> Tuple[Dict[Any, float], float, List[Any]]:
        lp_index_to_val, lp_obj_val = self.solve_lp_and_undo()
        lp_floating_discrete_indexes = [i for i in discrete_indexes if not math.isclose(lp_index_to_val[i] % 1, 0.0)]
        return lp_index_to_val, lp_obj_val, lp_floating_discrete_indexes

    def fix_vars(self, index_to_val, skip_indexes=None) -> None:
        # Thinking problems: Do we always fix all the continuous variables?
        
        # If no solution given, do nothing
        if index_to_val is None:
            return

        # Walk the variables
        for var in self.variables:
            # If a value for this variable is not given, do nothing
            if var.index not in index_to_val or index_to_val[var.index] is None:
                continue
            # if skip list given, and the variable is in there, do nothing
            if skip_indexes and var.index in skip_indexes:
                continue
                
            # Variable has a value, and it's not in the skip set, FIX
            self.constraints.append(self.model.addConstr(var == index_to_val[var.index]))

    def dins(self, index_to_val, dins_set, lp_index_to_val) -> None:
        for var in self.variables:
            index = var.index
            if index in dins_set:
                # Add bounding constraint around initial lp solution
                current_lp_diff = abs(index_to_val[index] - lp_index_to_val[index])
                # Create an auxiliary variable for the absolute value
                # Same constraint as in SCIP except that it is different to  represent the absolute value.
                # In SCIP, we use abs() function
                # In Gurobi, we split the absolute into two constraints and an additional variables.
                # Standard way to linearize the abs function
                self.dins_abs_var = self.model.addVar(lb=0.0, name=f'abs_{var.VarName}')
                self.model.update()  # Update model to include new variable
                self.constraints.append(self.model.addConstr(self.dins_abs_var >= var - lp_index_to_val[index]))
                self.constraints.append(self.model.addConstr(self.dins_abs_var >= -(var - lp_index_to_val[index])))
                self.constraints.append(self.model.addConstr(self.dins_abs_var <= current_lp_diff))
            else:
                # fix the variable
                self.constraints.append(self.model.addConstr(var == index_to_val[index]))

    def local_branching(self, index_to_val, local_branching_size, binary_indexes) -> None:
        zero_binary_vars, one_binary_vars = self.split_binary_vars(self.variables, binary_indexes, index_to_val)
        # For zero_binary_vars, we add x_i
        # For one_binary_vars, we add (1 - x_i)
        zero_expr = quicksum(zero_binary_vars)
        one_expr = quicksum(1 - var for var in one_binary_vars)
        total_expr = zero_expr + one_expr
        self.constraints.append(self.model.addConstr(total_expr <= local_branching_size))

    def proximity(self, index_to_val, obj_val, proximity_delta, binary_indexes) -> None:
        # Set the flag so we can undo
        self.is_obj_transformed = True

        zero_binary_vars, one_binary_vars = self.split_binary_vars(self.variables, binary_indexes, index_to_val)
        # if x_inc=0, new objective expression is x_inc.
        # if x_inc=1, new objective expression is 1 - x_inc.
        # Drop all other vars (when not in the expr it is set to 0 by default)
        zero_expr = quicksum(zero_binary_vars)
        one_expr = quicksum(1 - var for var in one_binary_vars)

        # add cutoff constraint depending on sense, so that next state is better quality
        # a slack variable z to prevent infeasible solution, \theta = 1
        self.proximity_z = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='proximity_z')
        self.model.update()  # Update model to include new variable
        # Use obj_val - delta * |obj_val| so the cutoff always tightens
        # (moves in the improvement direction), even when obj_val < 0.
        cutoff = obj_val - proximity_delta * abs(obj_val)
        self.constraints.append(self.model.addConstr(self.model.getObjective() <=
                                                     cutoff + self.proximity_z))

        # M * z is to make sure model does not use z, unless needed to avoid infeasibility
        self.model.setObjective(zero_expr + one_expr + self.big_m * self.proximity_z, GRB.MINIMIZE)

    def rens(self, index_to_val, rens_float_set, lp_index_to_val) -> None:
        for var in self.variables:
            if var.index in rens_float_set:
                # Restrict discrete vars to round up and down integer version of the lp
                # EX: If var = 3.5, the constraint is var >= 3 and var <= 4
                self.constraints.append(self.model.addConstr(var >= math.floor(lp_index_to_val[var.index])))
                self.constraints.append(self.model.addConstr(var <= math.ceil(lp_index_to_val[var.index])))
            else:
                # If not in the set, fix the var to the current state
                self.constraints.append(self.model.addConstr(var == index_to_val[var.index]))

    def random_objective(self) -> None:
        # Set the flag so we can undo
        self.is_obj_transformed = True

        # Get original objective function
        expr = self.org_objective_fn

        # Extract variables and coefficients
        vars = []
        coeffs = []

        for i in range(expr.size()):
            vars.append(expr.getVar(i))
            coeffs.append(expr.getCoeff(i))

        num_vars = len(vars)
        num_keep = max(1, int(0.25 * num_vars))  # Ensure at least one variable is kept

        # Randomly select indices to keep
        indices = list(range(num_vars))
        random.shuffle(indices)
        keep_indices = set(indices[:num_keep])

        # Build new objective function with 25% coefficients
        new_obj = gp.LinExpr()
        for i in range(num_vars):
            if i in keep_indices:
                if self.is_obj_sense_changed:
                    new_obj.addTerms(-coeffs[i], vars[i])
                else:
                    new_obj.addTerms(coeffs[i], vars[i])
            else:
                # Coefficient is zero, do not add term
                pass

        # Set the new objective function
        self.model.setObjective(new_obj, GRB.MINIMIZE)

    def solve_and_undo(self,
                       time_limit_in_sc=None,
                       solution_limit=None,
                       is_feasibility_focus=False) -> Tuple[Dict[Any, float], float]:
        # Set limits
        if time_limit_in_sc is not None:
            self.model.Params.TimeLimit = time_limit_in_sc
        if solution_limit is not None:
            self.model.Params.SolutionLimit = solution_limit
        if is_feasibility_focus:
            self.model.Params.MIPFocus = 1 # prioritize feasibility
            self.model.Params.Heuristics = 0.5  # spend 50% of node time on heuristics (default is 0.05)

        # Gurobi specific: Update model after adding constraints and variables
        self.model.update()

        # Optimize
        self.model.optimize()

        # Get solution
        index_to_val, obj_val = self.get_index_to_val_and_objective()

        # Free the model
        self.model.reset(0)

        # Undo limits
        if time_limit_in_sc is not None:
            self.model.Params.TimeLimit = GRB.INFINITY
        if solution_limit is not None:
            self.model.Params.SolutionLimit = GRB.MAXINT
        if is_feasibility_focus:
            self.model.Params.MIPFocus = 0
            self.model.Params.Heuristics = 0.05

        # Remove constraints, and reset
        for ct in self.constraints:
            self.model.remove(ct)
        self.constraints = []

        # if dins_abs_var variable is added, remove it
        if self.dins_abs_var:
            self.model.remove(self.dins_abs_var)
            self.dins_abs_var = None

        # Reset to original objective (random objective and proximity change objective)
        if self.is_obj_transformed:
            self.is_obj_transformed = False

            # Reset back to original minimize objective
            if self.is_obj_sense_changed:
                self.model.setObjective(-1 * self.org_objective_fn, GRB.MINIMIZE)
            else:
                self.model.setObjective(self.org_objective_fn, GRB.MINIMIZE)

            # if proximity_z variable is added, remove it
            if self.proximity_z:  # if proximity_delta > 0
                self.model.remove(self.proximity_z)
                self.proximity_z = None
            else:  # if random heuristic used, reset heuristics
                self.model.Params.Heuristics = 0.05  # Reset to default value

        # Gurobi specific: Update model after removals
        self.model.update()

        # Return solution
        return index_to_val, obj_val

    def solve_random_and_undo(self,
                              time_limit_in_sc=None,
                              solution_limit=None,
                              is_feasibility_focus=False) -> Tuple[Dict[Any, float], float]:
        # Set limits
        if time_limit_in_sc is not None:
            self.model.Params.TimeLimit = time_limit_in_sc
        if solution_limit is not None:
            self.model.Params.SolutionLimit = solution_limit
        if is_feasibility_focus:
            self.model.Params.MIPFocus = 1 # prioritize feasibility
            self.model.Params.Heuristics = 0.5  # spend 50% of node time on heuristics (default is 0.05)

        # Set random objective
        self.random_objective()

        # Gurobi specific: Update model
        self.model.update()

        # Solve
        self.model.optimize()

        index_to_val, obj_val = self.get_index_to_val_and_objective()

        # Free the model
        self.model.reset(0)

        # Restore original objective
        self.is_obj_transformed = False

        # Reset back to original minimize objective
        if self.is_obj_sense_changed:
            self.model.setObjective(-1 * self.org_objective_fn, GRB.MINIMIZE)
        else:
            self.model.setObjective(self.org_objective_fn, GRB.MINIMIZE)

        # Undo limits
        if time_limit_in_sc is not None:
            self.model.Params.TimeLimit = GRB.INFINITY
        if solution_limit is not None:
            self.model.Params.SolutionLimit = GRB.MAXINT
        if is_feasibility_focus:
            self.model.Params.MIPFocus = 0
            self.model.Params.Heuristics = 0.05

        return index_to_val, obj_val

    def solve_lp_and_undo(self) -> Tuple[Dict[Any, float], float]:
        # Solve LP relaxation
        int_vars = []
        bin_vars = []
        variables = self.model.getVars()

        for var in variables:
            if self.is_binary(var.VType):
                var.VType = "C"
                bin_vars.append(var)
            elif self.is_discrete(var.VType):
                var.VType = "C"
                int_vars.append(var)

        # Gurobi specific: Update model after removals
        self.model.update()

        # Solve
        self.model.optimize()
        lp_index_to_val, lp_obj_val = self.get_index_to_val_and_objective()

        # Get back the original model
        self.model.reset(0)

        # Revert variable types
        for var in int_vars:
            var.VType = "I"
        for var in bin_vars:
            var.VType = "B"

        # Gurobi specific: Update model after removals
        self.model.update()

        return lp_index_to_val, lp_obj_val

    def get_index_to_val_and_objective(self) -> Tuple[Dict[Any, float], float]:
        # we check if the optimized model has solutions, feasible, and is in the solved state
        if self.model.SolCount == 0 or self.model.Status == GRB.INFEASIBLE:
            return dict(), 9999999
        else:
            index_to_val = dict([(var.index, var.X) for var in self.model.getVars()])
            obj_value = self.model.ObjVal
            return index_to_val, obj_value

    @staticmethod
    def is_discrete(var_type) -> bool:
        return var_type in ("B", "I")

    @staticmethod
    def is_binary(var_type) -> bool:
        return var_type in "B"

    @staticmethod
    def split_binary_vars(variables, binary_indexes, index_to_val) -> Tuple[List[Any], List[Any]]:
        zero_binary_vars = []
        one_binary_vars = []
        for var in variables:
            if var.index in binary_indexes:
                if math.isclose(index_to_val[var.index], 0.0):
                    zero_binary_vars.append(var)
                else:
                    one_binary_vars.append(var)
        return zero_binary_vars, one_binary_vars
