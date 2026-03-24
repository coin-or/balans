# NOTE: This is experimental, and not used for now
import copy

from balans.base_state import _State
from balans.utils import timestamp


# def sort_list_with_indices(lst):
#     # Enumerate the list to keep track of original indices
#     indexed_list = [(val, idx) for idx, val in enumerate(lst)]

#     # Sort the indexed list based on the values
#     sorted_list = sorted(indexed_list, key=lambda x: x[0], reverse=True)

#     # Extract the sorted values and indices
#     sorted_values = [val for val, _ in sorted_list]
#     sorted_indices = [idx for _, idx in sorted_list]

#     return sorted_values, sorted_indices

def local_branching_relax(current: _State, rnd_state, delta) -> _State:
    print(f"{timestamp()} *** Operator: LB relax")
    print(f"{timestamp()} \t Destroy current objective: {current.obj_val}")
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # This is SCIP specific
    # variables = current.instance.mip.getVars()
    # binary_indexes = current.instance.binary_indexes
    # local_branching_size = int(len(binary_indexes) * delta)
    # zero_binary_vars, one_binary_vars = split_binary_vars(variables,
    #                                                       binary_indexes, current.index_to_val)
    #
    # # if current binary var is 0, flip to 1 consumes 1 unit of budget
    # # if current binary var is 1, flip to 0 consumes 1 unit of budget by (1-x)
    # zero_expr = quicksum(zero_var for zero_var in zero_binary_vars)
    # one_expr = quicksum(1 - one_var for one_var in one_binary_vars)
    # con = current.instance.model.addCons(zero_expr + one_expr <= local_branching_size)
    #
    # int_index = []
    # bin_index = []
    # count = 0
    # for var in variables:
    #     if var.vtype() == Constants.integer:
    #         current.instance.model.chgVarType(var, Constants.continuous)
    #         int_index.append(count)
    #     if var.vtype() == Constants.binary:
    #         current.instance.model.chgVarType(var, Constants.continuous)
    #         bin_index.append(count)
    #     count += 1
    #
    # current.instance.model.optimize()
    # if current.instance.model.getNSols() == 0 or current.instance.model.getStatus() == "infeasible":
    #     current.instance.model.freeTransform()
    #     count = 0
    #     for var in variables:
    #         if count in int_index:
    #             current.instance.model.chgVarType(var, Constants.integer)
    #         if count in bin_index:
    #             current.instance.model.chgVarType(var, Constants.binary)
    #         count += 1
    #     current.instance.model.delCons(con)
    #     return next_state
    #
    # lp_index_to_val, lp_obj_val = get_index_to_val_and_objective(current.instance.model)
    #
    # # Get back the original model
    # current.instance.model.freeTransform()
    # count = 0
    # for var in variables:
    #     if count in int_index:
    #         current.instance.model.chgVarType(var, Constants.integer)
    #     if count in bin_index:
    #         current.instance.model.chgVarType(var, Constants.binary)
    #     count += 1
    # current.instance.model.delCons(con)
    #
    # #  If a discrete variable x_rand1 = x_inc, do not change it.
    # indexes_with_same_value = [i for i in binary_indexes if
    #                            math.isclose(lp_index_to_val[i], current.index_to_val[i])]
    #
    # # Else potentially change it
    # indexes_with_diff_value = [i for i in binary_indexes if i not in indexes_with_same_value]
    # size = int(delta * len(indexes_with_diff_value))
    # next_state.destroy_set = set(rnd_state.choice(indexes_with_diff_value, size))
    #
    # #     zero_set = set([i for i in binary_indexes if i in indexes_with_same_value])
    # #     if len(non_zero_set) >= local_branching_size:
    # #         diff = [abs(lp_index_to_val[i] - current.index_to_val[i]) if i in binary_indexes else 0
    # #                 for i in range(len(lp_index_to_val))]
    # #         diff_value, diff_index = sort_list_with_indices(diff)
    # #         lb_relax_set = set(diff_index[:local_branching_size])
    # #     else:
    # #         random_set = set(rnd_state.choice(list(zero_set),
    # #         local_branching_size - len(non_zero_set), replace=False))
    # #         lb_relax_set = non_zero_set | random_set

    return next_state


def local_branching_relax_10(current: _State, rnd_state) -> _State:
    return local_branching_relax(current, rnd_state, delta=0.1)


def local_branching_relax_25(current: _State, rnd_state) -> _State:
    return local_branching_relax(current, rnd_state, delta=0.25)