import copy

from balans.base_state import _State
from balans.utils import timestamp


def dins(current: _State, rnd_state) -> _State:
    #  Take an LP relaxed solution of the original MIP.
    #  By considering only discrete variables, forms a Set J where |x_lp -x_inc| >= 0.5
    #  IF delta is given, do local branching for binary variables, change at most delta of them.
    #  If a variable is inside the Set J, it is part of the destroy set.
    #  Send the dins set (Set J) and local branching size to base_instance.

    print(f"{timestamp()} *** Operator: DINS")
    print(f"{timestamp()} \t Destroy current objective: {current.obj_val}")
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    integer_indexes = current.instance.integer_indexes
    lp_index_to_val = current.instance.lp_index_to_val

    # DINS for integer variables: form a set_j where |x_lp - x_inc| >= 0.5
    next_state.dins_set = set([i for i in integer_indexes
                               if abs(lp_index_to_val[i] - current.index_to_val[i]) >= 0.5])

    return next_state
