import copy

from balans.base_state import _State
from balans.utils import timestamp


def random_objective(current: _State, rnd_state) -> _State:
    print(f"{timestamp()} *** Operator: RANDOM OBJECTIVE")
    print(f"{timestamp()} \t Destroy current objective: {current.obj_val}")
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    next_state.has_random_obj = True

    return next_state
