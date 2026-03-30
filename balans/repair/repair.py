from balans.base_state import _State
from balans.utils import timestamp


def repair(current: _State, rnd_state) -> _State:
    print(f"{timestamp()} \t Repair")

    # Solve the state with fixed variables to repair and update solution and objective
    current.solve_and_update()

    print(f"{timestamp()} \t Repair objective: {current.instance.display_obj(current.obj_val)}")

    return current
