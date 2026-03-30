import copy

from balans.base_state import _State
from balans.utils import timestamp


def proximity(current: _State, rnd_state, delta) -> _State:
    # Objective function modification
    # Change the objective coefficients of the original
    # problem based on the current solution value.
    # For binary variables,
    # if x_inc =0, update its objective coefficient to 1.
    # if x_inc =1, update its objective coefficient to -1.
    # Drop all other variables by making their coefficient 0.
    # Send the destroy set to base_instance.
    # Note : the required objective operations for proximity search happens in base_instance file.

    print(f"{timestamp()} *** Operator: PROXIMITY SEARCH (delta={delta * 100:.0f}%)")
    print(f"{timestamp()} \t Destroy current objective: {current.instance.display_obj(current.obj_val)}")
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    next_state.proximity_delta = delta

    return next_state


def proximity_005(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.005)


def proximity_010(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.010)


def proximity_015(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.015)


def proximity_020(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.020)


def proximity_025(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.025)


def proximity_030(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.030)


def proximity_035(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.035)


def proximity_040(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.040)


def proximity_045(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.045)


def proximity_05(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.05)


def proximity_055(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.055)


def proximity_060(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.060)


def proximity_065(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.065)


def proximity_070(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.070)


def proximity_075(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.075)


def proximity_080(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.080)


def proximity_085(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.085)


def proximity_090(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.090)


def proximity_095(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.095)


def proximity_10(current: _State, rnd_state) -> _State:
    return proximity(current, rnd_state, delta=0.10)
