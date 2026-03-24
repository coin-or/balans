import copy
import math

from balans.base_state import _State
from balans.utils import timestamp


def _rins(current: _State, rnd_state, delta) -> _State:
    #  Take the LP relaxed solution of the original MIP and the current incumbent.
    #  If a DISCRETE variable x_inc = x_lp, then do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.

    print(f"{timestamp()} *** Operator: RINS (delta={delta * 100:.0f}%)")
    print(f"{timestamp()} \t Destroy current objective: {current.obj_val}")
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    lp_index_to_val = current.instance.lp_index_to_val

    #  If a variable x_cur = x_lp, do not change it.
    indexes_with_diff_value = [i for i in discrete_indexes if not
    math.isclose(lp_index_to_val[i], current.index_to_val[i])]

    # Else potentially change it
    size = int(delta * len(indexes_with_diff_value))
    next_state.destroy_set = set(rnd_state.choice(indexes_with_diff_value, size))

    return next_state


def rins_05(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.05)


def rins_10(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.10)


def rins_15(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.15)


def rins_20(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.20)


def rins_25(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.25)


def rins_30(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.30)


def rins_35(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.35)


def rins_40(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.40)


def rins_45(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.45)


def rins_50(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.50)


def rins_55(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.55)


def rins_60(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.60)


def rins_65(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.65)


def rins_70(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.70)


def rins_75(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.75)


def rins_80(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.80)


def rins_85(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.85)


def rins_90(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.90)


def rins_95(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.95)
