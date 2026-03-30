import copy

from balans.base_state import _State
from balans.utils import timestamp


def _rens(current: _State, rnd_state, delta) -> _State:
    #  Take the LP relaxed solution of the original MIP.
    #  For discrete variables, choose the non-integral ones
    #  and store them in the rens_float_set to be bounded list.
    #  Send the destroy set (None for the rens case) and
    #  rens_float_set to base_instance.

    print(f"{timestamp()} *** Operator: RENS (delta={delta * 100:.0f}%)")
    print(f"{timestamp()} \t Destroy current objective: {current.instance.display_obj(current.obj_val)}")
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    floating_discrete_indexes = current.instance.lp_floating_discrete_indexes

    # Randomization STEP
    size = int(delta * len(floating_discrete_indexes))
    next_state.rens_float_set = set(rnd_state.choice(floating_discrete_indexes, size))

    return next_state


def rens_05(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.05)


def rens_10(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.10)


def rens_15(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.15)


def rens_20(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.20)


def rens_25(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.25)


def rens_30(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.30)


def rens_35(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.35)


def rens_40(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.40)


def rens_45(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.45)


def rens_50(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.50)


def rens_55(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.55)


def rens_60(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.60)


def rens_65(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.65)


def rens_70(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.70)


def rens_75(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.75)


def rens_80(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.80)


def rens_85(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.85)


def rens_90(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.90)


def rens_95(current: _State, rnd_state) -> _State:
    return _rens(current, rnd_state, delta=0.95)
