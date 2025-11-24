import copy

from balans.base_state import _State


def _local_branching(current: _State, rnd_state, delta) -> _State:
    #  For binary variables we have a hard constraint,
    #  here we say change at most half of them (delta=0.5).
    #  Other possible delta values are 0.25 and 0.75.
    #  Send the destroy set to base_instance.

    print("*** Operator: ", "LOCAL BRANCHING")
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    binary_indexes = current.instance.binary_indexes

    # <= k in local branching
    next_state.local_branching_size = rnd_state.integers(int(0.05 * len(binary_indexes)),
                                                         int(delta * len(binary_indexes)) + 1)

    return next_state


def local_branching_05(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.05)


def local_branching_10(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.10)


def local_branching_15(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.15)


def local_branching_20(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.20)


def local_branching_25(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.25)


def local_branching_30(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.30)


def local_branching_35(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.35)


def local_branching_40(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.40)


def local_branching_45(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.45)


def local_branching_50(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.50)


def local_branching_55(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.55)


def local_branching_60(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.60)


def local_branching_65(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.65)


def local_branching_70(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.70)


def local_branching_75(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.75)


def local_branching_80(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.80)


def local_branching_85(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.85)


def local_branching_90(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.90)


def local_branching_95(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.95)
