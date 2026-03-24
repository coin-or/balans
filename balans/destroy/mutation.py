import copy

from balans.base_state import _State
from balans.utils import timestamp


def _mutation(current: _State, rnd_state, delta) -> _State:
    # Take the discrete variables of the original problem
    # Then, depending on delta parameter, choose a random subset of discrete variables to destroy.
    # Send the destroy set to base_instance.

    print(f"{timestamp()} *** Operator: MUTATION (delta={delta * 100:.0f}%)")
    print(f"{timestamp()} \t Destroy current objective: {current.obj_val}")
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes

    destroy_size = int(delta * len(discrete_indexes))
    next_state.destroy_set = set(rnd_state.choice(discrete_indexes, destroy_size, replace=False))

    return next_state


def mutation_05(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.05)


def mutation_10(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.10)


def mutation_15(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.15)


def mutation_20(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.20)


def mutation_25(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.25)


def mutation_30(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.30)


def mutation_35(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.35)


def mutation_40(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.40)


def mutation_45(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.45)


def mutation_50(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.50)


def mutation_55(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.55)


def mutation_60(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.60)


def mutation_65(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.65)


def mutation_70(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.70)


def mutation_75(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.75)


def mutation_80(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.80)


def mutation_85(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.85)


def mutation_90(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.90)


def mutation_95(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.95)
