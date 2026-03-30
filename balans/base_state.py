from copy import deepcopy
from typing import Any, Dict

from balans.base_instance import _Instance


class _State:
    """
    State of an instance with its solution and objective
    First, operators sets the solve settings of the state object
    Then, state calls instance.solve() with settings dictated by the operators
    """

    def __init__(self,
                 instance: _Instance,
                 index_to_val: Dict[Any, float],
                 obj_val: float,
                 destroy_set=None,
                 dins_set=None,
                 rens_float_set=None,
                 has_random_obj=False,
                 previous_index_to_val=None,
                 local_branching_size=0,
                 proximity_delta=None):

        # State holds an instance and its solution and objective
        # Instance holds features and the solve() logic dictated by operators
        self.instance = instance
        self.index_to_val = index_to_val  # index defined by SCIP var.getIndex()
        self.obj_val = obj_val

        # State receives the solve settings from the operators
        # and passes to instance.solve() and updates its solution and objective
        self.destroy_set = destroy_set
        self.dins_set = dins_set
        self.rens_float_set = rens_float_set
        self.has_random_obj = has_random_obj
        self.previous_index_to_val = previous_index_to_val
        self.local_branching_size = local_branching_size
        self.proximity_delta = proximity_delta

    def solution(self):
        return self.index_to_val

    def objective(self):
        return self.obj_val

    def reset_solve_settings(self):
        self.destroy_set = None
        self.dins_set = None
        self.rens_float_set = None
        self.has_random_obj = False
        self.local_branching_size = 0
        self.proximity_delta = 0

    def solve_and_update(self):
        # Solve the current state with the destroyed variables and update solution and objective
        self.index_to_val, self.obj_val = self.instance.solve(index_to_val=self.index_to_val,
                                                              obj_val=self.obj_val,
                                                              destroy_set=self.destroy_set,
                                                              dins_set=self.dins_set,
                                                              rens_float_set=self.rens_float_set,
                                                              has_random_obj=self.has_random_obj,
                                                              local_branching_size=self.local_branching_size,
                                                              proximity_delta=self.proximity_delta)

    def __deepcopy__(self, memo):
        # No need to copy the instance when create a deepcopy of state object, share the instance between the copy
        shared_attribute_names = ["instance"]
        assert isinstance(shared_attribute_names, (list, tuple))
        shared_attributes = {k: getattr(self, k) for k in shared_attribute_names}

        deepcopy_method = None

        if hasattr(self, '__deepcopy__'):
            # Do hack to prevent infinite recursion in call to deepcopy
            deepcopy_method = self.__deepcopy__
            self.__deepcopy__ = None

        for attr in shared_attribute_names:
            del self.__dict__[attr]

        clone = deepcopy(self)

        for attr, val in shared_attributes.items():
            setattr(self, attr, val)
            setattr(clone, attr, val)

        if hasattr(self, '__deepcopy__'):
            # Undo hack
            self.__deepcopy__ = deepcopy_method
            del clone.__deepcopy__

        return clone
