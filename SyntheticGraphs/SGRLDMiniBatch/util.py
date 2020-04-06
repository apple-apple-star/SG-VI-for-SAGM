# From ed.set_seed (edward/edward/util/graphs.py)
import numpy as np
import random
import tensorflow as tf
import six

def set_seed(x):
    """Set seed for both NumPy and TensorFlow.
    Args:
        x: int, float.
        seed
    """
    tf.reset_default_graph()
    node_names = list(six.iterkeys(tf.get_default_graph()._nodes_by_name))
    if len(node_names) > 0 and node_names != ['keras_learning_phase']:
        raise RuntimeError("Seeding is not supported after initializing "
                           "part of the graph. "
                           "Please move set_seed to the beginning of your code.")
    random.seed(x)
    np.random.seed(x)
    tf.set_random_seed(x)

def printing(var_tf):
    return var_tf

class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]