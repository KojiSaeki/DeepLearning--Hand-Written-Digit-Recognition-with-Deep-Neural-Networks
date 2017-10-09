# -*- coding: utf-8 -*-
"""Parameter penalties.

This module defines parameter penalty functions which can be used to regularise
training.
"""

import numpy as np


class L1Penalty(object):
    """L1 parameter penalty.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.
        """
        return self.coefficient * abs(parameter).sum()

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.
        """
        return self.coefficient * np.sign(parameter)

    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)


class L2Penalty(object):
    """L2 parameter penalty.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.
        """
        return 0.5 * self.coefficient * (parameter**2).sum()

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.
        """
        return self.coefficient * parameter

    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)
