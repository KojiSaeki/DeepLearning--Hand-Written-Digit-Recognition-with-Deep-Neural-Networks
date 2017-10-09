# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (learning rate) over a
training run.
"""

import numpy as np


class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.
        Run at the beginning of each epoch.
        """
        learning_rule.learning_rate = self.learning_rate*epoch_number

class TimeLearningRateScheduler(object):
    """Example of scheduler interface which sets a time learning rate."""

    def __init__(self, learning_rate):
        """Construct a new time learning rate scheduler object.
        """
        self.learning_rate = learning_rate
        self.decay = 10

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.
        Run at the beginning of each epoch.
        """
        learning_rule.learning_rate = self.learning_rate*(1./(1.+(epoch_number/self.decay)))

class MomLearningRateScheduler(object):
    """Example of scheduler interface which sets a time learning rate."""

    def __init__(self,mom_coeff):
        """Construct a new time learning rate scheduler object.

        Args:
            learning_rate
            alpha : determines the asymptotic momentum
			tau : tau>=10
			gama: 0<=gamma<=tau
        """
        self.mom_coeff = mom_coeff
        self.alpha = 0.9
        self.tau = 5
        self.gamma = 5

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.
        """
        learning_rule.mom_coeff = self.alpha*(1. - (self.gamma/(epoch_number+self.tau)))