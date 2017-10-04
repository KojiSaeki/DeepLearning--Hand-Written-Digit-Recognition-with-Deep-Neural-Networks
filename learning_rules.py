# -*- coding: utf-8 -*-
"""Learning rules.
This module contains classes implementing gradient based learning rules.
"""

import numpy as np
import math


class GradientDescentLearningRule(object):
    """Simple (stochastic) gradient descent learning rule.

    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
    """

    def __init__(self, learning_rate=1e-3):
        """Creates a new learning rule object. """
        assert learning_rate > 0., 'learning_rate should be positive.'
        self.learning_rate = learning_rate

    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters."""
        self.params = params

    def print_coeff(self):
         print "EPS = "

    def reset(self):
        """Resets any additional state variables to their initial values.
        For this learning rule there are no additional state variables so we
        do nothing here.
        """
        pass

    def update_params(self, grads_wrt_params):
        """Applies a single gradient descent update to all parameters."""
		
        #print "grads_wrt_params shapu:"+str(np.array(grads_wrt_params).shape)
        #print "params:"+str(np.array(self.params).shape)
        for param, grad in zip(self.params, grads_wrt_params):
            #print "grad:"+str(np.array(grad).shape)
            #print "param:"+str(np.array(param).shape)
            param -= self.learning_rate * grad


class MomentumLearningRule(GradientDescentLearningRule):
    """Gradient descent with momentum learning rule.

    This extends the basic gradient learning rule by introducing extra
    momentum state variables for each parameter. These can help the learning
    dynamic help overcome shallow local minima and speed convergence when
    making multiple successive steps in a similar direction in parameter space.
    """

    def __init__(self, learning_rate=1e-3, mom_coeff=0.1):
        """Creates a new learning rule object."""
        super(MomentumLearningRule, self).__init__(learning_rate)
        assert mom_coeff >= 0. and mom_coeff <= 1., (
            'mom_coeff should be in the range [0, 1].'
        )
        self.mom_coeff = mom_coeff

    def initialise(self, params):
	
        """Initialises the state of the learning rule for a set or parameters."""
        super(MomentumLearningRule, self).initialise(params)
        self.moms = []
        for param in self.params:
            self.moms.append(np.zeros_like(param))

    def reset(self):
        """Resets any additional state variables to their initial values."""
        for mom in self.moms:
            mom *= 0.

    def print_coeff(self):
         print "Coeff is :"+str(self.mom_coeff)

    def update_params(self, grads_wrt_params):
        """Applies a single update to all parameters."""
        #print self.mom_coeff
        for param, mom, grad in zip(self.params, self.moms, grads_wrt_params):
            mom *= self.mom_coeff
            mom -= self.learning_rate * grad
            param += mom

class AdagradLearningRule(GradientDescentLearningRule):
    """The Adagrad learning rule.
    """

    def __init__(self, learning_rate=1e-3, eps=0.0001):
        """Creates a new learning rule object."""
		
        super(AdagradLearningRule, self).__init__(learning_rate)
        assert eps >= 0. and eps <= 1., (
            'mom_coeff should be in the range [0, 1].'
        )
        self.eps = eps

    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters."""
		
        super(AdagradLearningRule, self).initialise(params)
        self.cache = []
        for param in self.params:
            self.cache.append(np.zeros_like(param))

    def reset(self):
        """Resets any additional state variables to their initial values.
        For this learning rule this corresponds to zeroing all the momenta.
        """
        for cache in self.cache:
            cache *= 0.

    def print_coeff(self):
        print "Coeff is :"+str(self.eps)

    def update_params(self, grads_wrt_params):
        """Applies a single update to all parameters."""
        cache = self.cache
        for param,cac ,grad in zip(self.params,cache, grads_wrt_params):
            cac+=grad**2
            param += -self.learning_rate*grad/(np.sqrt(cac)+self.eps)

class RMSPropLearningRule(AdaptiveLearningRule):
    """Gradient descent with momentum learning rule.
    """
    def __init__(self, learning_rate=1e-3, eps=0.0001):
        """Creates a new learning rule object."""
        super(RMSPropLearningRule, self).__init__(learning_rate)
        assert eps >= 0. and eps <= 1., (
            'mom_coeff should be in the range [0, 1].'
        )
        self.eps = eps
        self.decay_rate = 0.9

    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters. """
        super(RMSPropLearningRule, self).initialise(params)
        self.cache = []
        for param in self.params:
            self.cache.append(np.zeros_like(param))

    def reset(self):
        """Resets any additional state variables to their initial values.
        For this learning rule this corresponds to zeroing all the momenta.
        """
        for cache in self.cache:
            cache *= 0.

    def print_coeff(self):
        print "Coeff is :"+str(self.eps)

    def update_params(self, grads_wrt_params):
        """Applies a single update to all parameters. """
        cache = self.cache
        for param,cac ,grad in zip(self.params,cache, grads_wrt_params):
            cac=self.decay_rate*cac + (1.-self.decay_rate)*(grad**2)
            param -= self.learning_rate*grad/(np.sqrt(cac)+self.eps)
			
			
