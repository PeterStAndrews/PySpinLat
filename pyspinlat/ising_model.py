# Ising model dynamics
#
# Copyright (C) 2018 Peter Mann
# 
# This file is part of `PySpinLat`, for statistical mechanical 
# spin lattice models using Python.
#
# `PySpinLat` is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# `PySpinLat` is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with `PySpinLat`. If not, see <http://www.gnu.org/licenses/gpl.html>.

from numpy.random import rand
import numpy as np
import networkx
import epyc

class IsingModel( pyspinlat.LatticeProcess ):
    '''Subclasses the `LatticeProcess` to fill in the dynamics of 
    each update according to the Ising model under local Monte Carlo 
    updates. '''
    
    def __init__(self):
        super(IsingModel, self).__init__()
        
    def dynamics( self, g, beta ):
        '''Given a current lattice spin configuration, this 
        function executes a Monte Carlo update for the system.
        It first picks a site at random from the lattice, before 
        summing the spin states of its neighbours. The energy 
        is computed and the Metropolis-Hastings criteria is 
        evaluated to decide if the spin flip is kept or not. Since
        spins are +-1, multiplication by -1 will always flip the spin.
        The current spin configuration is then returned. 
        
        :param g: lattice
        :param beta: inverse temperature
        '''
        # pick a site at random
        n = choice(g.nodes())
        
        # grab a list of its neighbours
        nb = g.neighbors(n)
        
        # interate over the nearest neighbours and sum spins
        ns = 0
        for nn in nb:
            ns += g.node[nn][self.SPIN]
        
        # compute the energy change
        dH = 2 * g.node[n][self.SPIN] * ns
        
        # evaluate Metropolis condition
        if dH < 0 or rand() < np.exp(-dH*beta):
            g.node[n][self.SPIN] *= -1

        return g
    
