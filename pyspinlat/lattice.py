# LatticeProcess base class
#
# Copyright (C) 2018 Peter Mann
# 
# This file is part of `PySpinLat`, for epidemic network 
# analytical results using Python.
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

from random import choice
import numpy as np
import networkx
import epyc

class LatticeProcess( epyc.Experiment ):
    '''Base class that generates a spin-lattice with an abstract 
    process running on it. The update dynamics and the process
    details can be subclassed.
    '''
    N = 'N'                       # size of lattice
    SPIN = 'spin'                 # key for nodes spin
    _BETA = 'beta'                # inverse temperature
    _NUM_UPDATES = 'num_updates'  # number of updates for dynamics

    def __init__(self):
        super(LatticeProcess, self).__init__()
        
    def configure( self, params ):
        '''Creates a 2-dimensional lattice. Each lattice point 
        is then given a spin (-1,+1) at random and the prototype 
        network is saved for later use. The lattice dimensions and 
        set of admissible spins can easily be modified.
        
        :param params: the experimental parameters'''
        epyc.Experiment.configure(self, params)
        N = params[self.N]
        
        # create an (NxN) 2-dimensional lattice with periodic boundary condition
        g = networkx.grid_graph(dim=[N,N],periodic=True)
        
        # initialise the nodes with a spin configuration
        for n in g.nodes():
            g.node[n][self.SPIN] = choice([-1, 1])
            
        # store it for later
        self._prototype = g
        
    def setUp( self, params ):
        '''Set up a working network for this run of the experiment.
        This is useful when performing lab experiments.
        :param params: the experimental parameters'''
        epyc.Experiment.setUp(self, params)
        self._network = self._prototype.copy()

    def tearDown( self ):
        '''Delete the current network.'''
        epyc.Experiment.tearDown(self)
        self._network = None

    def _magnetisation( self, g ):
        '''Computes the magnetisation of the state.
        :param g: the current lattice state
        :returns m: the magnetisation'''
        m = 0
        for n in g.nodes():
            m += g.node[n][self.SPIN]
        return m
    
    def _lattice_energy( self, g ):
        '''Computes the energy of a given state by 
        computing the local environment of every spin
        state in the lattice. The return is divided by
        the number of neighbours each site has (assuming
        it is a regular lattice) accounting for zero index. 
        
        :param g: the current lattice state
        :returns H: the current lattice energy'''
        energy = 0
        for n in g.nodes():
            ns = 0
            for nn in g.neighbors(n):
                ns += g.node[nn][self.SPIN]
            energy += - ns * g.node[n][self.SPIN]
        return energy/(len(g.neighbors(n))+1.0)
    
    def returnField( self, g, params ):
        '''Returns the spin field as a numpy array for post-processing.'''
        # get spin attributes from network: returns dict as {node_id: spin}
        config = networkx.get_node_attributes(g,self.SPIN)
        N = params[self.N]
        # convert spins to array and reshape as lattice
        config_array = np.asarray(config.values()).reshape((N,N))
        return config_array
    
    def do( self, params ):
        '''Performs the simulation and returns the results.'''
        # initialise the parameters 
        N = params[self.N]                       # lattice dimension
        beta = params[self._BETA]                # inverse temperature
        num_updates = params[self._NUM_UPDATES]  # number of updates
        
        # create a results dict 
        rc = dict()
        
        # grab a copy of the network
        g = self._network
            
        # initialise system quantities
        lattice_energy = 0 
        magnetisation  = 0
        
        # update the dynamics 
        for i in range(num_updates):
            self.dynamics( g, beta)
            
            # compute expectation quantities 
            lattice_energy = self._lattice_energy(g)
            magnetisation = self._magnetisation(g)
            
        # report equilibrium lattice energy, magnetisation and temperature
        rc['E'] = lattice_energy
        rc['M'] = magnetisation
        rc['T'] = 1.0/beta
        
        return rc
   