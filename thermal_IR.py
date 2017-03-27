#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is to explore the possibility of thermally driven IR

@author: changyaochen
"""

import numpy as np
import matplotlib.pyplot as plt  
import os

dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)
plt.close('all')

def single_osc(Y, t, 
               h  = 10e-6,  # thickness, in unit of m
               w  = 10e-6,  # width, in unit of m
               L  = 500e-6,  # length, in unit of m
               rho = 2329,  # mass density, in unit of kg/m^3
               E = 150e9,  # Young's modulus
               Q = 100,  # quality factor
               beta = 0.0,  # Duffing nonlinear coeff
               ):
  '''
  This is a simple (harmonic if beta = 0.0) oscillator driven purely by 
  thermal noise (due to finite temperature T)
  
  the model system is a doubly clamped beam, and the material is assumed to be
  silicon
  
  Here I need to solve a Stochastic Differential Equation (SDE), or in physics
  realm, often called Langevin equation. See here:
    https://en.wikipedia.org/wiki/Langevin_dynamics
  and
    https://en.wikipedia.org/wiki/Stochastic_differential_equation
  The link above also explains the definitions of additive and multiplicative 
  noise quite well 
  '''
  # spring constant
  k = E * w * h**3 * L / 12 * (1.875/L)**4
  # mass
  m = rho * w * h * L
  # angular resonant frequency
  omega0 = np.sqrt(k / m)
  # Boltzmann constant
  kB = 1.38064852e-23  # SI unit
  # spectral force density, in unit of N^2/Hz
  SF = 4 * kB * T * m * (m*omega0/Q)
  
  return [
            Y[1], 
            (-(Y[0] + (m * omega0 / Q) * Y[1] + beta*Y[0]**3) + force) / m           
            ] 
  
tMax = 100 
tspan = np.linspace(0, tMax, 100*tMax+1)
