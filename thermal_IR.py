#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is to explore the possibility of thermally driven IR

@author: changyaochen
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import ccylib 

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
               T = 300.0,  # Temperature, in K
               f_d = 0.0,  # drive frequency
               force_d = 0.0  # drive force               
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
  # spring constant, SI unit
  k = E * w * h**3 * L / 12 * (1.875/L)**4
  # mass, SI unit
  m = rho * w * h * L
  # angular resonant frequency, in unit of rad/s
  omega0 = np.sqrt(k / m)
  # Boltzmann constant
  kB = 1.38064852e-23  # SI unit
  # spectral force density, in unit of N^2/Hz
  SF = 4 * kB * T * m * (m*omega0/Q)
  
  return [
            Y[1], 
            (
            -(k * Y[0] + (m * omega0 / Q) * Y[1] + beta*Y[0]**3)  # internal dynamic 
            + force_d * np.cos(2*np.pi*f_d*t)  #  coherent drive
            + 0*np.random.normal(0, 0.5)  #  thermal noise
            ) / m           
            ] 
  
#  assgin system parameters
h  = 10e-6  # thickness, in unit of m
w  = 10e-6  # width, in unit of m
L  = 500e-6  # length, in unit of m
rho = 2329  # mass density, in unit of kg/m^3
E = 150e9  # Young's modulus
Q = 100  # quality factor
beta = 0.0  # Duffing nonlinear coeff
T = 300.0  # temperature, in K


#  spring constant
k = E * w * h**3 * L / 12 * (1.875/L)**4
#  mass
m = rho * w * h * L
#  angular resonant frequency
omega0 = np.sqrt(k / m)
#  resonant frequency
f0 = omega0 / 2 / np.pi
#  drive frequency
f_d = 1 * f0
force_d = 1.0
  
#  maximum simulation time, in unit of s
#  we need to be careful with this number: I only need about 100 cycles
#  therefore, it is important to have a good estimate of the resonant freq. 
#  In most case, it will be in the kHz range, therefore, tMax will be in ms  
tMax = 200 / f0
#  simulation time spans, with resolution
tspan = np.linspace(0, tMax, 10000)

#  start the numerical process
x1Int, v1Int = 0.0, 0.0
sols = sp.integrate.odeint(single_osc, [x1Int, v1Int], tspan,
                           args = (h, w, L, rho, E, Q, beta, T, f_d, force_d))  #   solve the equations!
                           
#  plot the result
plt.figure(1)
plt.plot(tspan, sols[:, 0], '-')
ccylib.prettify_figure('Amplitude (a.u.)', 'time (s)', '')   

#  perform and then plot the FFT
freq_fft, fft_out = ccylib.myFFT(tspan, sols[:, 0])  
plt.figure(2) 
plt.plot(freq_fft, fft_out)
ccylib.prettify_figure('FFT Amplitude (a.u.)', 'FFT frequency (Hz)', '') 

