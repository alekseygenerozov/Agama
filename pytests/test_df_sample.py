#!/usr/bin/python

import py_wrapper
py_wrapper.set_units(mass=1, length=1e-3, velocity=1)
dfp = dict(type='DoublePowerLaw', norm=1, ar=1.5465, az=0.72674, aphi=0.72674, br=1, bz=1, bphi=1, alpha=2.18, beta=9.06, j0=16)
pp  = dict(type='Dehnen', mass=1.286e+08, gamma=1.59, scaleRadius=123.16)
pot = py_wrapper.Potential(**pp)
df  = py_wrapper.DistributionFunction(**dfp)
x   = py_wrapper.sample(pot=pot, df=df, N=10000)