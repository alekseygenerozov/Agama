#!/usr/bin/python

import agama
import py_unsio
import numpy
import time

#1. set units (in Msun, Kpc, km/s)
agama.setUnits(mass=1e10, length=1, velocity=1)

#2. get in N-body snapshots
tbegin     = time.clock()
diskSnap   = py_unsio.CunsIn("../temp/disk.gadget","all","all")
diskSnap.nextFrame("")
ok,diskPos = diskSnap.getArrayF("all","pos")
diskPos    = diskPos.reshape(-1,3)
ok,diskVel = diskSnap.getArrayF("all","vel")
diskVel    = diskVel.reshape(-1,3)
ok,diskMass= diskSnap.getArrayF("all","mass")
diskPart   = numpy.hstack((diskPos,diskVel))

haloSnap   = py_unsio.CunsIn("../temp/halo.gadget","all","all")
haloSnap.nextFrame("")
ok,haloPos = haloSnap.getArrayF("all","pos")
haloPos    = haloPos.reshape(-1,3)
ok,haloMass= haloSnap.getArrayF("all","mass")
print (time.clock()-tbegin),"s to load",len(diskMass),"disk particles and",len(haloMass),"halo particles"

#3. create an axisymmetric potential from these snapshots
tbegin     = time.clock()
haloPot    = agama.Potential(type="Multipole", particles=(haloPos, haloMass),
             symmetry='a', gridsizeR=20, lmax=2)
##haloPot    = agama.Potential(file="halo.coef_spl")  # could load previously stored coefs instead of computing them
print (time.clock()-tbegin),"s to init",haloPot.name(),"potential for the halo; ", \
    "value at origin=",haloPot(0,0,0),"(km/s)^2"

tbegin     = time.clock()
diskPot    = agama.Potential(type="CylSpline", particles=(diskPos, diskMass),
             gridsizer=20, gridsizez=20, mmax=0)
##diskPot    = agama.Potential(file="disk.coef_cyl")
print (time.clock()-tbegin),"s to init",diskPot.name(),"potential for the disk; ", \
    "value at origin=",diskPot(0,0,0),"(km/s)^2"

diskPot.export("disk.coef_cyl")
haloPot.export("halo.coef_spl")
totalPot   = agama.Potential(diskPot, haloPot)  # create a composite potential

#4. compute actions for disk particles
tbegin     = time.clock()
actFinder  = agama.ActionFinder(totalPot)
print (time.clock()-tbegin),"s to init action finder"

tbegin     = time.clock()
actions    = actFinder(diskPart)
print (time.clock()-tbegin),"s to compute actions for",len(diskMass),"particles"

#5. write out data
Rz         = numpy.vstack(( (diskPos[:,0]**2+diskPos[:,1]**2)**0.5, diskPos[:,2] )).T
energy     = numpy.atleast_2d(totalPot(diskPos) + 0.5*(diskVel[:,0]**2+diskVel[:,1]**2+diskVel[:,2]**2) ).T
numpy.savetxt( "disk_actions.txt", numpy.hstack((Rz, actions, energy)),
    fmt="%.6g", header="R[Kpc]\tz[Kpc]\tJ_r[Kpc*km/s]\tJ_z[Kpc*km/s]\tJ_phi[Kpc*km/s]\tE[(km/s)^2]", delimiter="\t" )

print "ALL TESTS PASSED"
