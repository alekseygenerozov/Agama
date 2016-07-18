#!/usr/bin/python

### shows a pretty checkerboard picture - samples drawn from the given function
import agama,math,matplotlib.pyplot as plt
def fnc(x):
    return max(0, math.sin(11*math.pi*x[0]) * math.sin(15*math.pi*x[1]))
valI,errI,_ = agama.integrateNdim(fnc, 2, maxeval=50000)
exact =  (2/math.pi)**2 * 83./165
print "N-dimensional integration: result =", valI, "+-", errI, " (exact value:", exact, ")"
arr,valS,errS,_ = agama.sampleNdim(fnc, 50000, [-1,1], [0,2])
print "N-dimensional sampling: result =", valS, "+-", errS
plt.plot(arr[:,0], arr[:,1], ',')
plt.show()
if abs(valI-exact)<errI and abs(valS-exact)<errS and errI<1e-3 and errS<1e-3:
    print "\033[1;32mALL TESTS PASSED\033[0m"