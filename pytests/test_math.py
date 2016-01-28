#!/usr/bin/python

### shows a pretty checkerboard picture - samples drawn from the given function
import agama,math,matplotlib.pyplot as plt
M_PI = math.pi
def fnc(x):
    return max(0, math.sin(11*M_PI*x[0]) * math.sin(15*M_PI*x[1]))
val,err,_ = agama.integrateNdim(fnc, 2, maxeval=50000)
print "N-dimensional integration: result =", val, "+-", err, " (exact value:", 4/M_PI/M_PI * 83./165, ")"
arr,val,err,_ = agama.sampleNdim(fnc, 50000, [-1,1], [0,2])
print "N-dimensional sampling: result =",val, "+-", err
plt.plot(arr[:,0], arr[:,1], ',')
plt.show()
