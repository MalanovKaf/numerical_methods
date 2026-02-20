from LabN1 import *

N=np.array([5,10,50,100,150])
a,b=0,10


for i in range (len(N)):
    interpolation(a,b,N[i],0,0)
    interpolation(a, b, N[i], 1, 0)
    interpolation(a, b, N[i], 0, 1)


delta_max(a,b,N,0,0)
delta_max(a,b,N,1,0)
delta_max(a,b,N,0,1)