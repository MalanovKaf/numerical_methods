from LabN2 import *
a,b=-5,5
N=20
N_a=[10,20,50,100,200,1000]

data=create_data_grid(a, b, N)
#compare_diff_plot(data,"second",N)
delta(a,b,N_a,"second")