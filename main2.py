from LabN2 import *
a,b=-5,5
N=10
N_a=[10,20,50,100,200,400,800,1600,3200,6400]

data=create_data_grid(a, b, N)
compare_diff_plot(data,"first",N)
#delta(a,b,N_a,"first")
#delta(a,b,N_a,"second")