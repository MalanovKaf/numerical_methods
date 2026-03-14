
from LabN3 import *

integrator = function_integral(-1, 1, 11)
# Массив различных N для исследования
N_val=[32,64,128,256,512,1024,2048,4096]
#integrator.plot_all_methods(N_val)
integrator.compare_functions_error(N_val)

