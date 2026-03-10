
from LabN3 import *

integrator = function_integral(-1, 1, 11)
# Массив различных N для исследования
N_val=[2,4,8]
integrator.plot_all_methods(N_val)

