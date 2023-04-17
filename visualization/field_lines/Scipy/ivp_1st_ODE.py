import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 

def f(t, y):
    dydt = np.cos(t)
    return dydt

t_span = [0, 2*np.pi]
y0 = (0,)

sol = solve_ivp(f, t_span, y0, t_eval=np.linspace(0, 2*np.pi, 50))
print(sol)

t = sol.t
y, = sol.y

plt.figure(1)
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')
plt.xscale('linear')
plt.yscale('linear')
plt.savefig('ivp_1st_ODE.png')