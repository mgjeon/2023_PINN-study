import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 

def f(t, r):
    x, y = r
    print(r)
    dxdt = -y 
    dydt = x 
    return dxdt, dydt 

t_span = [0, 2*np.pi]
r0 = (1, 0)

sol = solve_ivp(f, t_span, r0, t_eval=np.linspace(0, 2*np.pi, 50))
print(sol)
print(sol.t.shape)
print(sol.y.shape)

t = sol.t 
x, y = sol.y 

plt.figure(1)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.xscale('linear')
plt.yscale('linear')
plt.axis('scaled')
plt.savefig('ivp_1st_ODE_system.png')