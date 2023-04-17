import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 



def E(s, r):
    x, y = r
    print(r)
    d = np.sqrt(x**2 + y**2)
    Ex = x/(d**3)
    Ey = y/(d**3)
    E = np.sqrt(Ex**2 + Ey**2)
    dxdt = Ex/E
    dydt = Ey/E
    return dxdt, dydt 

t_span = [0, 2]
r0 = (1, 1)

sol = solve_ivp(E, t_span, r0)
print(sol)
print(sol.t.shape)
print(sol.y.shape)

t = sol.t 
x, y = sol.y 

plt.figure(1)
plt.plot(x, y)
plt.plot(0, 0, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((-3, 3))
plt.ylim((-3, 3))
plt.xscale('linear')
plt.yscale('linear')
plt.axis('scaled')
plt.savefig('ivp_1st_ODE_field_line.png')