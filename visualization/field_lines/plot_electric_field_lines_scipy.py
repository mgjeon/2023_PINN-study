import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

class charge:
    def __init__(self, q, pos):
        self.q = q
        self.pos = pos 

def E_point_charge(q, a, x, y):
    r = (x - a[0])**2 + (y - a[1])**2
    Ex = q * (x - a[0]) / (r**1.5)
    Ey = q * (y - a[0]) / (r**1.5)
    return [Ex, Ey]

def E_total(x, y, charges):
    Ex, Ey = 0, 0
    for C in charges:
        C_Ex, C_Ey = E_point_charge(C.q, C.pos, x, y)
        Ex += C_Ex 
        Ey += C_Ey 
    return [Ex, Ey]

def E_dir(t, y, charges):
    Ex, Ey = E_total(y[0], y[1], charges)
    E = np.sqrt(Ex**2 + Ey**2)
    dxds = Ex/E
    dyds = Ey/E 
    return [dxds, dyds]

charges = [charge(1, [-1, 0]), 
           charge(-1, [1, 0])]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

x0, x1=-2, 2
y0, y1=-1.5, 1.5
R = 0.01
for C in charges:
    ds = 0.8*R 
    if C.q < 0:
        ds = -ds 
    for alpha in np.linspace(0, 2*np.pi*15/16, 16):
        r = ode(E_dir)
        r.set_integrator('vode')
        r.set_f_params(charges)
        x = [C.pos[0] + np.cos(alpha)*R]
        y = [C.pos[0] + np.sin(alpha)*R]
        r.set_initial_value([x[0], y[0]], 0)
        while r.successful():
            r.integrate(r.t + ds)
            x.append(r.y[0])
            y.append(r.y[1])
            hit_charge=False

            for C2 in charges:
                if np.sqrt((r.y[0]-C2.pos[0])**2+(r.y[1]-C2.pos[1])**2)<R:
                    hit_charge=True
            if hit_charge or (not (x0<r.y[0] and r.y[0]<x1)) or \
                    (not (y0<r.y[1] and r.y[1]<y1)):
                break
        ax.plot(x, y, '-k')

# plot point charges
for C in charges:
    if C.q>0:
        ax.plot(C.pos[0], C.pos[1], 'bo', ms=8*np.sqrt(C.q))
    if C.q<0:
        ax.plot(C.pos[0], C.pos[1], 'ro', ms=8*np.sqrt(-C.q))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
fig.savefig('electric_field_scipy.png', dpi=300)      