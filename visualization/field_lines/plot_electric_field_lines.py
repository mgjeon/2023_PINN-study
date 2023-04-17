import numpy as np
import matplotlib.pyplot as plt

Lx = 2
Ly = 2

xlim = 0.5
ylim = 0.5
x = np.linspace(-Lx, Lx, 8)
y = np.linspace(-Ly, Ly, 8)
x, y = np.meshgrid(x, y)

r = np.sqrt(x**2 + y**2)

q = 1
Ex = (q*x)/(r**2)
Ey = (q*y)/(r**2)
E = np.sqrt(Ex**2 + Ey**2)

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.plot(0, 0, 'ro')
# ax.plot(x, y, 'ko')
# ax.quiver(x, y, Ex, Ey, color='blue')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_xlim(-Lx-xlim, Lx+xlim)
# ax.set_ylim(-Ly-ylim, Ly+ylim)
# fig.savefig('electric_field_quiver.png', dpi=300)

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.plot(0, 0, 'ro')
# ax.plot(x, y, 'ko')
# ax.quiver(x, y, Ex/E, Ey/E, color='blue')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_xlim(-Lx-xlim, Lx+xlim)
# ax.set_ylim(-Ly-ylim, Ly+ylim)
# fig.savefig('electric_field_quiver_normalized.png', dpi=300)

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.plot(0, 0, 'ro')
# ax.plot(x, y, 'ko')
# ax.streamplot(x, y, Ex, Ey)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_xlim(-Lx-xlim, Lx+xlim)
# ax.set_ylim(-Ly-ylim, Ly+ylim)
# fig.savefig('electric_field_streamplot.png', dpi=300)

# start_points = np.column_stack([x.ravel(), y.ravel()])
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.plot(0, 0, 'ro')
# ax.plot(x, y, 'ko')
# ax.streamplot(x, y, Ex, Ey, start_points=start_points, integration_direction='forward')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_xlim(-Lx-xlim, Lx+xlim)
# ax.set_ylim(-Ly-ylim, Ly+ylim)
# fig.savefig('electric_field_streamplot_startpoints.png', dpi=300)


from scipy.integrate import odeint 

E_vec = np.dstack((Ex, Ey))

def model(x, t, E_vec):
    Ex = E_vec[..., 0]
    Ey = E_vec[..., 1]
    n = np.sqrt(Ex**2 + Ey**2)
    drdt = [Ex/n, Ey/n]
    print(drdt[0].shape)
    return drdt 

r0 = [1, 1]
tp = np.linspace(0, 10, 100)
r_sol = odeint(model, r0, tp, args=(E_vec,))
print(r_sol.shape)