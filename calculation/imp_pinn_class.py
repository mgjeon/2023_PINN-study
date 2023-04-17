# forward function
class PINN(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x, y, z):
        X = jnp.concatenate([x, y, z], axis=1)
        init = nn.initializers.glorot_normal()
        for fs in self.features[:-1]:
            X = nn.Dense(fs, kernel_init=init)(X)
            X = nn.activation.tanh(X)
        X = nn.Dense(self.features[-1], kernel_init=init)(X)
        return X


# hessian-vector product
def hvp_fwdrev(f, primals, tangents, return_primals=False):
    g = lambda primals: vjp(f, primals)[1](tangents[0])[0]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# loss function
def pinn_loss_klein_gordon3d(apply_fn, *train_data):
    def residual_loss(params, t, x, y, source_term):
        # compute u
        u = apply_fn(params, t, x, y)
        # tangent vector du/du
        v = jnp.ones(u.shape)
        # 2nd derivatives of u
        utt = hvp_fwdrev(lambda t: apply_fn(params, t, x, y), (t,), (v,))
        uxx = hvp_fwdrev(lambda x: apply_fn(params, t, x, y), (x,), (v,))
        uyy = hvp_fwdrev(lambda y: apply_fn(params, t, x, y), (y,), (v,))
        return jnp.mean((utt - uxx - uyy + u**2 - source_term)**2)
    
    def initial_boundary_loss(params, t, x, y, u):
        return jnp.mean((apply_fn(params, t, x, y) - u)**2)

    # unpack data
    tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data

    # isolate loss function from redundant arguments
    fn = lambda params: residual_loss(params, tc, xc, yc, uc) + \
                        initial_boundary_loss(params, ti, xi, yi, ui) + \
                        initial_boundary_loss(params, tb, xb, yb, ub)

    return fn


# optimizer step function
@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state

# 2d time-dependent klein-gordon exact u
def _klein_gordon3d_exact_u(t, x, y):
    return (x + y) * jnp.cos(2*t) + (x * y) * jnp.sin(2*t)


# 2d time-dependent klein-gordon source term
def _klein_gordon3d_source_term(t, x, y):
    u = _klein_gordon3d_exact_u(t, x, y)
    return u**2 - 4*u


# train data
def pinn_train_generator_klein_gordon3d(nc, ni, nb, key):
    keys = jax.random.split(key, 13)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=10.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-1., maxval=1.)
    uc = _klein_gordon3d_source_term(tc, xc, yc)
    # initial points
    ti = jnp.zeros((ni, 1))
    xi = jax.random.uniform(keys[3], (ni, 1), minval=-1., maxval=1.)
    yi = jax.random.uniform(keys[4], (ni, 1), minval=-1., maxval=1.)
    ui = _klein_gordon3d_exact_u(ti, xi, yi)
    # boundary points (hard-coded)
    tb = [
        jax.random.uniform(keys[5], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[6], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[7], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[8], (nb, 1), minval=0., maxval=10.)
    ]
    xb = [
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb),
        jax.random.uniform(keys[9], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[10], (nb, 1), minval=-1., maxval=1.)
    ]
    yb = [
        jax.random.uniform(keys[11], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[12], (nb, 1), minval=-1., maxval=1.),
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb)
    ]
    ub = []
    for i in range(4):
        ub += [_klein_gordon3d_exact_u(tb[i], xb[i], yb[i])]
    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    ub = jnp.concatenate(ub)
    return tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub


# test data
def pinn_test_generator_klein_gordon3d(nc_test):
    t = jnp.linspace(0, 10, nc_test)
    x = jnp.linspace(-1, 1, nc_test)
    y = jnp.linspace(-1, 1, nc_test)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    u_gt = _klein_gordon3d_exact_u(tm, xm, ym)
    t = tm.reshape(-1, 1)
    x = xm.reshape(-1, 1)
    y = ym.reshape(-1, 1)
    u_gt = u_gt.reshape(-1, 1)
    return t, x, y, u_gt

def relative_l2(u, u_gt):
    return jnp.linalg.norm(u-u_gt) / jnp.linalg.norm(u_gt)

def plot_klein_gordon3d(t, x, y, u):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t, x, y, c=u, s=0.5, cmap='seismic')
    ax.set_title('U(t, x, y)', fontsize=20, pad=-5)
    ax.set_xlabel('t', fontsize=18, labelpad=10)
    ax.set_ylabel('x', fontsize=18, labelpad=10)
    ax.set_zlabel('y', fontsize=18, labelpad=10)
    plt.show()