def main(NC, NI, NB, NC_TEST, SEED, LR, EPOCHS, N_LAYERS, FEATURES, LOG_ITER):
    # force jax to use one device
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

    # random key
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)

    # feature sizes
    feat_sizes = tuple([FEATURES for _ in range(N_LAYERS - 1)] + [1])

    # make & init model
    model = PINN(feat_sizes)
    params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)), jnp.ones((NC, 1)))

    # optimizer
    optim = optax.adam(LR)
    state = optim.init(params)

    # dataset
    key, subkey = jax.random.split(key, 2)
    train_data = pinn_train_generator_klein_gordon3d(NC, NI, NB, subkey)
    t, x, y, u_gt = pinn_test_generator_klein_gordon3d(NC_TEST)

    # forward & loss function
    apply_fn = jax.jit(model.apply)
    loss_fn = pinn_loss_klein_gordon3d(apply_fn, *train_data)

    @jax.jit
    def train_one_step(params, state):
        # compute loss and gradient
        loss, gradient = value_and_grad(loss_fn)(params)
        # update state
        params, state = update_model(optim, gradient, params, state)
        return loss, params, state
    
    start = time.time()
    for e in trange(1, EPOCHS+1):
        # single run
        loss, params, state = train_one_step(params, state)
        if e % LOG_ITER == 0:
            u = apply_fn(params, t, x, y)
            error = relative_l2(u, u_gt)
            print(f'Epoch: {e}/{EPOCHS} --> loss: {loss:.8f}, error: {error:.8f}')
    end = time.time()
    print(f'Runtime: {((end-start)/EPOCHS*1000):.2f} ms/iter.')

    print('Solution:')
    u = apply_fn(params, t, x, y)
    plot_klein_gordon3d(t, x, y, u)