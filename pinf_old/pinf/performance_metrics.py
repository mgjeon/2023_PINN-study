import numpy as np 

from pinf.metric import vector_norm, curl, divergence

def metrics(B, b, B_potential):
    """
    B is the numerical solution
    b is the reference magnetic field
    """

    c_vec = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())

    M = np.prod(B.shape[:-1])
    c_cs = (1 / M) * np.sum((B * b).sum(-1) / (vector_norm(B)*vector_norm(b)))

    E_n = vector_norm(B - b).sum() / vector_norm(b).sum()

    E_m = (1 / M) * (vector_norm(B - b) / vector_norm(b)).sum()

    eps = (vector_norm(B) ** 2).sum() / (vector_norm(b) ** 2).sum()

    eps_p = (vector_norm(B) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()

    j = curl(B)
    sig_J = (vector_norm(np.cross(j, B, -1)) / vector_norm(B)).sum() / vector_norm(j).sum()
    L1 = (vector_norm(np.cross(j, B, -1)) ** 2 / vector_norm(B) ** 2).mean()
    L2 = (divergence(B) ** 2).mean()
    curlB = vector_norm(j).sum() / vector_norm(curl(b)).sum()

    key = ["C_vec", "C_cs", "1-En", "1-Em", "eps", "eps_p", "sig_J", "L1", "L2", "curlB"]
    metric = [c_vec, c_cs, 1-E_n, 1-E_m, eps, eps_p, sig_J, L1, L2, curlB]
    return dict(zip(key, metric))