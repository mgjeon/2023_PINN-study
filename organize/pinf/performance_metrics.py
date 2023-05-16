import numpy as np 

from pinf.potential_field import get_potential
from pinf.metric import vector_norm, curl, divergence

def metrics(B, b, base_path='./evaluation.txt'):
    """
    B is the numerical solution
    b is the reference magnetic field
    """

    c_vec = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())

    M = np.prod(B.shape[:-1])
    c_cs = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b))

    E_n = vector_norm(B - b).sum() / vector_norm(b).sum()

    E_m = 1 / M * (vector_norm(B - b) / vector_norm(b)).sum()

    eps = (vector_norm(B) ** 2).sum() / (vector_norm(b) ** 2).sum()

    B_potential = get_potential(B[:, :, 0, 2], B.shape[-1])
    eps_p = (vector_norm(B) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()

    b_potential = get_potential(b[:, :, 0, 2], b.shape[-1])
    eps_p_b = (vector_norm(b) ** 2).sum() / (vector_norm(b_potential) ** 2).sum()

    j = curl(B)
    sig_J = (vector_norm(np.cross(j, B, -1)) / vector_norm(B)).sum() / vector_norm(j).sum()
    L1 = (vector_norm(np.cross(j, B, -1)) ** 2 / vector_norm(B) ** 2).mean()
    L2 = (divergence(B) ** 2).mean()

    j_b = curl(b)
    sig_J_b = (vector_norm(np.cross(j_b, b, -1)) / vector_norm(b)).sum() / vector_norm(j_b).sum()
    L1_b = (vector_norm(np.cross(j_b, b, -1)) ** 2 / vector_norm(b) ** 2).mean()
    L2_b = (divergence(b) ** 2).mean()

    with open(base_path, 'w') as f:
        print('c_vec', c_vec, file=f)
        print('c_cs', c_cs, file=f)
        print('1 - E_n', 1 - E_n, file=f)
        print('1 - E_m', 1 - E_m, file=f)
        print('eps', eps, file=f)
        print('--------------', file=f)
        print('eps_p', eps_p, file=f)
        print('sig_J', sig_J, file=f)
        print('L1', L1, file=f)
        print('L2', L2, file=f)
        print('eps_p_b', eps_p_b, file=f)
        print('sig_J_b', sig_J_b, file=f)
        print('L1_b', L1_b, file=f)
        print('L2_b', L2_b, file=f)