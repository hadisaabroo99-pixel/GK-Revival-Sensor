import numpy as np


def qfi_form(rho, X, P, Xb=None, Pb=None, eps=1e-12):
    """Completeness-corrected SLD-QFI 2x2 form for generators
       G_theta = cos(theta) P - sin(theta) X.

       rho : N x N density matrix in the energy eigenbasis (support basis)
       X,P : N x N quadrature matrices
       Xb,Pb : (optional) M x M quadratures on an enlarged basis, M >= N, used
               to close the out-of-support block <k|G^2|k> analytically.

       Returns (F, Fmin, Fmax) where F is the 2x2 symmetric form in
       coordinates (cos theta, -sin theta) i.e. G = a P + b X.
    """
    N = rho.shape[0]
    lam, U = np.linalg.eigh(rho)
    keep = lam > eps
    lam = lam[keep]
    lam = lam / lam.sum()
    U = U[:, keep]                       # N x r

    Xe = U.conj().T @ X @ U              # r x r
    Pe = U.conj().T @ P @ U

    dl = lam[:, None] - lam[None, :]
    sl = lam[:, None] + lam[None, :]
    W = np.where(sl > 0, 2 * dl**2 / np.where(sl > 0, sl, 1.0), 0.0)

    F = np.zeros((2, 2))
    F[0, 0] = np.sum(W * np.abs(Pe) ** 2)
    F[1, 1] = np.sum(W * np.abs(Xe) ** 2)
    F[0, 1] = F[1, 0] = np.sum(W * np.real(Pe * Xe.conj()))

    # ---- out-of-support closure (exact) --------------------------------
    if Xb is None:
        Xb, Pb = X, P
    M = Xb.shape[0]
    Ub = np.zeros((M, U.shape[1]), dtype=complex)
    Ub[:N, :] = U
    XX = Xb @ Xb
    PP = Pb @ Pb
    XP = Xb @ Pb
    dP2 = np.real(np.einsum('ik,ij,jk->k', Ub.conj(), PP, Ub))
    dX2 = np.real(np.einsum('ik,ij,jk->k', Ub.conj(), XX, Ub))
    dXP = np.real(np.einsum('ik,ij,jk->k', Ub.conj(), XP, Ub))   # Re<k|XP|k>

    F[0, 0] += 4 * np.sum(lam * (dP2 - np.sum(np.abs(Pe) ** 2, axis=0)))
    F[1, 1] += 4 * np.sum(lam * (dX2 - np.sum(np.abs(Xe) ** 2, axis=0)))
    F[0, 1] += 4 * np.sum(lam * (dXP - np.sum(np.real(Pe * Xe.conj()), axis=0)))
    F[1, 0] = F[0, 1]

    # G = a P + b X with (a,b) = (cos t, -sin t); eigenvalues of F give the
    # extremal information over direction.
    ev = np.linalg.eigvalsh(F)
    return F, float(max(ev.min(), 0.0)), float(ev.max())


def fq_at(F, theta):
    a, b = np.cos(theta), -np.sin(theta)
    return F[0, 0] * a * a + F[1, 1] * b * b + 2 * F[0, 1] * a * b


def theta_min(F):
    ev, V = np.linalg.eigh(F)
    a, b = V[:, 0]
    return float(np.arctan2(-b, a))
