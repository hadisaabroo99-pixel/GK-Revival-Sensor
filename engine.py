import numpy as np
from scipy.linalg import eigh_tridiagonal, expm

# ---------------------------------------------------------------- systems
def pt_system(nu=4.0, Ng=20000, Nlev=28):
    """Trigonometric Poschl-Teller, lambda=mu=nu/2, hbar=m=a=1.
       H = -1/2 d2/dx2 + V,  V = V0[l(l-1)/sin^2(x/2)+m(m-1)/cos^2(x/2)], V0=1/8
       e_n = n(n+nu), omega = 1/2."""
    lam = mu = nu / 2.0
    x = np.linspace(0.0, np.pi, Ng + 2)[1:-1]
    dx = x[1] - x[0]
    V0 = 0.125
    V = V0 * (lam * (lam - 1) / np.sin(x / 2) ** 2 + mu * (mu - 1) / np.cos(x / 2) ** 2)
    d = 1.0 / dx**2 + V            # (1/2m)*2/dx^2 with m=1 -> 1/dx^2
    e = -0.5 / dx**2 * np.ones(Ng - 1)
    w, v = eigh_tridiagonal(d, e, select='i', select_range=(0, Nlev - 1))
    v = v / np.sqrt(dx)
    return dict(name='pt', x=x, dx=dx, V=V, E=w, psi=v, omega=0.5,
                e_n=np.arange(Nlev) * (np.arange(Nlev) + nu),
                mass=1.0, omega_h=np.sqrt(2.0), x0=np.pi / 2, nu=nu, Nlev=Nlev)


def morse_system(s=20.0, Ng=24000, Nlev=21, xmin=-0.75, xmax=9.0):
    """Foldi et al. dimensionless Morse: H = P^2 + (s+1/2)^2[e^{-2X}-2e^{-X}]
       i.e. mass 1/2.  E_m = -(s-m)^2, e_n = n(2s-n), omega = 1."""
    x = np.linspace(xmin, xmax, Ng + 2)[1:-1]
    dx = x[1] - x[0]
    A = (s + 0.5) ** 2
    V = A * (np.exp(-2 * x) - 2 * np.exp(-x))
    d = 2.0 / dx**2 + V            # p^2/2m with m=1/2  ->  -d2/dx2
    e = -1.0 / dx**2 * np.ones(Ng - 1)
    w, v = eigh_tridiagonal(d, e, select='i', select_range=(0, Nlev - 1))
    v = v / np.sqrt(dx)
    return dict(name='morse', x=x, dx=dx, V=V, E=w, psi=v, omega=1.0,
                e_n=np.arange(Nlev) * (2 * s - np.arange(Nlev)),
                mass=0.5, omega_h=2 * s + 1, x0=0.0, s=s, Nlev=Nlev)


def fix_signs(sys):
    """Anchor eigenvector signs: make the first significant lobe positive
       (analytic convention: psi_n ~ positive near the inner turning point)."""
    psi = sys['psi']
    for n in range(psi.shape[1]):
        col = psi[:, n]
        i = np.argmax(np.abs(col) > 0.05 * np.max(np.abs(col)))
        if col[i] < 0:
            psi[:, n] = -col
    return sys


def quadratures(sys):
    """X = sqrt(m*omega_h) q,  P = p/sqrt(m*omega_h), q measured from the minimum."""
    psi, x, dx = sys['psi'], sys['x'], sys['dx']
    N = psi.shape[1]
    sc = np.sqrt(sys['mass'] * sys['omega_h'])
    q = (x - sys['x0'])
    Xm = (psi.T * q) @ psi * dx * sc                       # <m|X|n>
    # momentum via central differences
    dpsi = np.gradient(psi, dx, axis=0)
    Pm = -1j * (psi.T @ dpsi) * dx / sc
    Pm = 0.5 * (Pm + Pm.conj().T)
    qm = (psi.T * q) @ psi * dx                            # bare position (dipole)
    return Xm.astype(complex), Pm.astype(complex), qm


# ---------------------------------------------------------------- GK states
def gk_coeffs(e_n, J):
    N = len(e_n)
    logrho = np.concatenate([[0.0], np.cumsum(np.log(e_n[1:]))])
    logc = 0.5 * np.arange(N) * np.log(J) - 0.5 * logrho
    logc -= logc.max()
    c = np.exp(logc)
    return c / np.linalg.norm(c)


def find_J(e_n, nbar_target=7.75, lo=1.0, hi=1e4):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        c = gk_coeffs(e_n, mid)
        nb = np.sum(np.arange(len(e_n)) * c**2)
        if nb < nbar_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- QFI
def qfi_matrix(rho, X, P, eps=1e-12):
    """2x2 QFI form in the (P, X) basis of generators G_theta = cos.P - sin.X.
       Rank-change protocol: drop eigenvalues below eps, renormalise."""
    lam, U = np.linalg.eigh(rho)
    keep = lam > eps
    lam = lam[keep]
    lam = lam / lam.sum()
    U = U[:, keep]
    Xe = U.conj().T @ X @ U
    Pe = U.conj().T @ P @ U
    dl = lam[:, None] - lam[None, :]
    sl = lam[:, None] + lam[None, :]
    W = np.where(sl > 0, 2 * dl**2 / np.where(sl > 0, sl, 1.0), 0.0)
    F = np.zeros((2, 2))
    F[0, 0] = np.sum(W * np.abs(Pe) ** 2)
    F[1, 1] = np.sum(W * np.abs(Xe) ** 2)
    # cross term:  sum_kl W_kl Re[<k|P|l> conj(<k|X|l>)]
    F[0, 1] = F[1, 0] = -np.sum(W * np.real(Pe * Xe.conj()))
    ev = np.linalg.eigvalsh(F)
    return F, float(max(ev.min(), 0.0)), float(ev.max())


def fq_theta(F, theta):
    c, s = np.cos(theta), np.sin(theta)
    return F[0, 0] * c**2 + F[1, 1] * s**2 + 2 * F[0, 1] * c * s
