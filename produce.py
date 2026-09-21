import numpy as np, json, base64
from scipy.linalg import expm
from engine import pt_system, morse_system, quadratures, gk_coeffs, find_J
from qfi import qfi_form, theta_min

TP = 2 * np.pi
NG = 401                      # interrogation-time grid (matches the thesis)
NGN = 101                     # coarse grid for the smooth robustness surface
NX = 224                      # spatial samples for the 3D packet
NH = 48                       # Husimi grid
NHG = 49                      # Husimi time samples


def signs(sys, where):
    psi = sys['psi']
    for n in range(psi.shape[1]):
        col = psi[:, n]; a = np.abs(col); thr = 0.03 * a.max()
        i = np.argmax(a > thr) if where == 'inner' else len(col) - 1 - np.argmax(a[::-1] > thr)
        if col[i] < 0:
            psi[:, n] = -col
    return sys


def make(name):
    if name == 'pt':
        s = signs(pt_system(4.0, 20000, 44), 'inner'); Nlev = 28
        s['xwin'] = (0.0, np.pi)
    else:
        s = signs(morse_system(20.0, 30000, 21, -0.75, 12.0), 'outer'); Nlev = 21
        s['xwin'] = (-0.6, 4.2)
    X, P, qm = quadratures(s)
    s.update(Xb=X, Pb=P, qmat=qm, Nlev=Nlev, X=X[:Nlev, :Nlev], P=P[:Nlev, :Nlev],
             q=qm[:Nlev, :Nlev], en=s['e_n'][:Nlev])
    s['J'] = find_J(s['en'], 7.75)
    s['c0'] = gk_coeffs(s['en'], s['J'])
    return s


def damp_rates(s):
    N = s['Nlev']; en = s['en']; q2 = np.abs(s['q'])**2
    G = np.zeros((N, N))
    for m in range(1, N):
        for n in range(m):
            G[m, n] = (en[m] - en[n])**3 * q2[m, n]
    return G / G[1, 0]


def deph_exp(s, model):
    N = s['Nlev']; en = s['en']; n = np.arange(N)
    D = (en[:, None] - en[None, :]) / (en[1] - en[0]) if model == 'energy' \
        else (n[:, None] - n[None, :]).astype(float)
    return D**2


def pure(s, g):
    c = s['c0'] * np.exp(-1j * s['en'] * g)
    return np.outer(c, c.conj())


def channel(s, g, Gt, chan, harm_prop=None):
    if Gt <= 0:
        return pure(s, g)
    if chan == 'deph_energy':
        return pure(s, g) * np.exp(-Gt * deph_exp(s, 'energy'))
    if chan == 'deph_number':
        return pure(s, g) * np.exp(-Gt * deph_exp(s, 'number'))
    if chan == 'damp_faithful':
        G = damp_rates(s); tot = G.sum(axis=1)
        r0 = pure(s, g)
        W = G.T - np.diag(tot)
        p = expm(W * Gt) @ np.real(np.diag(r0))
        r = r0 * np.exp(-0.5 * Gt * (tot[:, None] + tot[None, :]))
        np.fill_diagonal(r, p)
        return r
    raise ValueError(chan)


def qfi(s, rho):
    return qfi_form(rho.astype(complex), s['X'], s['P'], s['Xb'], s['Pb'])


def liou_harm(s, kappa):
    N = s['Nlev']; a = np.diag(np.sqrt(np.arange(1, N)), 1)
    H = np.diag(s['en']); I = np.eye(N); ad = a.conj().T
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    L += kappa * (np.kron(a, a.conj()) - .5 * np.kron(ad @ a, I) - .5 * np.kron(I, (ad @ a).T))
    return L


def q8(A, lo=None, hi=None):
    A = np.asarray(A, float)
    lo = A.min() if lo is None else lo
    hi = A.max() if hi is None else hi
    if hi <= lo:
        hi = lo + 1e-12
    Q = np.clip((A - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return base64.b64encode(Q.tobytes()).decode(), float(lo), float(hi)


gs = np.linspace(0, TP, NG)
gsn = np.linspace(0, TP, NGN)
DEPH = np.concatenate([[0.0], np.logspace(-5, 0.0, 16)])
DAMP = np.concatenate([[0.0], np.logspace(-5, 0.3, 16)])

DATA = {'gammaN': NG, 'gammaNn': NGN, 'deph_grid': [float(x) for x in DEPH],
        'damp_grid': [float(x) for x in DAMP], 'systems': {}}

for name in ['pt', 'morse']:
    s = make(name); N = s['Nlev']; en = s['en']
    print('===', name, 'J=%.2f' % s['J'])
    R = {'J': round(float(s['J']), 2), 'Nlev': N, 'e_n': [float(v) for v in en],
         'omega_h': float(s['omega_h']), 'mass': float(s['mass']),
         'c0': [round(float(v), 6) for v in s['c0']]}
    nb = float(np.sum(np.arange(N) * s['c0']**2))
    R['nbar'] = round(nb, 3)
    R['dn'] = round(float(np.sqrt(np.sum(np.arange(N)**2 * s['c0']**2) - nb**2)), 3)
    R['Trev_Tcl'] = float(TP / abs(en[1] - en[0] if False else 1))  # replaced below

    # ---- spectral / bath structure
    Gf = damp_rates(s); Gh = np.zeros((N, N))
    for m in range(1, N):
        Gh[m, m-1] = m
    R['bohr'] = [float(en[n+1] - en[n]) for n in range(min(N-1, 15))]
    R['rates_f'] = [round(float(Gf[m, m-1]), 5) for m in range(1, min(N, 16))]
    R['rates_h'] = [float(m) for m in range(1, min(N, 16))]
    q2 = np.abs(s['q'])**2
    tot = sum(np.trace(q2, offset=k) for k in range(1, N))
    R['dipole_frac'] = [round(float(np.trace(q2, offset=k) / tot), 4) for k in range(1, 5)]
    R['R8'] = round(float(((en[12] - en[4]) / ((en[1] - en[0]) * 8))**2), 3)

    # ---- closed system
    Fmin = np.zeros(NG); Fmax = np.zeros(NG); Fqq = np.zeros(NG); th = np.zeros(NG)
    for i, g in enumerate(gs):
        F, fm, fM = qfi(s, pure(s, g))
        Fmin[i], Fmax[i], Fqq[i], th[i] = fm, fM, F[0, 0], theta_min(F)
    ib = int(Fmin.argmax())
    R['closed'] = {'Fmin': [round(float(v), 4) for v in Fmin],
                   'Fmax': [round(float(v), 3) for v in Fmax],
                   'Fqq': [round(float(v), 3) for v in Fqq],
                   'theta': [round(float(v), 4) for v in th]}
    R['best'] = {'i': ib, 'g': round(float(gs[ib] / TP), 5), 'Fmin': round(float(Fmin[ib]), 3),
                 'Fqq': round(float(Fqq[ib]), 2), 'gain': round(float(Fmin[ib] / Fmin[0]), 1),
                 'Fmin0': round(float(Fmin[0]), 4), 'Fqq0': round(float(Fqq[0]), 3),
                 'FqqMax': round(float(Fqq.max()), 2),
                 'gFqqMax': round(float(gs[int(Fqq.argmax())] / TP), 4)}
    print('   closed Fmin0=%.4f best=%.3f @%.4f gain=%.1f maxFqq=%.1f'
          % (Fmin[0], Fmin[ib], gs[ib] / TP, Fmin[ib] / Fmin[0], Fqq.max()))

    # ---- robustness surfaces R(gamma,Gamma) = Fmin(g,G)/Fmin(g,0)
    Fclosed_n = np.interp(gsn, gs, Fmin)
    FmaxC_n = np.interp(gsn, gs, Fmax)
    surf = {}
    for chan, grid in [('deph_energy', DEPH), ('deph_number', DEPH), ('damp_faithful', DAMP)]:
        Rr = np.zeros((len(grid), NGN)); Pu = np.zeros((len(grid), NGN))
        RM = np.zeros((len(grid), NGN)); TH = np.zeros((len(grid), NGN))
        for j, Gt in enumerate(grid):
            for i, g in enumerate(gsn):
                r = channel(s, g, Gt, chan)
                F_, fm, fM = qfi(s, r)
                Rr[j, i] = fm / max(Fclosed_n[i], 1e-12)
                RM[j, i] = fM / max(FmaxC_n[i], 1e-12)
                TH[j, i] = theta_min(F_)
                Pu[j, i] = float(np.real(np.trace(r @ r)))
        surf[chan] = {'R': [[round(float(v), 4) for v in row] for row in Rr],
                      'Rmax': [[round(float(v), 4) for v in row] for row in RM],
                      'theta': [[round(float(v), 3) for v in row] for row in TH],
                      'purity': [[round(float(v), 4) for v in row] for row in Pu]}
        print('   ', chan, 'ok')

    # harmonic damping: full non-secular Liouvillian
    Rr = np.zeros((len(DAMP), NGN)); Pu = np.zeros((len(DAMP), NGN))
    RM = np.zeros((len(DAMP), NGN)); TH = np.zeros((len(DAMP), NGN))
    dg = gsn[1] - gsn[0]
    for j, kap in enumerate(DAMP):
        U = expm(liou_harm(s, kap) * dg)
        v = pure(s, 0.0).reshape(-1).astype(complex)
        for i in range(NGN):
            r = v.reshape(N, N)
            F_, fm, fM = qfi(s, r)
            Rr[j, i] = fm / max(Fclosed_n[i], 1e-12)
            RM[j, i] = fM / max(FmaxC_n[i], 1e-12)
            TH[j, i] = theta_min(F_)
            Pu[j, i] = float(np.real(np.trace(r @ r)))
            v = U @ v
    surf['damp_harmonic'] = {'R': [[round(float(v), 4) for v in row] for row in Rr],
                             'Rmax': [[round(float(v), 4) for v in row] for row in RM],
                             'theta': [[round(float(v), 3) for v in row] for row in TH],
                             'purity': [[round(float(v), 4) for v in row] for row in Pu]}
    print('    damp_harmonic ok')
    R['surf'] = surf

    # ---- thresholds in this convention
    thr = {}
    for chan, grid in [('deph_energy', DEPH), ('deph_number', DEPH),
                       ('damp_faithful', DAMP), ('damp_harmonic', DAMP)]:
        best = (np.array(surf[chan]['R']) * Fclosed_n[None, :]).max(axis=1)
        k = np.where(best < 2.0)[0]
        thr[chan] = float(np.exp(np.interp(np.log(2.0), np.log(best[:k[0]+1][::-1]),
                                           np.log(np.maximum(grid[:k[0]+1], 1e-12))[::-1]))) \
            if len(k) and k[0] > 0 else (None if not len(k) else float(grid[k[0]]))
    R['thresholds'] = {k: (round(v, 6) if v else None) for k, v in thr.items()}
    print('    thresholds', R['thresholds'])

    # ---- static vs dynamic assembly: explicit superoperator test
    gstar = gs[ib]
    I = np.eye(N); Hd = np.diag(en)
    LH = -1j * (np.kron(Hd, I) - np.kron(I, Hd))
    G = damp_rates(s)
    LDf = np.zeros((N * N, N * N), complex)
    for m in range(1, N):
        for n in range(m):
            if G[m, n] <= 0:
                continue
            L = np.zeros((N, N)); L[n, m] = 1.0
            Ld_ = L.conj().T @ L
            LDf += G[m, n] * (np.kron(L, L.conj()) - .5 * np.kron(Ld_, I)
                              - .5 * np.kron(I, Ld_.T))
    a = np.diag(np.sqrt(np.arange(1, N)), 1); ad = a.conj().T
    LDh = (np.kron(a, a.conj()) - .5 * np.kron(ad @ a, I) - .5 * np.kron(I, (ad @ a).T))
    nrm = lambda A: float(np.linalg.norm(A))
    sv = {'comm_faithful': nrm(LH @ LDf - LDf @ LH) / (nrm(LH) * nrm(LDf)),
          'comm_harmonic': nrm(LH @ LDh - LDh @ LH) / (nrm(LH) * nrm(LDh)),
          'cases': []}
    for kap in [0.003, 0.05]:
        for tag, LD in [('faithful', LDf), ('harmonic', LDh)]:
            dyn = (expm((LH + kap * LD) * gstar) @ pure(s, 0.).reshape(-1)).reshape(N, N)
            stat = (expm(kap * LD * gstar) @ pure(s, gstar).reshape(-1)).reshape(N, N)
            _, fd, _ = qfi(s, dyn); _, fst, _ = qfi(s, stat)
            td = float(.5 * np.sum(np.abs(np.linalg.eigvalsh(dyn - stat))))
            sv['cases'].append({'model': tag, 'kappa': kap, 'dyn': round(float(fd), 4),
                                'stat': round(float(fst), 4), 'trace': round(td, 6)})
    # purity comparison of the two damping models at matched coupling
    sv['purity'] = []
    for kap in [1e-3, 1e-2]:
        pf = float(np.real(np.trace(channel(s, gstar, kap, 'damp_faithful') ** 2 @ np.eye(N))))
        rf = channel(s, gstar, kap, 'damp_faithful')
        rh = (expm((LH + kap * LDh) * gstar) @ pure(s, 0.).reshape(-1)).reshape(N, N)
        sv['purity'].append({'kappa': kap,
                             'faithful': round(float(np.real(np.trace(rf @ rf))), 4),
                             'harmonic': round(float(np.real(np.trace(rh @ rh))), 4)})
    R['assembly'] = sv
    print('    assembly', {k: sv[k] for k in ["comm_faithful","comm_harmonic"]}, sv['cases'], sv['purity'])

    # ---- spatial packet density (closed) on a display window
    x0, x1 = s['xwin']
    xs = np.linspace(x0, x1, NX)
    idx = np.clip(np.searchsorted(s['x'], xs), 0, len(s['x']) - 1)
    psiX = s['psi'][idx][:, :N]                  # NX x N
    Vx = np.interp(xs, s['x'], s['V'])
    gsp = np.linspace(0, TP, 201)
    dens = np.zeros((201, NX))
    for i, g in enumerate(gsp):
        c = s['c0'] * np.exp(-1j * en * g)
        w = psiX @ c
        dens[i] = np.abs(w)**2
    R['xs'] = [round(float(v), 5) for v in xs]
    R['V'] = [round(float(v), 4) for v in np.clip(Vx, None, np.percentile(Vx, 99.2))]
    b, lo, hi = q8(dens, 0.0, dens.max())
    R['dens'] = {'b64': b, 'lo': lo, 'hi': hi, 'ng': 201, 'nx': NX}
    R['levels'] = [float(v) for v in (s['E'][:N] - s['E'][0]) / s['omega']]
    R['E_abs'] = [float(v) for v in s['E'][:N]]
    R['Vtop'] = float(np.max(R['V'])); R['Vmin'] = float(np.min(R['V']))

    # ---- Husimi phase space, three exposure levels of the faithful damping
    sc = np.sqrt(s['mass'] * s['omega_h'])
    Xg = (xs - s['x0']) * sc
    varP = R['best']['FqqMax'] / 4
    Pmax = 1.15 * np.sqrt(varP) * 2.2
    Xc = float(np.sum(np.abs(psiX @ s['c0'])**2 * Xg) / np.sum(np.abs(psiX @ s['c0'])**2))
    Xspan = max(abs(Xg.min() - 0), abs(Xg.max()))
    XL, XR = Xg.min(), min(Xg.max(), Xc + 3.2 * Xspan)
    xq = np.linspace(XL, XR, NH); pq = np.linspace(-Pmax, Pmax, NH)
    # coherent-state overlaps <alpha|n>
    xs_fine = np.linspace(x0, x1, 900)
    idxf = np.clip(np.searchsorted(s['x'], xs_fine), 0, len(s['x']) - 1)
    psif = s['psi'][idxf][:, :N]
    dxf = xs_fine[1] - xs_fine[0]
    Xf = (xs_fine - s['x0']) * sc
    QQ, PP = np.meshgrid(xq, pq, indexing='ij')
    gauss = np.exp(-0.5 * (Xf[None, :] - QQ.reshape(-1, 1))**2
                   - 1j * PP.reshape(-1, 1) * Xf[None, :]) * (np.pi ** -0.25)
    Ov = (gauss.conj() * np.sqrt(sc)) @ psif * dxf          # (NH*NH) x N
    ghus = np.linspace(0, TP, NHG)
    cubes = {}
    for tag, Gt in [('clean', 0.0), ('mid', float(DAMP[9])), ('hi', float(DAMP[13]))]:
        Q = np.zeros((NHG, NH, NH))
        for i, g in enumerate(ghus):
            r = channel(s, g, Gt, 'damp_faithful')
            Q[i] = np.real(np.einsum('km,mn,kn->k', Ov.conj(), r, Ov)).reshape(NH, NH)
        Q = np.maximum(Q, 0)
        b, lo, hi = q8(Q ** 0.55, 0.0, (Q ** 0.55).max())
        cubes[tag] = {'b64': b, 'lo': lo, 'hi': hi, 'Gt': Gt}
    R['husimi'] = {'cubes': cubes, 'n': NH, 'ng': NHG,
                   'X': [round(float(XL), 3), round(float(XR), 3)],
                   'P': [round(float(-Pmax), 3), round(float(Pmax), 3)]}
    print('    husimi ok')

    DATA['systems'][name] = R

json.dump(DATA, open('/home/claude/sensor_data.json', 'w'), separators=(',', ':'))
import os
print('bytes', os.path.getsize('/home/claude/sensor_data.json'))
