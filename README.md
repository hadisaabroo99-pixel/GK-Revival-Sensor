# Gazeau–Klauder revival displacement sensor — interactive 3D model

An interactive, browser-based model of a displacement sensor built from a
Gazeau–Klauder (GK) coherent state undergoing fractional revival in a bounded
anharmonic potential, and of how that sensor degrades under Markovian
decoherence. It accompanies the MPhil dissertation *Quantum Fisher information
of Gazeau–Klauder coherent states under decoherence* (Hadisa Abroo).

## Contents

| File | What it is |
|---|---|
| `gk-revival-sensor.html` | The model. A single self-contained HTML file. Open it in any modern browser; no installation, no network connection required. |
| `engine.py` | Grid diagonalisation of the two potentials, eigenvector sign anchoring, quadrature matrix elements, GK state construction. |
| `qfi.py` | Completeness-corrected SLD quantum Fisher information, returned as a 2×2 form over displacement direction. |
| `produce.py` | The production run. Builds every number and array the model displays and writes `sensor_data.json`. |
| `sensor_data.json` | The output of `produce.py`, embedded in the HTML file. |

To regenerate the data: `python produce.py` (requires NumPy and SciPy).
The HTML file embeds `sensor_data.json` directly, so rebuilding the data
requires re-embedding it.

## The two systems

Both have exactly quadratic spectra with opposite curvature, so neither has a
superrevival hierarchy and a sweep over one revival period is exhaustive.

- **Trigonometric Pöschl–Teller**, ν = 4, e_n = n(n+ν), 28 levels retained
  from a 20 000-point grid.
- **Morse** in the dimensionless form of Földi et al., s = 20,
  e_n = n(2s−n), 21 levels from a 30 000-point grid (20 bound states plus the
  box-discretised threshold state).

Probes are matched at mean excitation n̄ = 7.75, which fixes the action labels
J = 95.85 (PT) and J = 239.80 (Morse).

## Method

1. Diagonalise each Hamiltonian on a grid as a tridiagonal matrix.
2. Anchor eigenvector signs to the analytic convention of each family
   (inner lobe for PT, outer tail for Morse). Getting this wrong changes
   F_Q^min(γ=0) for Morse by a factor of 23 without any other visible symptom.
3. Build GK coefficients c_n = J^(n/2)/√ρ_n and solve for J at fixed n̄.
4. Form the quadrature matrices in the energy eigenbasis, with
   X̂ = √(mω_h) q̂ and ω_h the harmonic frequency about the potential minimum.
5. Evolve. Free evolution is a phase per level. Energy-basis dephasing decays
   ρ_mn by exp[−Γt(e_m−e_n)²/(e₁−e₀)²]. Transition-resolved amplitude damping
   is secular, so populations follow a Pauli master equation solved by one
   matrix exponential and coherences decay at ½(Γ_m+Γ_n) with no coupling
   between coherences. The harmonic dissipator 𝒟[â] is not secular with
   respect to these Hamiltonians and is integrated as a full Liouvillian.
6. Evaluate the quantum Fisher information from the SLD spectral sum with an
   exact out-of-support closure, computed against an enlarged basis. Because
   F_Q is a quadratic form in the generator, the 2×2 form in (P̂, X̂) gives
   the information along every displacement direction from one diagonalisation;
   its eigenvalues are F_Q^min and F_Q^max.

## Conventions

- Natural units, ℏ = 1. γ = ωt is dimensionless interrogation time; one
  revival period is γ ∈ [0, 2π).
- Displacement generators are Ĝ_θ = cos θ P̂ − sin θ X̂. A coherent state of
  the reference oscillator gives F_Q = 2 at any amplitude (the shot-noise
  reference).
- **Decoherence is parametrised as a dimensionless exposure Γt with all rates
  normalised to the 1→0 transition.** Thresholds quoted by the model are in
  this convention and are not directly comparable to a bare coupling constant
  κ multiplying an unnormalised rate.
- Benchmarks at matched n̄ = 7.75: coherent state F_Q = 2 in every direction;
  squeezed state F_Q^(q) = 65.9, F_Q^min = 0.0607.

## What the model shows

Four modules on a bench, orbitable and zoomable:

1. **Trap and probe** — the potential, the level ladder with rung brightness
   set by the probe populations, and |ψ(x,γ)|² as a live ridge.
2. **Phase space** — the Husimi Q-function over (X, P).
3. **Information by direction** — F_Q(θ) as a polar curve on logarithmic
   decade rings, with the coherent-state reference marked. This is the
   clearest statement of the central mechanism: at γ = 0 the curve is a
   pinched figure-eight; at the fractional revival it is nearly round.
4. **Ladder and bath coupling** — transition-resolved emission rates against
   the harmonic 𝒟[â] rates, arc thickness proportional to rate.

Readouts update live: F_Q^min, F_Q^(q), isotropy, purity, surviving fraction,
and ratios against both benchmarks.

## Principal results reproduced

| Quantity | Pöschl–Teller | Morse |
|---|---|---|
| F_Q^min at γ = 0 | 0.0569 | 0.1839 |
| F_Q^min at best γ | 9.750 | 25.539 |
| Gain from free evolution alone | 171× | 139× |
| Maximum F_Q^(q) | 275.1 | 27.3 |
| Advantage over coherent state | 4.9× | 12.8× |
| Advantage over squeezed state | 161× | 421× |
| Emission rate ratio to harmonic model at n = 8 | 13.2 | 0.32 |
| Dephasing exponent ratio at coherence order 8 | 16.0 | 0.38 |
| [ℒ_H, ℒ_𝒟] relative norm, transition-resolved | 0 | 0 |
| [ℒ_H, ℒ_𝒟] relative norm, harmonic 𝒟[â] | 1.2×10⁻³ | 2.5×10⁻³ |

Dissipation during revival assembly is exactly equivalent to dissipation on a
statically prepared superposition for any channel secular with respect to the
system Hamiltonian; the model verifies this to machine precision, and shows
the spurious asymmetry the harmonic dissipator produces instead.

## Known limitations

- The decoherence axis uses the normalisation stated above. Absolute
  thresholds will differ from any treatment that normalises the coupling
  differently.
- The Husimi function is plotted rather than the Wigner function, so the
  finest sub-Planck structure is Gaussian-smoothed and negativity is not
  visible.
- The smoothing applied to the spatial density under decoherence is
  illustrative. The numerical readouts are computed from the density matrix
  and are not affected.

## Requirements

A browser with WebGL. The HTML file has no external dependencies: three.js
(r128, MIT licence) is embedded.

## Citation

Please cite the archived version DOI for the release you used.

## Licence

<!-- Choose one and state it here, e.g. CC BY 4.0 for the model and
     MIT for the code. Zenodo will ask for this during deposit. -->
