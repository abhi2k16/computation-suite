# Important Lessons in Nonlinear FEM

**Source:** Ted Belytschko, Wing Kam Liu, Brian Moran, Khalil Elkhodary, *Nonlinear Finite Elements for Continua and Structures*, 2nd edition (Wiley, 2014) — 834 pages, 13 chapters + 5 appendices. Study conducted by extracting the book's text (via `pypdf`) and reading it in eight chapter-group passes (Ch1–2, Ch3, Ch4, Ch5, Ch6, Ch7+10, Ch8–9, Ch11–13), each summarized below with section/equation citations.

This document is the nonlinear-FEM counterpart to `fem_implementation_lessons.md` (the earlier study of Gockenbach's *Understanding and Implementing the Finite Element Method*). Where that book covers linear FEM implementation mechanics, this book covers everything that changes once the problem becomes geometrically, materially, or contact nonlinear: large-deformation kinematics, stress/strain measures and objectivity, constitutive integration (plasticity return-mapping), nonlinear solution procedures (Newton, arc-length), element locking/hourglassing, ALE, contact-impact, XFEM, multiresolution continuum theory, and crystal plasticity.

---

## The distilled lessons

These are the ideas that recur across chapters and matter most for actually writing/debugging a nonlinear FE code.

1. **Nonlinearity has three independent sources** — geometric (large deformation/rotation), material (path-dependent constitutive response), and contact/boundary (changing contact area, friction) — and each demands different numerical machinery. The book doesn't treat this as a checklist so much as a lived reality: geometric nonlinearity produces the geometric/stress stiffness matrix (Ch. 6), material nonlinearity demands return-mapping stress-update algorithms with algorithmic tangents (Ch. 5), and contact demands complementarity conditions and non-smooth Newton handling (Ch. 10).

2. **The deformation gradient F is not a strain measure.** It equals a pure rotation R under rigid body motion, so any strain measure must be built from F (via FᵀF or its rate) in a way that provably vanishes under rigid rotation. Green strain E = ½(FᵀF − I) satisfies this exactly; F itself does not (Ch. 3).

3. **Rate-of-deformation D is a rate-only quantity, not a valid total strain measure.** Its integral over a closed deformation cycle is generally nonzero (proven by explicit worked example, Ch. 3, Example 3.7) — this has real consequences for hypoelastic materials whose stress is driven by integrating D-based rate laws.

4. **Stress/strain measures come in work-conjugate pairs, and mixing them breaks the virtual work statement.** Cauchy stress σ pairs with rate-of-deformation D (current configuration); nominal/PK1 stress P pairs with Ḟ; PK2 stress S pairs with Green strain rate Ė (reference configuration). Using the wrong pairing silently produces an incorrect internal-power expression (Ch. 3, Ch. 4's Voigt-notation warnings are the same lesson at the matrix-implementation level).

5. **A naive material stress-rate is not objective (frame-indifferent) under rigid rotation, so hypoelastic rate laws need a corotational/objective rate** — Jaumann, Truesdell, or Green–Naghdi. These are NOT interchangeable without recalibrating material constants, and the Jaumann rate famously produces oscillatory shear stress under monotonic simple shear (Ch. 3, Example 3.13) — a textbook cautionary tale about picking an objective rate without understanding its behavior.

6. **Total Lagrangian and Updated Lagrangian formulations are mathematically identical** — UL is literally the special case of TL where the reference configuration is redefined to be the current configuration (proven directly, Ch. 4, eq. 4.9.26–4.9.29). Terms can be mixed TL/UL within the same code depending on which is more convenient for a given constitutive law ("evaluate each term in whatever configuration is most convenient," Ch. 4 p.168).

7. **Voigt notation has two different conventions and mixing them silently corrupts energy calculations**: the kinematic rule (strain-type quantities, factor of 2 on shear/off-diagonal terms) versus the kinetic rule (stress-type quantities, no factor of 2). The B-matrix uses the kinematic rule; the stress vector it's dotted against uses the kinetic rule (Ch. 4).

8. **Return-mapping stress-update algorithms follow a universal elastic-predictor/plastic-corrector structure.** Fully implicit backward Euler (flow direction evaluated at the end of the step) needs derivatives of the flow rule — more work, better convergence. Semi-implicit backward Euler (flow direction frozen at the start of the step) gives a closed-form update with no local Newton loop, but the resulting tangent is generally non-symmetric even for associative plasticity. Radial return for J2 plasticity reduces to a single scalar equation because the von Mises yield surface is a circle in deviatoric stress space, so the flow direction stays fixed throughout the correction (Ch. 5).

9. **The algorithmic (consistent) tangent is not the same as the continuum tangent, and using the continuum tangent in Newton iteration causes spurious loading/unloading and destroys quadratic convergence.** The algorithmic tangent is the exact derivative of the discrete stress-update *map*, not of the continuum rate law (Ch. 5, eq. 5.9.38; same lesson reappears from the solver side in Ch. 6 §6.4.7 as "algorithmically consistent tangent stiffness").

10. **Incremental objectivity (Hughes–Winget) is a hard requirement for large-rotation hypoelastic-plastic stress updates:** σ_{n+1} = Q σ_n Qᵀ must hold when F_{n+1} = Q F_n (pure rigid rotation). Failing this generates spurious stress under rotation alone — a classic large-rotation UMAT bug (Ch. 5, Ch. 9's Hughes-Winget rotation-tensor update is the same fix applied to shell directors).

11. **Multiplicative decomposition F = Fe·Fp sidesteps the objectivity problem architecturally rather than through careful rate integration.** The elastic Green strain computed on the (fictitious) intermediate configuration is provably rotation-invariant, so stress can be a direct hyperelastic potential evaluation rather than an integrated rate law — this also removes the isotropy restrictions that hypoelastic-plastic models are stuck with (Ch. 5; specialized to slip systems in crystal plasticity, Ch. 13).

12. **Newton's method for the FE residual rarely achieves textbook quadratic convergence** because real constitutive laws (elastic-plastic, contact) are not continuously differentiable — the Jacobian loses regularity exactly where you need it most (near yield, near contact, near instability). Convergence criteria must be scaled (force-norm, displacement-norm, or energy-norm) or they're meaningless across problems (Ch. 6).

13. **Geometric (stress) stiffness is what linear FEM never needs and nonlinear FEM cannot avoid.** It arises from existing stress acting through changing geometry (S·Ḟᵀ term) and is absent from linear FE because Ḟ≈0 there. It's exactly what makes buckling analysis possible — it can turn an otherwise positive-definite stiffness matrix indefinite under compression (Ch. 6).

14. **Explicit vs. implicit is not simply "conditionally stable and cheap" vs. "unconditionally stable and expensive."** The book's real decision criteria are PDE type, data smoothness, and which response (low-frequency structural vs. high-frequency wave) is of interest — and severity of contact/nonlinearity can override the formal PDE-type argument (parabolic shell problems are still often run explicit because Newton can't converge through contact noise) (Ch. 6).

15. **Material instability is a genuinely different phenomenon from numerical/structural instability, and it changes the type of the governing PDE** (hyperbolic ↔ elliptic) at the point of instability. Once a rate-independent softening material goes unstable, strain localizes to a set of measure zero — refining the mesh makes this worse, not better, and dissipated energy goes to zero without an explicit regularization (gradient, nonlocal, Cosserat, or rate-dependent) (Ch. 6; Ch. 12's multiresolution continuum theory is presented explicitly as one systematic fix, supplying an intrinsic length scale from real microstructure rather than an ad hoc regularization parameter).

16. **Locking is fundamentally a constraint-counting problem: the element's interpolation cannot represent an admissible (isochoric or pure-bending) deformation exactly**, so the numerics lock onto whatever spurious extra stiffness the mismatched interpolation can't eliminate. Reduced integration cures locking by simply not sampling the problematic strain component, but this creates rank-deficient, spurious zero-energy hourglass modes that must be separately controlled (stiffness-based or viscosity-based) without over-stiffening the element (Ch. 8).

17. **Shear and membrane locking in beams/shells scale with slenderness (ℓ/h)², not a fixed material parameter** — so they get worse exactly as the elements are pushed toward the thin-structure regime they're meant for. Assumed-strain / Barlow-point / ANS-type methods fix this by sampling strain where the parasitic term happens to vanish (Ch. 9).

18. **A "correct" (patch-test-passing) element formulation and a practically "robust" one are not the same thing** — the book's own comparison table shows fully-integrated elements are more accurate but abort under large mesh distortion, while cheaper, less rigorous reduced-integration elements survive extreme distortion without failing. Crash/impact codes systematically trade formulation purity for robustness (Ch. 9).

19. **Contact is a complementarity condition (Kuhn–Tucker), and the three standard enforcement methods trade off the same way general constraints do:** Lagrange multipliers enforce contact exactly but need a separate multiplier mesh and get noisy at high velocity; penalty methods need no extra unknowns but only approximately enforce non-penetration and are sensitive to parameter choice; augmented Lagrangian combines both, generally preferred when exact enforcement and good conditioning both matter (Ch. 10; same tradeoff triangle appears generically for constraints in Ch. 6 §6.3.8).

20. **XFEM and multiresolution continuum theory both attack the same underlying failure of classical FEM/continuum theory: needing an ever-finer mesh to resolve a feature (a crack, a localization band) that the theory has no intrinsic length scale to describe.** XFEM enriches the FE approximation via partition-of-unity to represent a discontinuity without a conforming mesh; MCT enriches the continuum kinematics itself with sub-scale deformation measures to give localization a physically-grounded finite width (Ch. 11, Ch. 12).

---

## Chapters 1–2: Introduction and 1D Lagrangian/Eulerian Finite Elements

### 1. What makes a problem "nonlinear" and why it needs different solution machinery

Belytschko et al. do not present the classic three-way taxonomy (geometric / material / boundary-contact nonlinearity) as a formal framework in these chapters — that packaging is more common in other texts (e.g. Bathe, Crisfield). Instead, Chapter 1 frames nonlinearity through motivating engineering applications and through the five-step analysis workflow it lays out in §1.1: (1) development of a model, (2) formulation of the governing equations, (3) discretization, (4) solution, (5) interpretation of results. The book stresses that steps 2–4 are "within the analysis code," while the analyst owns 1 and 5 — nonlinear analysis is unforgiving of a black-box mentality (§1.1): "nonlinear finite element analysis confronts the analyst with many choices and pitfalls."

Historically (§1.2), the book implicitly names the two classic sources of nonlinearity describing Sam Key's HONDO code (Sandia, 1975), which "treated both material nonlinearities and geometric nonlinearities" — the closest the text comes to the standard taxonomy, appearing as incidental historical language rather than a defined category. Contact-impact is repeatedly cited as a third major complicating physics (Hallquist's DYNA codes, Hibbitt/ABAQUS, Hughes's contact-impact work) but treated as a numerically hard interface/boundary-condition problem addressed later (regularization, penalty methods — "hidden regularizations, such as penalty methods in contact-impact," §1.1).

Motivating real-world drivers named explicitly: automotive crashworthiness (replacing crash tests), sheet-metal forming/extrusion/casting, electronics drop-test simulation, biomechanics (human head modeling), and nuclear safety.

Chapter 2's own model problem is illuminating by contrast: it deliberately uses a linear elastic constitutive law (Eq. 2.2.19) so the resulting semidiscrete equations Ma = f^ext − f^int (Eq. 2.4.19) remain amenable to explicit central-difference integration without solving any algebraic system — "explicit integration is simpler than static linear stress analysis" (§2.12). Harder implicit/iterative solution of genuinely nonlinear algebraic systems is deferred to later chapters; §2.13 flags that dropping inertia turns the equilibrium equations into "either nonlinear algebraic equations or ordinary differential equations, depending on the character of the constitutive equation."

### 2. Lagrangian vs. Eulerian description in 1D

Introduced in §1.4 as one of three classification aspects of a formulation (Belytschko, 1977): mesh description, kinetic description (stress tensor/momentum form choice), kinematic description (strain measure choice).

Core definitions (Eqs. 1.4.1–1.4.8): motion x = φ(X,t); Lagrangian (material) description expresses fields in terms of X; Eulerian (spatial) description expresses fields in terms of x via X = φ⁻¹(x,t). A worked 1D example shows the same physical velocity field yields two mathematically different functions v(X,t) vs. v̄(x,t) — "if a field variable is expressed in terms of different independent variables, then the functions must be different."

**Mesh philosophies:** in a Lagrangian mesh, nodes coincide with material points (nodal coordinates X_I are time-invariant, no material crosses element interfaces, quadrature points track the same material point forever). In an Eulerian mesh, nodes coincide with spatial points (material passes across element interfaces, the material point at a given quadrature point changes with time).

**Tradeoffs stated explicitly:** because quadrature points track material points, Lagrangian meshes are essential for history-dependent materials ("this complicates the treatment of materials for which the stress is history-dependent" in Eulerian meshes). Lagrangian meshes keep boundary/interface nodes on the boundary/interface automatically, simplifying BC imposition, but suffer severe distortion under large deformation ("the magnitude of deformation that can be simulated with a Lagrangian mesh is limited"). Eulerian elements never distort but the mesh must be large enough to enclose the deformed material, and boundary/interface tracking needs extra machinery. A 2D pure-shear example (Fig. 1.2) is used for intuition: "a Lagrangian mesh is like an etching on the material... an Eulerian mesh is like an etching on a sheet of glass held in front of the material." ALE is mentioned at the end of §1.4 as blending both, deferred to Chapter 7.

### 3. Total vs. Updated Lagrangian in the 1D setting

First introduced at chapter level in §2.1, before the general multi-dimensional treatment in Chapter 4:

- **Total Lagrangian**: formulated in terms of Lagrangian stress/strain measures, derivatives/integrals taken with respect to material coordinates X.
- **Updated Lagrangian**: formulated in terms of Eulerian stress/strain measures, derivatives/integrals taken with respect to spatial coordinates x.

| | Total Lagrangian | Updated Lagrangian |
|---|---|---|
| Dependent var | displacement u(X,t) | velocity v(X,t), stress σ(X,t) |
| Stress measure | nominal stress P = T/A₀ | Cauchy stress σ = T/A |
| Strain measure | engineering strain ε = F−1 | rate-of-deformation D_x = ∂v/∂x |
| Domain | reference config [X_a,X_b] | current config [x_a(t),x_b(t)] |

The book proves the two are equivalent in §2.8.5: using the chain rule and σA = PA₀ (mass conservation), the updated-Lagrangian internal force integral reduces exactly to the total-Lagrangian expression. Practical takeaway stated directly: "either of these formulations may be used for different nodal forces in the same calculation" — e.g. internal forces via UL, external forces via TL, in the same program. The book candidly notes the pedagogical redundancy this creates and says it's "not unwise to skip one of these Lagrangian formulations" in a first course, but both are needed to read the literature/software.

An Eulerian formulation (§2.9–2.11) is presented as a third, genuinely distinct option: no reference configuration exists, mass conservation must be a full PDE (not the algebraic ρJ=ρ₀), and the momentum equation carries an explicit convective/transport term.

### 4. Weak-form derivation and discretization for the 1D nonlinear bar

The derivation (§2.3) is the book's reusable "virtual work schema": multiply the strong-form momentum equation by test function δu, integrate over the reference domain, integrate by parts to move the derivative onto the test function (producing a boundary term that becomes external traction virtual work), yielding the weak form (Eq. 2.3.4). The book distinguishes classical C¹ smoothness (needed for a literal strong form) from the weaker C⁰ continuity actually used in FEM — with C⁰ trial/test functions the strong form regained from the weak form includes an extra interior jump condition at nodes/interfaces, essential for handling material interfaces and discontinuous cross-sections.

Terms get physical names: internal virtual work δW^int = ∫δF·P A₀ dX, external virtual work δW^ext, kinetic virtual work δW^kin — assembled into the Principle of Virtual Work: δW ≡ δW^int − δW^ext + δW^kin = 0 (Box 2.1). The updated-Lagrangian analog, integrating over the current domain, is the Principle of Virtual Power.

Discretization proceeds by substituting FE trial/test functions and invoking arbitrariness of nodal δu_I, yielding nodal force definitions (internal, external, mass matrix), assembled into Ma = f^ext − f^int (Eq. 2.4.19) — explicitly likened to Newton's second law for a particle system when mass is lumped (row-sum technique, shown to conserve total momentum). A stated design rule: nodal forces must always be conjugate to nodal displacements in the sense of work, or symmetry of stiffness/mass matrices is lost.

The chapter closes (§2.12) using this discretization to build a full explicit central-difference algorithm (Box 2.5), including the critical time-step stability criterion Δt_crit = ℓ₀/c₀ with c₀ the material wave speed — flagged as "described in detail in Chapter 6."

### 5. Explicit warnings, rules of thumb, and motivating context

- "Nonlinear finite element analysis confronts the analyst with many choices and pitfalls. Without an understanding of the implication and meaning of these choices and difficulties, an analyst is at a severe disadvantage." (§1.1)
- Nonlinear solids "can undergo instabilities, their response can be sensitive to imperfections, and the results can depend dramatically on material parameters. Unless the analyst is aware of these phenomena, misinterpretation of simulation results is quite possible." (§1.1)
- Regularization is a double-edged tool: "regularization procedures are often not based on physical phenomena and in many cases the constants associated with the regularization are difficult to determine" — penalty methods in contact-impact are explicitly called a "hidden regularization" (§1.1).
- On PDE character: removing the inertial term changes the governing equation from hyperbolic (wave equation) to elliptic — statics and dynamics are not just "the same equations minus a term," they have fundamentally different mathematical character (§2.2.5).
- Historical cautionary tale: WRECKER, Belytschko's early 3D nonlinear crash code (1972), cost ~$30,000 (three years of an Assistant Professor's salary) to run a 300-element, 20 ms simulation; DOT pulled funding in 1975 judging simulation "too expensive," redirecting to physical testing — illustrating how compute cost historically gated nonlinear FE adoption until vectorized explicit codes with one-point-quadrature elements and hourglass control made it practical.
- Practical guidance: "while the explicit method is probably best suited for simulating sheet metal forming operations, in the springback simulation implicit methods are more suitable... A robust capability today requires the availability of both classes of methods." (§1.2)

---

## Chapter 3: Continuum Mechanics

### 1. Deformation gradient F

Defined (§3.2.6, Eq. 3.2.19) as F_ij = ∂x_i/∂X_j, the Jacobian of the motion x = φ(X,t); maps reference line segments to current ones, dx = F·dX. J = det(F) converts volume/area integrals between configurations; admissibility requires F continuously differentiable, one-to-one, J > 0 (the last follows from mass conservation, not just invertibility).

**F is explicitly not a strain measure.** Under a pure rigid rotation x = R·X, F = R — nonzero and non-identity in general (Example 3.2). A strain measure "must vanish for any rigid body motion, and in particular for rigid body rotation" (§3.3) — F fails this, which is precisely why strain tensors are built from F (via FᵀF or its rate) rather than used directly.

**Polar decomposition** (§3.7.1): F = R·U = V·R (Eq. 3.7.1, 3.7.7), R orthogonal (rotation), U/V symmetric positive-definite stretch tensors. U = (FᵀF)^(1/2), computed via spectral decomposition; R = F·U⁻¹. Only line segments aligned with U's principal directions rotate exactly by R.

### 2. Strain measures: Green strain E vs. rate-of-deformation D

The chapter deliberately restricts itself to two strain measures "most widely used in finite element methods" (§3.3): Green strain E and rate-of-deformation D.

**Green strain** (§3.3.1): ds² − dS² = dX·2E·dX (Eq. 3.3.1); closed form E = ½(FᵀF − I). Proven to vanish identically under rigid rotation (F=R ⟹ E=0), the key validating property. **E is path-independent (a valid total strain measure)** because it is literally the difference of squared lengths between two fixed configurations — its rate integrates trivially back to itself over a closed cycle.

**Rate-of-deformation** (§3.3.2–3.3.3): velocity gradient L = ∇v splits as L = D + W (symmetric D, skew spin W). D measures the rate of change of squared length of a *current* line element — explicitly a rate-only quantity defined w.r.t. the current configuration. D vanishes in rigid body motion (satisfying the strain criterion instantaneously, but only as a rate).

**The classic counterexample (Example 3.7, pp.101–104):** an element is taken through a closed four-stage deformation cycle returning exactly to F=I (so Green strain there is exactly zero, consistent with path-independence). D is computed in each stage and integrated over the cycle — the result is manifestly nonzero. The book's conclusion: "the integral of the rate-of-deformation over a cycle ending in the initial configuration does not vanish... the integral of the rate-of-deformation is not a good measure of total strain. This has significant repercussions for hypoelastic materials." The error is second-order in the deformation constants — negligible for small deformations, real for finite ones.

### 3. Stress measures and work conjugacy

Three stress measures (Box 3.1, §3.4.1): **Cauchy stress σ** (true/physical, current config, symmetric); **nominal stress P** (transpose of PK1, two-point tensor, current-config force over reference-config area, generally not symmetric — different authors swap "nominal"/"PK1" naming, a real literature trap); **second Piola-Kirchhoff S** (fully reference-configuration, symmetric, but not physically interpretable as a direct force/area — Example 3.9 shows S₁₁ = σ_x(A₀/A)² for uniaxial stress, distinct from both true and engineering stress).

**Work conjugacy** (§3.5.7/3.6.4–3.6.5, Box 3.4): σ pairs with D (w_int = σ:D, current config); P pairs with Ḟ (ρ₀ẇ_int = P:Ḟᵀ, reference config); S pairs with Ė (ρ₀ẇ_int = S:Ė, reference config). These are algebraically equivalent via J, F, and symmetric/antisymmetric decomposition. Mismatching a stress with a non-conjugate strain-rate produces an incorrect internal-power/virtual-work expression — the root of the Voigt-notation warnings that reappear in Chapter 4.

### 4. Objective (frame-invariant) rates

**The problem** (§3.7.2): the ordinary material time derivative Dσ/Dt of Cauchy stress is not usable directly in rate-type constitutive laws. Counterexample: rigidly rotate a prestressed bar (D=0 throughout, no deformation). Physically an observer riding with the body sees no stress change, but fixed-frame Cauchy stress components do change during rotation, so Dσ/Dt ≠ 0 — yet a naive hypoelastic law Dσ/Dt = C:D predicts zero. "Something must be missing."

**Three objective rates (Box 3.5, §3.7.3–3.7.4):**
- **Jaumann rate**: σ∇J = Dσ/Dt − W·σ + σ·W, using the spin tensor W — equivalently the corotational rate of Cauchy stress.
- **Truesdell rate**: σ∇T = Dσ/Dt + σ(div v) − L·σ − σ·Lᵀ — contains the same spin terms plus extra D-dependent terms; reduces to Jaumann exactly in pure rigid rotation (D=0), but differs once the body deforms.
- **Green-Naghdi rate**: σ∇G = Dσ/Dt − Ω·σ − σ·Ωᵀ, using the polar-decomposition angular velocity Ω=Ṙ·Rᵀ instead of spin W — "differs from the Jaumann rate only in using a different measure of the rotation of the material... this markedly changes the behavior of the material model."

**Explicit non-interchangeability warning** (§3.7.4): "if the two laws are to model the same material response, C^σT will differ from C^σJ. For this reason, we will add superscripts to specify the objective rate associated with the material response tensor." The same numerical material constants cannot be reused across different objective-rate formulations without recalibration.

**Pathological example (Example 3.13, simple shear):** integrating the same hypoelastic law σ∇ = λ(trD)I + 2μD under each rate gives strikingly different results — Jaumann gives σ_xy = μᴶ sin t (**oscillatory shear stress under monotonically increasing shear strain**, along with spurious oscillating normal stresses), Truesdell gives unbounded linear growth, Green-Naghdi gives yet another trigonometric closed form. The book's blunt conclusion: "it is a misuse of material models to employ the same constants with different objective rates." Fluid mechanics rarely needs this machinery because Newtonian laws relate stress (not stress-rate) directly to D.

### 5. Explicit misconceptions/warnings flagged in the chapter

- A strain measure must vanish under rigid rotation, or it predicts spurious stress under pure rotation — "the key reason why the usual linear strain displacement equations are abandoned in nonlinear theory."
- The corotational formulation is "often confusing to experienced mechanicians" who mistakenly interpret the rotated ("hatted") basis as a curvilinear coordinate system requiring extra derivative terms — it is a rotated but otherwise ordinary Cartesian basis, no extra terms needed (§3.4.3).
- "Nominal stress" and "PK1 stress" naming conventions are swapped between this book and other classic references (Malvern vs. Truesdell/Noll/Ogden/Marsden-Hughes) — a genuine literature-reading trap.
- Once stress is not symmetric (nominal stress P), left vs. right divergence placement matters and gives different results (§3.5.5) — "it is important to become accustomed to placing the divergence operator where it belongs."
- PK2 and nominal/PK1 stress are "nonphysical," awkward for plasticity (which needs yield surfaces in terms of physical Cauchy stress) even though mathematically convenient for path-independent elasticity.
- Objective rates are "frequently used in current finite element software" yet are not freely interchangeable — a real, live pitfall, not a historical curiosity.

---

## Chapter 4: Lagrangian Meshes

### 1. TL vs UL: weak forms and the equivalence argument

**Updated Lagrangian — principle of virtual power** (§4.3, Box 4.2): derived by testing the momentum equation with δv and integrating over the current configuration Ω. δP_int = ∫_Ω δD:σ dΩ (only the symmetric part of the velocity gradient survives, since δW:σ = 0 by symmetry of Cauchy stress); δP_ext, δP_kin follow analogously. Weak form: δP_int − δP_ext + δP_kin = 0 (Box 4.2.1), proven equivalent to the strong form both directions.

**Total Lagrangian — principle of virtual work** (§4.8, Box 4.5): same procedure with test displacement δu, integrated over the reference configuration Ω₀, using nominal stress P. δW_int = ∫_Ω₀ δḞ:P dΩ₀ (Box 4.5.2) — the ∫P:δḞ dV₀ form, work-conjugate per Box 3.4.

**The explicit equivalence claims:** §4.1: "the two are basically identical. Any of the expressions in the updated Lagrangian formulation can be transformed to the total Lagrangian formulation by transformations of tensors and mappings of configurations." §4.7.2: internal force f_int = ∫_Ω₀ B₀ᵀP dΩ₀ is written "to stress the analogy to the updated Lagrangian form: if we consider the current configuration to be the reference configuration, B₀ is replaced by B, Ω₀ by Ω, and P by σ, we obtain the updated Lagrangian form."

**The formal proof that UL is the special case of TL with the reference redefined to the current configuration** (pp.214–215, after Eq. 4.9.26): letting the configuration at time t become the reference gives F=I, B₀=B, J·dΩ₀=dΩ, S=σ, which collapses the TL internal-force expression exactly to the UL form — "a helpful trick which we will use again later" (i.e. this is the mechanism underlying corotational/incremental-updated formulations generally). Earlier (§4.4.9, p.168): "each term in the discrete equations should be evaluated in whatever configuration is most convenient" — the book's general license to mix TL-style and UL-style term evaluations within one implementation, e.g. UL when the constitutive law is Cauchy-stress-based, TL when it is PK2-based (§4.7.1: "when the constitutive equations are expressed in terms of σ it is advantageous to use the updated Lagrangian formulation").

### 2. Discretization: internal force vector and the B-matrix

**UL** (§4.4, Box 4.3): f_int_iI = ∫_Ω σ_ji(∂N_I/∂x_j) dΩ = ∫_Ω Bᵀσ dΩ — integrated over the *current* volume, spatial derivatives B_Ij = ∂N_I/∂x_j. Element algorithm: form L=Bv, D=½(L+Lᵀ), get σ from the constitutive law, accumulate f_int += Bᵀσ J_ξ w_Q.

**TL** (§4.9, Box 4.6): F = I + H with H = B₀u, B₀_jI = ∂N_I/∂X_j (reference-configuration derivatives). f_int_iI = ∫_Ω₀ P_ji(∂N_I/∂X_j) dΩ₀ = ∫_Ω₀ B₀ᵀP dΩ₀ — integrated over the *fixed* reference volume, never remeshed. Green strain is computed as E = ½(H+Hᵀ+HᵀH) using the displacement gradient H rather than directly from F "because the resulting computation is susceptible to round-off errors for small strains" (p.212).

**Critical practical warning** (pp.213–214): "the reader should be cautioned about one characteristic of the B₀ matrix: although it carries a subscript zero, the matrix B₀ is not time invariant" — B₀ depends on the current deformation gradient F and must be recomputed every step/iteration even though it's integrated over a fixed reference domain. The mass matrix, by contrast, is genuinely time-invariant in both formulations and need only be computed once at t=0.

### 3. Voigt notation: kinematic vs. kinetic conventions

Two different Voigt packing rules, and the book is explicit they must be paired correctly:

- **Kinematic Voigt rule** (strain/rate-of-deformation quantities): off-diagonal shear components get a factor of 2 — {D} = [D_x, D_y, 2D_xy]ᵀ.
- **Kinetic Voigt rule** (stress/force quantities: σ, S, P): no factor of 2 — {σ} = [σ_x, σ_y, σ_xy]ᵀ.

Named explicitly at Eq. 4.9.21: the B₀ matrix is converted "by the Voigt kinematic rule," while S "is converted to S_a by the kinetic Voigt rule." **Implementation pitfall:** the B-matrix (kinematic rule) is dotted against the stress vector (kinetic rule) in f_int = ∫Bᵀσ dΩ or {D}ᵀ{σ}. Applying the factor of 2 to both, or to neither, silently corrupts the shear contribution to internal power/energy without necessarily causing an obvious blow-up.

### 4. DOF-transformation rule (§4.5.5)

For nodal DOFs related by a linear transformation d̂ = T d, work-invariance (δdᵀf = δd̂ᵀf̂) proves: **force transforms as f̂ = Tᵀf** (Eq. 4.5.35, holds for both internal and external nodal forces); **mass matrix transforms as M̂ = TᵀMT** (Eq. 4.5.38) — but only when T is *time-independent*; if T is time-dependent an extra Coriolis-like term appears (Eq. 4.5.41), so the clean rule strictly requires a fixed (non-rotating) transformation; **stiffness matrix transforms as K̂_tan = TᵀK_tanT** (Eq. 4.5.42), analogous.

**Mechanisms this underlies:** master–slave tie constraints (Example 4.5 — slave-node velocities expressed as a linear kinematic constraint on master-node velocities, joining meshes of different element size without transition elements) and corotational formulations (Example 4.6 — a 2-node rod's local axial DOFs related to global DOFs by a rotation matrix T, with internal forces computed simply in the local system then transformed via f=Tᵀf̂). This is the book's general recipe for corotational elements where expressing stress states in fixed global coordinates is awkward.

### 5. Explicit implementation warnings

- B₀ is not time-invariant despite its subscript — must be recomputed every configuration update.
- Element Jacobian J_x = det(x,ξ) must stay positive; since ρ=ρ₀/J, a non-positive J_x implies unphysical negative density — a real failure mode under excessive mesh distortion.
- Green strain computed via displacement gradient H (not directly from F) to avoid round-off error at small strains.
- Nominal stress P is not symmetric, so Voigt/B₀ formulations must be built from symmetric PK2 stress S (via P=SFᵀ) rather than from P directly.
- Kinematic/kinetic Voigt-rule mismatch (§3 above) is the chapter's central warning about silently corrupting energy calculations.
- Constitutive-law-driven formulation choice: a Cauchy-stress-based law inside a TL code forces extra stress conversions — "advantageous to use the updated Lagrangian formulation" when the constitutive law is σ-based.
- Traction sign/order convention: "n₀ appears before P in the traction expression... if the order is reversed the resulting matrix corresponds to the transpose of P" — an easy convention bug when coding reference-traction BCs.

---

## Chapter 5: Constitutive Models

### 1. Constitutive-routine architecture (the UMAT-style contract)

§5.9 names the object precisely: "the numerical algorithm for integrating the rate constitutive equations is called a constitutive integration algorithm or a stress update algorithm." The implicit box-algorithm contract (§5.9.1, Box 5.13): **inputs** are the converged state at step n (stress, plastic strain, internal variables) plus the strain increment (or F_n, F_{n+1} for large-strain forms) from the global Newton iteration; **outputs** are updated stress/state and, critically, the algorithmic (consistent) tangent dσ_{n+1}/dε_{n+1} — not the continuum tangent.

The book stresses (p.297): "the variables are updated from the converged values at the end of the previous time-step. This avoids nonphysical effects such as spurious unloading which can occur when path-dependent plasticity equations are driven by nonconverged values of the plastic strain and internal variables" — a material routine must always integrate from the last *converged* step state, never from an intermediate unconverged global iterate.

Governing-equation set every routine must satisfy (Box 5.5/5.6/5.10/5.11): additive strain-rate split D=Dᵉ+Dᵖ, hypoelastic stress rate σ∇J = C^J_el:Dᵉ, flow rule Dᵖ=λr(σ,q) and hardening q̇=λh(σ,q), yield function f(σ,q)=0, Kuhn-Tucker conditions λ≥0, f≤0, λf=0.

### 2. Return-mapping / radial-return algorithms

**Elastic-predictor/plastic-corrector structure** (Eq. 5.9.6): σ_{n+1} = C:(ε_{n+1}−εᵖ_n) [trial stress, elastic predictor] − Δλ_{n+1}C:r_{n+1} [plastic corrector]. "The plastic corrector returns or projects the trial stress onto the suitably updated yield surface along a direction specified by the plastic flow direction at the end-point" (closest-point projection, Fig. 5.15).

**Fully implicit backward Euler** (§5.9.2, Box 5.13): flow direction r_{n+1}, hardening moduli h_{n+1}, yield-surface normal all evaluated at the *end* of the step, requiring derivatives r_σ, r_q, h_σ, h_q for the local Newton system — flagged explicitly as "a complicating feature of the method... these expressions may be difficult to obtain for complex constitutive models."

**Semi-implicit backward Euler** (§5.9.6): implicit only in Δλ; flow direction r_n and hardening moduli h_n evaluated at the *start* of the step. Because these are fixed, the resulting matrix involves only elastic moduli, giving a **closed-form** Δλ — no local Newton loop, no flow-rule derivatives needed. Tradeoff: the algorithmic modulus is **generally non-symmetric even for associative plasticity**, since it mixes start-of-step r_n with end-of-step f_σ|_{n+1}.

**Radial return for J2** (§5.9.3, Box 5.14): because the flow direction r = 3s^dev/2σ = f_σ is proportional to deviatoric stress, and "in deviatoric stress space, the von Mises yield surface is circular and therefore the normal to the yield surface is radial," the unit normal stays radial and unchanged throughout the correction — the plastic-strain update is linear in Δλ, collapsing the general tensorial system to a single scalar equation (Eq. 5.9.29): f^(k) = (σ^(0) − 3μΔλ^(k)) − σ_Y(ε) = 0. "The radial return algorithm is usually coded directly along the lines of Box 5.14. For more complex constitutive models, the general scheme in Box 5.13 is used."

### 3. Algorithmic (consistent) tangent vs. continuum tangent

Explicit and emphatic (§5.9.4, p.302): "Due to the abrupt transition to plastic behavior at yield, the continuum elasto-plastic tangent modulus can cause spurious loading and unloading. To avoid this, an algorithmic modulus (also called the consistent tangent modulus) based on a systematic linearization of the constitutive integration algorithm is used instead."

C^alg ≡ dσ_{n+1}/dε_{n+1} (Eq. 5.9.30) — the **exact derivative of the discrete stress-update map**, not the linearization of the continuum rate law. General closed form for the fully implicit scheme (Eq. 5.9.38): C^alg = C̄ − (C̄:r)⊗(f_σ:C̄) / [f_σ:C̄:r + f_q·Y·h] — structurally the continuum expression but with elastic modulus and hardening term corrected by the return-mapping linearization. J2/radial-return specialization (Eq. 5.9.39–5.9.41): C^alg = C^ep − 2μβÎ, β = 1−σ_Y/σ^(0) — the formula a J2 UMAT's Jacobian must implement. Semi-implicit and rate-dependent analogues given in Eq. 5.9.50 and 5.9.57–5.9.65.

**Explicit stability warning** (p.307): "because the stability of deformation processes is closely associated with the symmetry properties of the tangent moduli, care should be taken in the use of algorithmic moduli which do not preserve symmetries... the practice of using the symmetric part of the moduli (on the supposed basis that only the rate of convergence is affected) should only be undertaken if it is clear that stability is not of concern" — a direct warning against a common shortcut.

### 4. Incremental objectivity (Hughes–Winget)

§5.9.10: "an update algorithm is incrementally objective if, in a rigid rotation where F_{n+1}=Q(t)·F_n, the Cauchy stress is given by σ_{n+1}=Q(t)·σ_n·Qᵀ(t)." Rashid (1993) distinction: **weak objectivity** (properly rotates stress under pure rotation) vs. **strong objectivity** (properly rotates stress for combined stretch+rotation motions). Worked example (Eq. 5.9.66–5.9.72): rotate the previous stress by the incremental rotation tensor Q_{n+1}=exp[WΔt] built from an "effective spin" W, form the objective trial stress, then run ordinary small-strain radial return — the standard large-strain hypoelastic-plastic UMAT pattern.

§5.7 warns: hypoelastic formulations "must be integrated in time to compute the stress... require incrementally objective stress update schemes... to ensure that finite rotations do not induce unacceptably large erroneous stresses" — failing this is a concrete source of spurious stress under pure rotation.

### 5. Multiplicative decomposition F = Fe·Fp

§5.7/5.7.1 motivates this as the fix for four listed drawbacks of hypoelastic-plastic (additive, rate-form) models: (1) elastic work in a closed cycle isn't exactly zero (energy leak); (2) elastic moduli must be isotropic; (3) yield function must be an isotropic function of stress; (4) the hypoelastic rate relation must be time-integrated with incrementally-objective schemes.

**Kinematics:** F=Fe·Fp introduces a genuine third "intermediate configuration" (Fig. 5.13) — noted carefully as "a misnomer in that this configuration does not exist as a continuous map," existing only pointwise as a pull-back/push-forward device.

**Rotation invariance proof** (§5.7.7, §5.10.8): imposing a superposed rotation Q(t) on only the current configuration (reference and intermediate fixed), F*=Q·F, and since the intermediate configuration is unaffected, Fp*=Fp, so Fe*=Q·Fe. Then C̄ᵉ*=Feᵀ*·g*·Fe* with g*=Q·g·Qᵀ=g (isometry), giving C̄ᵉ*=C̄ᵉ — **the elastic Green strain on the intermediate configuration is exactly invariant under rotation of the current configuration**, hence so is S̄=∂ψ̄(C̄ᵉ)/∂C̄ᵉ. "Defining the elastic response in terms of a potential ensures objectivity. Furthermore, C^S_el is also unaffected by rotations, so the elastic moduli may be anisotropic."

**Practical payoff** (stated twice, p.291 and p.326): "The stress on the intermediate configuration is thus completely independent of rotations. This eliminates the need for incrementally objective integration schemes which are required in the hypoelastic-plastic formulations." Since the yield function is stated in terms of the invariant S̄, "objectivity imposes no restrictions on the functional dependence... and anisotropic plastic yield behavior can be incorporated."

### 6. Explicit implementation warnings (collected)

- Forward-Euler ("tangent modulus") stress update is deprecated: it "does not satisfy the yield condition at the next step... the solution drifts from the yield surface... because of the inaccuracies of the method it is no longer favored."
- Always integrate from the last converged state, never from a non-converged global-iteration state.
- Continuum tangent in a Newton loop causes spurious loading/unloading at yield transitions.
- Fully implicit scheme needs flow-rule/hardening derivatives that can be hard to obtain for complex models — a real cost tradeoff against semi-implicit.
- Semi-implicit consistent tangent is generically non-symmetric even for associative flow.
- Do not casually symmetrize a non-symmetric algorithmic tangent unless stability is known not to be a concern.
- Symmetry of C^ep requires associative flow in both small-strain and hypoelastic-plastic formulations; non-associative flow generically breaks symmetry of the global tangent stiffness.
- Cauchy-stress-based hypoelastic-plastic formulations give a non-symmetric Truesdell-rate tangent even under associative flow; formulating in terms of Kirchhoff stress restores symmetry — a concrete reason codes carry both Cauchy- and Kirchhoff-stress plasticity formulations.
- Radial return's simplicity depends on J2's circular deviatoric yield surface (isotropic hardening); combined isotropic-kinematic hardening breaks the fixed-flow-direction property and needs more general treatment. (This "more general treatment" is exactly what `j2_radial_return_3d_kinematic()` provides -- see the "Non-associative / kinematic hardening" status-table row below, Wave 2 item 14.)

---

## Chapter 6: Solution Methods and Stability

### 1. Explicit vs. implicit — decision criteria and cost tradeoffs

An explicit method (central difference, §6.2) advances Ma = f_ext − f_int without solving any system, *provided the mass matrix is diagonal (lumped)* — "the salient characteristic of an explicit method." No matrix is factored per step, but stability is conditional: Δt_crit = α·min_e(ℓ_e/c_e) (the CFL condition, eq. 6.2.13), with α=0.8–0.98. Since Δt_crit scales with element size and inversely with stiffness/density, mesh refinement or stiffer material directly shrinks the allowable step. "The cost of an explicit simulation is independent of the frequency range of interest and depends only on the size of the model and the number of time steps."

Implicit methods (Newmark-β, §6.3) solve a nonlinear system every step via Newton — expensive per step, but for linear problems can be unconditionally stable, so step size is set by accuracy/robustness, not a CFL limit. The book cautions no proof of unconditional stability covers the full range of nonlinear practice, but "experience indicates that time steps for implicit integrators can be much larger than those for explicit integration."

**Selection criteria (§6.3.14)**: type of PDE, smoothness of data, response of interest.
- **Parabolic PDEs** (shell bending): implicit strongly preferred in principle (Richtmyer & Morton: parabolic systems "should never be integrated by explicit methods" — stable explicit step decreases by a factor of four every mesh halving) — **but** in practice (car-crash sheet-metal shells) explicit is still preferred despite parabolic character, because contact-impact roughness often prevents implicit Newton from converging robustly.
- **Hyperbolic, structural dynamics/inertial** (input frequency far below mesh resolution): implicit preferred.
- **Hyperbolic, wave propagation** (high-frequency content is the answer of interest): explicit preferred, since implicit gives no step-size benefit when accuracy already forces a small step.

Other concrete notes: explicit is "very robust... seldom aborts due to failure of the numerical algorithm" — a major advantage over Newton, which can simply fail to converge. Elastic-plastic materials cannot raise the explicit step via the slower plastic wavespeed, since unloading (which can occur due to numerical noise at any time) uses the elastic wavespeed. Mass scaling is recommended only when high-frequency response is unimportant. Dynamic relaxation gives poor solutions for path-dependent materials and is slow compared to Newton+preconditioned CG.

### 2. Newton's method for the FE residual

**Unified residual** (§6.3.1): r(d_{n+1}) = s_D·M·a_{n+1} + f_int(d_{n+1}) − f_ext(t_{n+1}) = 0, s_D=0 static, s_D=1 dynamic. Newmark-β expresses acceleration algebraically in terms of d_{n+1}, reducing the dynamic residual to the same nonlinear algebraic form Newton solves for statics.

**Iteration structure** (§6.3.4–6.3.5): linearize r(d_ν)+A·Δd=0, where A=∂r/∂d is the **effective tangent stiffness**: A = s_D·(1/(βΔt²))M + K_int − K_ext. Solve AΔd=−r(d_ν), update, repeat. Full Newton reassembles A every iteration (robust, slow); modified Newton reuses a factorized A across iterations (fast, less robust). Starting iterate: usually d_n, but the Newmark predictor is often better for dynamics.

**Conservative systems** (§6.3.6): the residual is the gradient of a potential W, and A becomes the Hessian K_int−K_ext; stable equilibrium is a local minimum of W — Newton on conservative systems is literally Newton for minimizing a potential.

**Quadratic convergence and when it fails** (§6.3.13): requires the Jacobian to remain a smooth, regular function of d along the whole iteration path — "usually not met in engineering problems": elastic-plastic residuals are not continuously differentiable at yield, Lagrange-multiplier contact residuals lack smoothness. Convergence difficulties are worst near instability, where the Jacobian loses regularity; the mass matrix (always positive-definite) helps dynamic problems but this benefit weakens as Δt grows (∝1/Δt²). For genuinely difficult equilibrium problems, "a straightforward Newton will fail completely" and arc-length enhancements are needed.

**Convergence criteria (§6.3.9):** (1) residual-norm, scaled by force magnitude for problem-independence; (2) displacement-increment norm; (3) energy-error criterion Δdᵀr ≤ ε·max(W_ext,W_int,W_kin) (Belytschko & Schoeberle 1975). Warning: norms must be scaled or the criterion is meaningless across problems; too loose a tolerance gives inaccurate solutions, too tight wastes computation.

### 3. Line search

Addresses the case where Newton's *direction* is good but the *step length* is not — happens when the residual deviates substantially from its linear model or the surface is "rough." Rather than recomputing a new Jacobian (expensive), evaluate the residual at points along the ray d=d_old+ξΔd and minimize a residual measure over ξ. For conservative systems this reduces to requiring the residual be orthogonal to the search direction at the minimum (same condition as the energy convergence measure) — a quadratic-interpolation line search is the practical algorithm suggested.

### 4. Tangent stiffness: material vs. geometric, and the algorithmically consistent tangent

**Derivation split** (§6.4.1): differentiating the TL internal force f_int=∫B0ᵀP dΩ0, the rate Ṗ splits via Ṗ=Ṡ·Fᵀ+S·Ḟᵀ into: **material tangent K_mat** (from stress rate Ṡ, involving tangent moduli C^SE) — structurally identical to the linear-elasticity stiffness matrix; and **geometric ("initial stress") stiffness K_geo** (from S·Ḟᵀ, involving current stress itself: K_geo,IJ = H_IJ·I) — invariant under local coordinate rotation.

**Why geometric stiffness is absent from linear FE:** it arises purely from S·Ḟᵀ — the existing stress state acting through changing geometry. In small-deformation linear FE, Ḟ≈0 to leading order, so this term vanishes; it appears only in geometrically nonlinear or pre-stressed formulations. This is exactly why geometric stiffness produces buckling — it can turn K=K_mat+K_geo indefinite under compression even when K_mat alone is positive-definite.

**Symmetry**: K_mat is symmetric when [C^SE] has major symmetry; K_geo is always symmetric. But this holds only for the specific stress rate chosen (Ṡ/Truesdell rate) — for other rates (e.g. Jaumann), major symmetry of the moduli does *not* guarantee symmetry of K_int, requiring extra correction terms (Example 6.1/6.3).

**Directional (Gateaux) derivatives — a warning about a common mistake** (§6.4.6): for materials like elastic-plastic that are not continuously differentiable, a standard derivative doesn't exist — a worked 2-bar-truss example at compressive yield has **four different tangent stiffness values** depending on which quadrant of incremental-displacement space you're in. Plain continuum-tangent Newton silently breaks here; the fix is directional derivatives, secant methods, or step-size-dependent algorithmic forms.

**Algorithmically consistent tangent stiffness** (§6.4.7) — direct link to Chapter 5: relates the *finite* increments actually produced by the stress-update algorithm, referencing Chapter 5's eq. 5.9.38/5.9.41/5.9.58-59. "The continuum tangent modulus is easy to implement but causes convergence trouble at non-smooth response... the consistent/algorithmic tangent gives better (often quadratic) convergence." Standard Newton is inappropriate when using algorithmic moduli — a **secant Newton method** (Box 6.6) is needed instead, recomputing stress/internal variables from the *last converged step* d_n, not the previous Newton iterate — otherwise non-converged intermediate iterates corrupt path-dependent constitutive state.

### 5. Stability and continuation methods

**Liapunov stability** (§6.5.1): a solution is stable if all perturbed trajectories within an ε-ball stay within a bounded C·ε ball for all future time; one diverging trajectory is enough to call the state unstable.

**Why load-controlled Newton fails at limit points, and what arc-length fixes** (§6.5.2–6.5.3): equilibrium branches show turning/limit points (snap-through), stationary bifurcations, or Hopf bifurcations. At a limit point, pushing load γ further is "fruitless" — no nearby intersection with the branch. **The arc-length (Riks) method** replaces load control with a combined-arc-length constraint (eq. 6.5.6): α(Δd)ᵀ(Δd) − (γ_{n+1}−γ_n)² − Δs² = 0 — geometrically a sphere about the last converged point. Since arc length (not load) must increase, branches with decreasing load past a limit point are traced naturally. The resulting system is generally non-symmetric and has two solutions per step; the correct one maximizes Δdᵀ(d_n−d_{n-1}) (same direction as the previous step).

**Detecting critical points** (§6.5.4–6.5.9): linearizing Md̈+Ad=0 and assuming exponential growth gives the eigenproblem λ_iy_i=AM⁻¹y_i; a critical point requires det(A)=0. For symmetric A, stability reduces to positive-definiteness. **Determinant-sign-test caveat**: usually but not always changes sign at a critical point — if two eigenvalues cross zero simultaneously at a bifurcation, the determinant doesn't change sign, so the test alone is not conclusive. A practical eigen-estimate interpolates A between the last two converged states and solves a generalized eigenproblem in inverted form (for numerical robustness). A one-shot linear-buckling estimate works well for bifurcation points but is much less reliable for limit points, since geometric stiffness typically changes substantially before a limit point.

### 6. Material stability, change of PDE type, and regularization

**Material instability** (§6.7.1–6.7.2), distinct from structural/numerical instability: perturbs an infinite homogeneous slab with a plane harmonic wave and asks whether the perturbation grows — a constitutive-level property independent of mesh/geometry/BCs. Hadamard (1903) via vanishing acoustic-wave speed for negative tangent modulus; Hill (1962) formalized the general criterion; Rudnicki & Rice (1975) showed instability/localization can occur even under strain *hardening* if plasticity is non-associative — loss of major symmetry alone can trigger instability, not just softening. The eigenproblem is on the **acoustic tensor** Â(n₀), built from the same "first elasticity tensor" that appears in the §6.4 tangent-stiffness linearization. Positive-semidefiniteness of Â for all n₀ is the **strong ellipticity condition** — satisfying it means the equilibrium PDE is elliptic.

**Connection to change of PDE type** (§6.7.3): in 1D, the perturbed wave equation is ρü=(E^T_s+σ₁₁)u_xx. When E^T_s+σ₁₁>0 the material is stable and the PDE is **hyperbolic**; when <0 the material is unstable and the PDE becomes **elliptic** — literally "loss of hyperbolicity." "Material instability is associated with a change in type of the PDE." The critical condition is shown equivalent to the classical **Considère criterion** for necking onset.

**Regularization** (§6.7.4): Bažant & Belytschko (1985) proved that once a rate-independent softening material goes unstable, strain localizes to a set of measure zero, so dissipated energy over that set is zero — the raw model cannot represent real fracture. This is the mesh-dependence pathology: as the mesh refines, the localization band shrinks and computed dissipated energy goes to zero — refinement makes it *worse*, an explicit warning against the instinct to "just refine the mesh." Four regularization strategies: gradient, integral/nonlocal, Cosserat (coupled-stress), rate-dependence (viscoplastic — the only one characterized as mature/robust in practice, though it has no intrinsic length scale on its own and can need coupling to the heat equation for one). A separate practical fix for fracture: Hillerborg et al. (1976) makes the terminal softening strain depend on element size so dissipated energy per element matches a prescribed fracture energy regardless of mesh — "strange but intriguing... experience shows it works very well."

### 7. Additional explicit warnings

- **"Arrested instability"** (§6.2.3): a purely elastic geometric instability can trigger exponential growth that drives the material into plasticity, softening/slowing the wavespeed and regaining formal numerical stability — the run doesn't blow up but displacements are grossly overpredicted, and this is **not detectable by inspecting results**. The only reliable check is energy-balance monitoring, recommended as a standard diagnostic in every nonlinear explicit run.
- A perfectly symmetric beam **will not buckle numerically** even past the theoretical buckling load, in either explicit or implicit integration — an imperfection is required to break symmetry ("if you do not believe this, try it").
- Asymmetric bifurcating structures are highly imperfection-sensitive; a single "perfect" simulation can badly overestimate real buckling strength.
- For non-smooth constitutive response, plain continuum-tangent Newton can hit genuinely undefined derivatives — directional/Gateaux derivatives or secant iteration from the last converged state are necessary.

---

## Chapter 7: Arbitrary Lagrangian-Eulerian (ALE) Formulations

### 1. What problem ALE solves, and the core idea

Pure Lagrangian meshes fail under severe deformation: shape-function accuracy degrades, quadrature-point Jacobians can go negative, the Newton system becomes ill-conditioned, and the stable explicit time step shrinks. Remeshing works around this but is "burdensome and introduces errors due to projections." Pure Eulerian meshes avoid distortion entirely but make constitutive updates and moving-boundary/free-surface tracking difficult.

The core idea (§7.2): introduce a third, independent referential (ALE) domain with its own coordinates χ, distinct from material X and spatial x. The mesh moves via φ̂(χ,t), chosen arbitrarily by the analyst, while material still moves via x=φ(X,t). Lagrangian (χ=X) and Eulerian (χ=x) fall out as special cases. The **convective velocity** c≡v−v̂ (material minus mesh velocity) vanishes for Lagrangian meshes, equals v for Eulerian meshes — the user's choice of mesh motion directly sets how much convection the numerics must handle.

### 2. Governing ALE conservation equations and the convective term

The material-derivative identity Df/Dt = ∂f/∂t|χ + c·∇f (Eq. 7.2.18) is substituted into the standard Eulerian conservation laws to get ALE forms — "the only difference... is in the material time derivative terms." Continuity must now be enforced as a PDE rather than the algebraic Jacobian relation used in Lagrangian formulations — "One major difference which arises in ALE descriptions." The convective term appears in continuity, momentum, and energy because the mesh observer moves relative to the material at rate c. The discrete mass matrix is not constant in time, and for Petrov-Galerkin discretization it is not symmetric — both flagged as consequences of the convective term.

### 3. Mesh update / mesh smoothing strategies

The book frames this as "one of the major hurdles in developing an effective implementation of the ALE description" — the mesh should avoid distortion while keeping boundaries/interfaces at least partially Lagrangian. Strategies presented (§7.10): mesh motion prescribed a priori when boundaries are known; the Lagrange-Euler matrix method (per-node mixing parameter, useful for free surfaces but hard to control interior mesh shape); the deformation-gradient/mixed method (enforces w·n=0 on material boundaries while smoothing the interior independently); automatic generation via Laplace/biharmonic equations (a 4th-order equation gives better mesh shape than Laplace's, which distorts near high-curvature boundaries); and the modified elasticity equation (mesh as fictitious elastic solid, with the material constant divided by local Jacobian so smaller/more distortion-prone elements are made stiffer) — this last one is used in the worked cylinder-in-channel example and "works for any mesh type and for any type of movement" at the cost of an extra linear solve every time the mesh deforms.

The tradeoff the book returns to: mesh velocity tracking material closely (small c) behaves Lagrangian — good for interfaces, bad for large distortion; smoothing toward regularity (large c) avoids tangling but pushes more work onto the convective terms and loses tight interface tracking unless boundary nodes stay Lagrangian while interior nodes smooth.

### 4. Numerical treatment of the convective term

Motivated via the linearized 1D advection-diffusion model problem: standard Galerkin reduces to central differencing, whose discrete solution is proven oscillatory (unstable) whenever the mesh Peclet number Pe>1 — labeled **spatial instability**, structurally analogous to but distinct from Chapter 6's temporal instabilities.

Fix: **Petrov-Galerkin/SUPG** (Brooks & Hughes 1982) — a different test function with an extra piece proportional to the derivative in the flow direction, equivalent to adding artificial viscosity υ*=υ+ᾱuΔx/2 with the optimal upwind parameter α=coth(Pe)−1/Pe in closed form (reproduces the exact nodal solution in 1D). The multi-dimensional SUPG generalization adds a stabilization term acting only along the streamline direction ("streamline upwind"). Applying this to the ALE momentum equation is nontrivial because the Petrov-Galerkin test function needs C⁻¹ continuity while the Galerkin term needs stress at least C⁰ — "a major drawback... requires the trial function to be C¹" — so the book integrates by parts a second time to eliminate the stress-gradient term entirely, replacing it with a boundary reaction-force integral, avoiding evaluation of the often "complicated and computationally time-consuming" stress divergence.

### 5. Path-dependent material state: the stress "remap"/advection step

Because ALE quadrature points don't coincide with material points, "the stress history needs to be convected by the relative velocity c." The Jaumann-rate constitutive law is rewritten in ALE form by adding the same convective term, handled with the same SUPG/upwind machinery. Three stress-update strategies given: standard Galerkin convective update (same spatial-oscillation risk); SUPG update (streamline-upwind artificial viscosity); and **operator splitting** (Liu/Belytschko/Chang 1986) — a Lagrangian phase (ordinary updated-Lagrangian stress update at Gauss points, ignoring convection) followed by rezoning and an Eulerian/convection phase that remaps state variables via a Lax-Wendroff explicit scheme.

**Explicit oscillation/diffusion warnings**: comparing exact-integration (central-difference-like) vs. full-upwind (donor-cell) transport matrices for stress, the book shows exact integration decouples odd/even element transport — "physically unrealistic oscillations would be expected" — while full upwind smears/diffuses but stays stable. Confirmed in the elastic-plastic wave-propagation example: "the scheme without the upwinding technique causes severe unrealistic spatial oscillations... The new method proposed here eliminates these oscillations completely" — and "transport of stresses as well as yield stress... plays an important role in ALE computations for path-dependent materials."

---

## Chapter 10: Contact-Impact

### 1. Contact constraint formulation: gap condition and Kuhn–Tucker/complementarity conditions

Restricted to Lagrangian meshes, with master body A and slave body B, contact interface Γc=ΓA∩ΓB. Impenetrability starts as a set condition (Ω^A∩Ω^B=∅) but cannot be expressed algebraically since it's impossible to anticipate which points will contact — recast in **rate form** on the contact surface: γ_N ≡ v_N^A+v_N^B = −(v^A−v^B)·n^A ≤ 0. A geometric closest-point-projection measure g_N is also derived for implicit/equilibrium solutions, noted to be **non-unique at surface kinks/non-convex regions** — a stated pitfall for contact search on non-smooth surfaces.

Kinetic conditions: tractions balance (t^A+t^B=0), normal traction cannot be tensile (t_N^B≤0, no adhesion). Combined into the **Kuhn-Tucker contact condition**: γ_N t_N = 0, with γ_N≤0 and t_N≤0 — either bodies stay in contact (γ_N=0) with compressive traction, or separate (γ_N<0) with vanishing traction; normal contact forces do no work.

### 2. Lagrange multiplier vs. penalty vs. augmented Lagrangian vs. perturbed Lagrangian

- **Lagrange multiplier**: enforces impenetrability essentially exactly; discrete system is indefinite, unbanded, with *more* unknowns, and requires constructing a **separate mesh for the multiplier field** — "not viable" in 3D without an independently constructed, sufficiently fine multiplier mesh. Explicit verdict: "for high velocity impact, Lagrange multipliers often result in very noisy solutions. Therefore, Lagrange multiplier methods are most suited for static and low velocity problems."
- **Penalty**: t_N=−βγ_N or the recommended p(g_N,ġ_N)=β₁g_N+β₂ġ_N form (pure rate-dependent penalty "may allow excessive interpenetration"). No extra unknowns, positive-definite, no extra mesh — but only **approximately** satisfies impenetrability. "If the penalty parameters are too small, excessive interpenetration occurs... In impact problems, small penalty parameters reduce the maximum computed stresses. Picking the correct penalty parameter is a challenge." Always decreases the stable explicit time step; if too large, causes spurious velocity reversal within a single time step, fixed with an explicit upper bound on the penalty force.
- **Perturbed Lagrangian**: algebraically shown to reduce to the penalty method — "a penalty weak form in disguise."
- **Augmented Lagrangian**: literally the sum of the Lagrange-multiplier and penalty contributions — "a synthesis of penalty and Lagrange multiplier methods," recovering exact impenetrability while retaining better conditioning than pure multiplier — the book's implicit rationale for generally preferring it.

**Regularization framing**: the book frames penalty as a regularization of the discontinuous velocity/traction response at impact, analogous to artificial viscosity for shock capturing — "it smoothes the discontinuous velocities... and preserves momentum conservation. It only relaxes one condition, the impenetrability condition... That is a small price to pay for smoother solutions." Penalty parameters "are generally not based on mechanics, but instead are chosen so that they eliminate or reduce frequencies above a certain threshold" — a numerical filter, not a physical stiffness.

### 3. Friction models

**Coulomb friction**: stick (|t_T|<−μ_F t_N ⟹ γ_T=0) / slip (|t_T|=−μ_F t_N ⟹ t_T opposes relative slip velocity) pair. Explicit plasticity analogy: "if the tangential velocity γ_T is interpreted as a strain and the tangential tractions are interpreted as stresses, the first relation... can be interpreted as a yield function... These attributes parallel a rigid plastic material model." The stick condition is flagged as "the most troublesome characteristic, since it induces discontinuities... which makes numerical procedures difficult."

**Interface constitutive equations** (Michalowski/Mroz, Curnier): recast friction as a full nonassociative-plasticity theory on the interface, splitting tangential velocity into elastic "adherence" and irreversible "slip" parts, with a yield function (Coulomb cone) and a *separate* nonassociative potential for the flow law. Nonassociative flow is required because an associated flow law would predict the bodies separate after sliding onset (unphysical) — the nonassociated potential correctly gives zero normal slip during sliding.

### 4. Contact detection / search

The geometric closest-point-projection measure is the mathematical basis for node-to-segment contact checking. The Lagrange-multiplier mesh discussion addresses the noncontiguous-node problem: placing multiplier nodes at master-body nodes fails when the other mesh is finer ("will then lead to interpenetration"); placing them wherever a node occurs in either body risks very small, ill-conditioned elements. For general 3D use, a wholly independent, sufficiently fine multiplier mesh is required. On cost: "in a large model, on the order of 10⁵ nodes may need to be checked against penetration into a similar number of elements... a brute force approach to this task is not going to work" — the extracted text is truncated at exactly this point, so no further detail on bucket-sorting/spatial-hashing search algorithms could be recovered from this pass.

### 5. Explicit pitfall warnings

- Explicit integration suits contact-impact because small stable time steps mean "the discontinuities due to contact-impact wreak less havoc," with no Newton solver whose convergence is impeded by the discontinuous Jacobian that contact introduces.
- Bodies are updated independently each step (ignoring contact) then corrected — this predictor/correct structure is shown to reduce, for the corrected velocities, to the classical plastic-impact momentum-conservation formula.
- **Release cannot occur in the same step as impact** — a rarefaction wave needs two traversals to the nearest free surface, which can't happen within one stable time step (does not apply to structural elements that don't resolve through-thickness stiffness).
- Explicit contact always dissipates energy at impact, decreasing with mesh refinement.
- Contact modifications can violate symmetry BCs at contacting nodes on a symmetry plane, since contact corrections are applied after BCs are enforced — "boundary conditions sometimes have to be reimposed at contact nodes after the [contact] modifications."
- Penalty stiffness sensitivity: too-large β1 causes non-physical velocity reversal within a step (fixable via an upper bound) and always reduces the critical stable time step; the Lagrange-multiplier approach doesn't have this stable-timestep penalty but is noisier at high velocity.

---

## Chapter 8: Element Technology

### 1. Patch tests

The patch test (Irons/Bazeley et al. 1965; Strang 1972) fundamentally tests **polynomial completeness/reproducing conditions**, not just a pass/fail exercise. A patch of distorted elements is subjected to a prescribed linear boundary displacement field; since the exact solution is a linear field with constant strain/stress, if the element space contains that field, the FE solution must reproduce it to ~5 significant digits. Failure means either the element is not linearly complete (a formulation defect) or there's an implementation bug — the book stresses this dual diagnostic role. Tied to the FE analog of the Lax equivalence theorem: completeness + stability → convergence — since no general FE convergence proof exists, the patch test is the practical surrogate.

Extensions: the **nonlinear patch test** (large-coefficient linear field, tests nonlinear kinematics/stress-update/Newton wiring specifically, since passing the linear test already guarantees the reproducing conditions); the **explicit-program patch test** (Belytschko, Wong & Chiang 1992 — prescribes a linear velocity field, integrates one step with no external forces, checks constant rate-of-deformation and near-zero interior nodal accelerations, validating the entire force-assembly/mass-matrix pipeline); the **stability patch test** (Taylor et al. 1986 — minimal BCs to remove rigid body motion, checking traction BCs and partial spatial stability, but explicitly *not infallible*: noncommunicable spurious modes can slip through, and a full check needs an eigenvalue analysis expecting exactly n_RB zero eigenvalues).

### 2. Locking: volumetric and shear

**Volumetric locking** (§8.4.3): for Q4, an isochoric motion requires zero dilatation rate everywhere. Using the basis-form velocity decomposition, the dilatation rate has a nonconstant part driven by the hourglass mode that cannot vanish everywhere except along a single line — so any admissible volume-preserving deformation forces neighboring elements toward zero motion, propagating lock through the mesh. **Root cause: mismatched interpolation — the element's velocity space cannot represent an isochoric field exactly**, i.e. more independent dilatational constraints at quadrature points than kinematic DOFs to satisfy them (constraint counting). As ν→0.5, the bulk modulus blows up and locks onto whatever nonzero dilatation the discretization can't eliminate.

**Shear locking** (§8.6.2): in a beam-in-pure-bending example, equilibrium requires zero shear resultant, but every element deforms partly via a bending-mode "x-hourglass," giving nonzero parasitic shear strain that absorbs energy that should be pure bending energy. Distinguishing feature: shear-locked elements *do* converge, just slowly (excessive stiffness, not complete failure) — the book prefers the term "excessive shear stiffness" for this reason.

### 3. Underintegration, hourglassing, and control strategies

**Why reduced integration cures locking**: one-point quadrature (evaluated only at element center) simply doesn't sample the nonconstant (hourglass-driven) strain that causes both locking types — the quadrature point sits exactly where the parasitic term vanishes.

**Why this creates spurious modes** (§8.3.9): rank(K_e) ≤ rank(B̄); Q4 full quadrature gives rank 5 (proper rank = 8 DOF − 3 rigid body); one-point quadrature gives only rank 3 — a **rank deficiency of 2**, corresponding exactly to x- and y-hourglass modes: nodal patterns with B̄(0)d=0 (zero energy at the sampled point despite a non-rigid-body deformed shape — "hourglass"/"keystoning"). Because they're communicable, they can propagate mesh-wide with zero energy and grow linearly in time as a weak instability.

**Control strategies** (§8.7.3–8.7.6): **perturbation (stiffness-type) control** (Flanagan-Belytschko 1981) — augments B̄ with extra rows orthogonal to all linear fields, preserving linear completeness while restoring rank, scaled per Belytschko-Bindeman (1991) via a Rayleigh-quotient bound to a fraction (≈0.1) of the true element stiffness eigenvalue; **viscosity-based control** — combines stiffness and viscous terms, recommended for large impulsive/shock loads (viscous) vs. moderate long-duration simulations (stiffness, energy recoverable); **physical (parameter-free) control** (Belytschko & Bindeman 1993) — derived from assumed-strain fields with constant spin/homogeneous response assumptions integrated in closed form, no arbitrary scaling parameter, and crucially **will not lock even for incompressible materials**.

**Tradeoff, explicitly stated**: too little control leaves hourglass modes essentially uncontrolled under "rich" loading (point loads, poorly-programmed contact producing sign-alternating nodal forces — the book notes this happened "in one widely used program"). Too much control (large parameter for nearly-incompressible material) reintroduces locking — "you will cure hourglassing by locking the mesh!" Recommended: monitor hourglass-energy/total-strain-energy ratio (should stay well under 3-5%; can be 90%+ on coarse meshes, needing mesh refinement not just parameter tuning).

### 4. Selective-reduced integration, B-bar, and assumed-strain; the Hu-Washizu basis

**SRI**: one-point quadrature on volumetric terms, full quadrature on deviatoric terms — cures volumetric locking while retaining full rank on the deviatoric part.

**B-bar (Simo-Hughes)**: assumes the stress field σ̄ is orthogonal to the difference between assumed strain D̄ and actual D(v) — makes the second Hu-Washizu term vanish identically, collapsing the three-field form to a single-field-looking form where B is simply replaced by an assumed-strain matrix B̄. "Striking in its simplicity": σ̄ never needs to be constructed, since the space of continuous functions always satisfies orthogonality.

**Hu-Washizu basis** (§8.5.2): the general three-field weak form (independent v, D̄, σ̄) is shown equivalent to momentum balance + traction BC + constitutive equation + strain-displacement relation — σ̄ acts as a Lagrange multiplier enforcing D̄=sym∇v. Preferred over the two-field Hellinger-Reissner form because HR is incompatible with strain-driven (rate-form) constitutive laws typical in nonlinear analysis.

**Limitation principle** (Stolarski & Belytschko 1987, generalizing Fraeijs de Veubeke 1965): adding terms to a strain field beyond the gradient of an already-complete velocity interpolation has *no effect* for linear constitutive laws — mixed methods can only *suppress* deleterious strain terms (curing locking), never *enhance* accuracy beyond what removing them already buys. This directly rebuts literature claims that mixed elements are inherently superior.

**Stability caveat**: mixed/multi-field and SRI elements can suffer pressure-field checkerboarding because the pressure Lagrange-multiplier field is under-constrained — governed by the LBB condition, with a practical necessary condition n_dof≥n_p (Zienkiewicz-Taylor test). The constant-pressure Q4 fails this; the 9-node quadrilateral with linear pressure passes.

### 5. Specific element formulations

Belytschko-Bachrach OI element, QBI element, ADS element — all built from the general recipe K_e = K^1pt_e + K^stab_e (one-point quadrature stiffness plus a rank-2 stabilization stiffness built from tabulated constants). Pian-Sumihara assumed-stress element uses covariant stress components; requires a single consistent curvilinear orientation to pass the patch test, and curvilinear-frame invariance doesn't actually improve convergence rate, so the extra transformation cost is often not worth it. Multi-point integration with assumed strain is recommended for elastic-plastic beams (2-10 points through-thickness) since one-point quadrature under-resolves the non-smooth (yielding) stress distribution even with hourglass control fixing the rank issue — hourglass control and integration accuracy are separate concerns. 3D hexahedral extension exists but the book candidly notes "a theory as to the best structure has not yet been developed."

---

## Chapter 9: Beams and Shells

### 1. Continuum-based (CB) approach vs. classical structural theory

The book adopts the continuum-based methodology almost exclusively over classical shell weak forms, since the latter is "difficult, particularly for nonlinear shells" (curvilinear tensors, disagreement on the "right" nonlinear shell equations, awkward thickness/junction/stiffener handling). CB kinematic constraints (fiber/director assumptions) are imposed either directly on an existing continuum element's discrete equations or on the motion in the weak form before discretization — both give identical discrete equations.

**Beam kinematics**: slave nodes on top/bottom surfaces of an underlying continuum element are constrained to linear-through-thickness motion via a **director** p(x,t) (the "modified Mindlin-Reissner assumption"): fibers remain straight, transverse normal stress is zero (plane stress), fibers are kinematically inextensible — reconciled with plane stress by *not* using kinematics to compute the through-thickness strain but instead deriving it from the constitutive plane-stress constraint. All element matrices are obtained from the underlying continuum element via a transformation matrix T — "these matrices do not need to be rederived for CB beams," extended identically to shells with 5- or 6-DOF nodes.

Contrast with classical theory: Euler-Bernoulli/Kirchhoff-Love (no shear, C¹ continuity) vs. Timoshenko/Mindlin-Reissner (admits shear, only C⁰ continuity needed) — C¹ interpolants are "the biggest disadvantage" of classical theory and "difficult to construct in multi-dimensions," a major reason CB/C⁰ dominates practice.

### 2. Shear and membrane locking: why worse than solids, and cures

Framed directly analogous to volumetric locking (constraint → inability to satisfy exactly → locking), but the penalty **scales with thickness**: shear-locking penalty stiffness grows as (ℓ/h)² relative to bending stiffness. As shells/beams get thinner — exactly the regime they're meant for — the ratio blows up, so locking gets worse precisely where structural elements are supposed to work, unlike volumetric locking's dependence on a material property (ν→0.5).

**Shear locking mechanism**: for a 2-node beam in pure bending, the interpolated transverse shear strain is nonzero everywhere except x=0, even though equilibrium requires zero shear under constant moment. **Membrane locking mechanism**: in an inextensional bending mode, membrane strain should vanish identically but the FE interpolation gives a nonzero field except at the Gauss points — root cause is mismatched interpolation order between the membrane displacement's derivative and the transverse displacement, again a constraint-counting argument. Membrane locking vanishes for flat/straight elements and only shows up for warped/curved geometry.

**Cures**: selective reduced integration on shear/membrane terms only; **Barlow points** (1976) — the reduced-integration Gauss points are exactly where both parasitic shear and parasitic membrane strain vanish for cubic-order fields, giving a principled rather than ad hoc quadrature-point choice; **Assumed Natural Strain (ANS)-type interpolation** (MacNeal 1982, Hughes & Tezduyar 1981, Dvorkin & Bathe 1984) — interpolates assumed shear strain from values sampled at edge midpoints, where shear vanishes exactly for a rectangle under constant moment and remarkably extends to arbitrary quadrilaterals. Even the "successful" SRI 4-node plate element (Hughes 1987) retains a residual spurious singular mode, the **w-hourglass mode** — shell locking cures are harder than for continua.

### 3. Large-rotation kinematics for shells

**Euler's theorem** underlies the Rodrigues rotation formula, with the explicit warning that the rotation pseudovector does **not** behave like a true vector: rotation composition is noncommutative — Euler angles are avoided (nonunique, awkward equations of motion) in favor of the exponential-map/rotation-tensor formalism. First-order updates are "too inaccurate for most purposes"; naive second-order updates don't preserve orthonormality. The **Hughes-Winget update** (a Cayley-transform-style midpoint rule) is second-order accurate and *exactly* orthogonal — the practically favored update, and the same fix as Chapter 5's incremental-objectivity mechanism applied to shell directors. Quaternions are given as an alternative singularity-free representation.

**Avoiding singularities / the "drilling" DOF problem**: the angular velocity component parallel to the director has no physical effect on the director (the "drilling" component). Since it's kinematically irrelevant, "a five-degree-of-freedom formulation is more consistent with CB shells than a six-degree-of-freedom formulation" — and explicitly: **"when the shell is flat, the stiffness is singular for a six-degree-of-freedom formulation"** — naively using 3 rotational DOFs per node produces a zero-stiffness drilling mode whenever the shell is locally flat, a classic implementation pitfall. Practical recommendation: use 6 DOF only at nodes where the extra DOF is actually needed (corners, stiffeners, kinks).

### 4. Explicit implementation warnings

- Flat-shell 6-DOF singularity (above) — the single most concrete "gotcha" flagged in the chapter.
- Fiber/director misalignment with the true normal: "if the directors are not normal to the midsurface, the motion deviates markedly from the motion which is observed experimentally" — and it's noted to be *impossible* to align fibers exactly with the normal along the whole length of a C⁰ element.
- Fiber inextensibility is kinematics-only, not stress-consistent — the through-thickness strain must be back-solved from the plane-stress constitutive constraint, not computed from the kinematic velocity field directly; getting this backwards is an easy bug.
- "Neither full quadrature nor the selective-reduced quadrature [in the usual SRI sense] can be used in a CB beam. Both quadrature schemes result in shear locking" — only a single quadrature stack at the beam's center avoids it.
- Insufficient through-thickness integration points for elastic-plastic beams/shells dramatically delays the predicted onset of plasticity in bending; trapezoidal rule is recommended over Gauss quadrature since the elastic-plastic through-thickness stress isn't smooth.
- The CB beam/shell mass matrix M=TᵀM̂T is genuinely time-dependent ("unusual for a Lagrangian element") — if Ṫ is neglected, inertial forces stay linear in velocity; if included, centrifugal-type nonlinear terms appear — an easy source of subtle energy/momentum errors if this coupling is silently dropped or inconsistently included.
- Corotational axis alignment for reduced-integration shells is a subtle coordinate-transformation pitfall — the original Belytschko-Tsay-type alignment "can lead to difficulties."
- One-point-quadrature shells suffer rank deficiency of 4 (3 communicable hourglass modes plus a noncommunicable in-plane twist mode); generalized hourglass strain rates are *not* orthogonal to rigid-body rotation for warped elements, requiring explicit projection to avoid hourglass control itself injecting spurious energy under rigid rotation.
- **"Correctness" vs. robustness tradeoff**: the book's own comparison table shows fully-integrated/more-rigorous elements (pass patch test, correct in twist) are more accurate but "fail suddenly and dramatically, aborting the simulation" under large distortion, while the cheaper Belytschko-Tsay-type element "is very robust under severe distortion and seldom aborts a computation" — a pointed real-world warning that formulation correctness and practical robustness under extreme mesh distortion are not the same thing, and crash-simulation codes systematically trade the former for the latter.

---

## Chapter 11: Extended Finite Element Method (XFEM)

### 1. Core problem XFEM solves

Standard FEM requires the mesh to conform to any discontinuity (crack faces, tips, material interfaces). The book lists four disadvantages of remeshing: demanding mesh generation, computational cost, error from state-variable projection between old/new meshes, and complicated post-processing from an ever-changing mesh. Alternatives (element deletion, inter-element cohesive methods) remain mesh-dependent, since the crack path must still follow element edges.

XFEM's fix decouples the mesh from the discontinuity's geometry via the **partition of unity** property (Melenk & Babuška 1996): since ΣN_I(X)=1 everywhere, any function Ψ(X) can be reproduced locally by multiplying against the PU. The FE approximation is enriched only at a small local subset of nodes, keeping the system sparse.

### 2. Enrichment functions and the enriched approximation

u^h(X) = ΣN_I(X)u_I + ΣN_I(X)Ψ(X)q_I (Eq. 11.2.3) — standard FE part plus PU enrichment part. **Strong discontinuity (crack body)**: Ψ=Heaviside H(f(X)), f a signed-distance/level-set function. **Crack-tip enrichment** for LEFM: four √r-based near-tip asymptotic functions (Eq. 11.4.4); for elastic-plastic fracture, a 6-term set scaled as r^(1/(N+1)) fitted to HRR fields (Elguedj et al. 2006). **Weak discontinuity** (material interfaces): C0-continuous but kinked-derivative enrichment via absolute-value-of-level-set forms (Sukumar 2001; Moës 2003). **Shifting** — subtracting the enrichment function's nodal value — forces the enrichment term to vanish at nodes, preserving the kinematic meaning of nodal DOFs and simplifying blending with unenriched elements (though it does not strictly guarantee pointwise satisfaction of displacement BCs).

### 3. Level-set method

The interface is represented implicitly as the zero level set of a signed-distance function φ(x,t), interpolated with the same FE shape functions at nodal geometric DOFs, evolving via a hyperbolic transport equation ∂φ/∂t+v·∇φ=0. For crack growth, **two** level sets are used — one tracking the crack surface, one tracking the tip/front — because a crack is irreversible (cannot "heal" behind the tip), unlike a generic moving interface. Nodes are classified as tip-enriched or step-enriched directly from the sign pattern/zero-crossing of the discretized φ over each element.

### 4. Blending elements

At the boundary between enriched and standard elements, "blending elements" (partially enriched) arise, causing difficulties because the partition-of-unity property that makes enrichment reproduce Ψ(X) exactly breaks down — extra unwanted polynomial terms appear that degrade local accuracy. The minimal fix mentioned: include the enriched field in blending elements too, but force enrichment DOFs to zero at unenriched nodes; shifting simplifies this by making the enrichment term vanish at unenriched element boundaries.

### 5. Integration of discontinuous/singular integrands

Standard Gauss quadrature fails for cut/singular elements since a discontinuity or singularity can't be represented by finite-order polynomials. **For jump enrichments**: sub-triangulation — decompose the cut element into conforming integration sub-elements aligned with the interface, then apply ordinary Gauss quadrature on each; adds no DOFs, doesn't affect the critical time step, but introduces a data-projection error. **For singular (crack-tip) enrichments**: "almost polar" integration (Laborde et al. 2005; Béchet et al. 2005) — split the tip element into triangles with one vertex at the singularity, map tensor-product Gauss points so two corners collapse onto the singular node, removing the singular term from the quadrature; an alternative converts the domain integral to a much cheaper contour integral.

### 6. Phantom node method

Song et al. (2006) reformulates the Heaviside-enriched element as **two overlapping standard elements**, each fully active only on one side of the crack, connected via a cohesive traction-separation law — nodes on the "wrong" side for a given sub-element are phantom nodes (DOFs exist mathematically but the shape-function contribution is switched off). The book lists three implementation burdens of "true" XFEM: special quadrature for discontinuous/singular integrands, coding the enrichment functions, and supporting a **variable number of DOFs per node** — disruptive to retrofit into an existing code and awkward for parallelization. Phantom nodes sidestep the DOF-count problem by adding extra elements/nodes with a fixed, standard DOF count per node — much easier to graft onto existing explicit/low-order-element codes — at the cost of only supporting Heaviside (strong discontinuity) enrichment, not near-tip singular enrichment.

---

## Chapter 12: Multiresolution Continuum Theory (MCT)

### 1. Core motivation

Classical (Cauchy) continuum mechanics treats material behavior as purely local and phenomenological, with no embedded microstructural length scale — but real materials (metals, alloys, composites) are heterogeneous across nested scales (grains, phases, precipitates, lattice), and it's exactly this microstructure that controls macroscopic strength, toughness, ductility. Direct numerical simulation of the full microstructure is normally computationally prohibitive; generalized continuum theories inject microstructural/length-scale information into the macroscopic balance laws without meshing every grain.

### 2. Kinematic enrichment and the link to localization/mesh-dependence

Information about a material point's neighborhood is added either as higher gradients of existing DOFs ("higher-order" theories) or as genuinely new DOFs ("higher-grade" theories, e.g. micromorphic/Cosserat). MCT extends the micromorphic approach (Eringen, Mindlin) to multiple nested sub-scales: internal power splits into a homogeneous part (macro-stress:macro-velocity-gradient) plus inhomogeneous parts at each sub-scale (micro-stress:relative-micro-deformation + double-stress:gradient-of-micro-deformation), generalized to N nested scales.

**Direct connection to the Chapter 6 localization/regularization discussion**: the book's own two-scale numerical example (void-sheeting/necking, compared against DNS and a classical Cauchy continuum) shows the classical continuum simulation is explicitly **"mesh dependent"** once softening/instability sets in — localization collapses into arbitrarily thin shear bands. The two-scale MCT model instead reproduces the DNS pattern well, because the micro-stress/double-stress terms carry an **intrinsic microstructural length scale** into the plastic potential — directly analogous to gradient/nonlocal regularization for mesh-dependent localization, except MCT supplies the missing length scale via physically-motivated micro-deformation kinematics rather than an ad hoc regularization parameter. A noted subtlety: referencing every scale's relative deformation to the same base velocity gradient can double-count energy for three or more scales, motivating an alternative sequential definition in more recent formulations at the cost of losing explicit concurrent cross-scale coupling.

### 3. RVE and scale-bridging

Rather than evaluating the homogeneous power density analytically, it's computed from an RVE via computational homogenization, invoking the **Hill-Mandel lemma** (valid under affine-velocity Dirichlet or affine-traction Neumann RVE boundary conditions). RVE-sizing requires separation of scales (macroscopic body scale ≫ RVE cell size ≫ microstructural feature spacing) plus statistical representativeness. Traction BCs on the RVE are simple but can go unstable under softening, so velocity/displacement BCs are often preferred; periodic BCs are also discussed. The chapter's numerical example shows MCT elements could be an order of magnitude coarser than the DNS mesh while still reproducing DNS softening/localization behavior, without needing to remesh when microstructural parameters change — presented as the central practical advantage of MCT for parametric materials-design studies.

---

## Chapter 13: Single Crystal Plasticity

### 1. Kinematics: specializing F = Fe·Fp for crystals

Starting from the general multiplicative decomposition of Chapter 5, plastic flow is constrained to a small discrete set of crystallographically fixed slip systems (slip direction s^α, plane normal n^α, Schmid tensor S^α=s^α⊗n^α). Plastic rate-of-deformation and spin are linear combinations of slip rates γ̇^α over all active systems: Dp=Σγ̇^αP^α, Wp=Σγ̇^αQ^α — replacing the smooth yield-surface-normal flow direction of general plasticity with a discrete crystallographic constraint: "a crystal can only deform plastically along special planes and directions, unlike the homogeneous continuum material point." Plastic incompressibility falls out automatically from s^α·n^α=0. For large-deformation metal plasticity the book adopts Fe≈I (elastic lattice stretch negligible), tracking plastic deformation directly in the near-deformed configuration with Cauchy stress from a hypoelastic law using De≡D−Dp. If slip is active on multiple non-coplanar systems, Fp is generally *incompatible* (not the gradient of any single-valued displacement field) — only the product FeFp must remain compatible.

### 2. Flow rule and stress update

The **resolved shear stress** on each system is τ^α=P^α:σ_dev. Slip is possible when this exceeds a **critical resolved shear stress** computed from the immobile dislocation density via Taylor's relation (τ^α_ref=Gb√(ρ_im/d)), or a poly-slip/thermally-modified version summing cross-system hardening via Taylor interaction coefficients plus a thermal-softening term. Rather than a rate-independent Schmid criterion, the book adopts a **rate-dependent power-law flow rule** (Kocks 1987): γ̇^α=γ̇_ref(τ^α/τ^α_ref)^(1/m). Combined with the material-rate equation for τ^α (a hypoelastic/corotational rate involving lattice spin), this gives a **coupled, stiff system of nonlinear ODEs**, since each slip system's evolution depends on all others through the shared deformation-rate tensor — explicitly flagged as needing mixed explicit-Runge-Kutta-with-implicit-switching or dedicated stiff-ODE solver libraries. The full algorithm is a **semi-implicit** procedure (Box 13.2) suitable for a VUMAT-style explicit dynamic material subroutine, including an adiabatic temperature update feeding thermal softening back into the next step. Choosing this rate-dependent power law from the outset sidesteps the classical rate-independent-crystal-plasticity ambiguity of which subset of simultaneously-critical slip systems is actually active.

### 3. Hardening: dislocation-density-based self/latent hardening

Hardening is built bottom-up from evolving mobile/immobile dislocation densities per slip system (Estrin & Kubin 1986 framework), with evolutionary ODEs balancing generation, annihilation, and self/cross-system (latent) interaction terms. This density directly determines crystal strength via the Taylor relation (self-hardening), while the poly-slip generalization sums immobile density from *all* systems weighted by Taylor interaction coefficients — this cross term is the **latent hardening** contribution. The dislocation-density formalism is also noted to enable straightforward stability analysis for predicting onset of localized shear banding.

### 4. Explicit numerical/implementation warnings

- The resolved-shear-stress update is explicitly flagged as a **coupled, stiff nonlinear ODE system**, requiring adaptive explicit/implicit switching or dedicated stiff-solver libraries — a direct numerical-robustness warning.
- Anisotropic elasticity is deliberately simplified to isotropic (single shear modulus) in the resolved-shear-stress-rate equation — "errors introduced by this simplification are typically small when large inelastic deformations are being modeled" — a stated, deliberate accuracy/cost tradeoff.
- Plastic incompatibility of Fp under multi-system slip (§5's kinematic subtlety, specialized here) is noted to be satisfied automatically by construction of the algorithm.
- The worked numerical example (FCC aluminum single crystal, VUMAT in ABAQUS/Explicit) demonstrates that a standard **isotropic J2 plasticity model cannot reproduce** the anisotropic, slip-system-oriented shear-band localization the dislocation-density crystal-plasticity model correctly predicts — underscoring that phenomenological isotropic plasticity is *structurally* incapable of capturing this behavior, not just quantitatively off.

---

## Bibliographic note

This document synthesizes eight parallel chapter-group studies of Belytschko, Liu, Moran & Elkhodary (2014), *Nonlinear Finite Elements for Continua and Structures*, 2nd ed., conducted by extracting the book's text via `pypdf` and dispatching each chapter group to a dedicated reading pass with targeted implementation-focused questions. Section, equation, and box numbers cited throughout refer to the book's own numbering as it appears in the extracted text; page numbers where cited refer to the book's internal pagination, not the PDF page index. This is a study/synthesis document, not a verbatim reproduction — direct quotations are marked and kept short; all other text is paraphrase and synthesis.

See also: `fem_implementation_lessons.md` (companion study of Gockenbach's *Understanding and Implementing the Finite Element Method*, covering linear FEM implementation mechanics — data structures, assembly, linear solvers, adaptive refinement — which this document assumes as background and does not repeat).

---

## Appendix: Implementation status against `fea_engine`

Every lesson topic above was cross-checked against the actual `fea_engine` source (`computation-suite/fea_engine/src/fea_engine/`) by five parallel code audits, one per chapter group, each grepping and reading the real implementation rather than trusting docstrings or roadmap claims. Short answer: **a lot of Chapters 1–6, 8–9 is genuinely implemented, and essentially none of Chapters 7, 10 (beyond a narrow slice), 11, 12, or 13 is.** The package is a Total-Lagrangian, hyperelastic/small-strain-J2, corotational structural-element code — it doesn't attempt ALE, general contact, fracture (XFEM), multiresolution continuum theory, or crystal plasticity, and it doesn't try to hide that fact: several of its own docstrings and roadmap docs candidly note the gaps (e.g. the corotational shell's rejected mixed-formulation attempt, the "MVP scope" note on its first-order rotation update).

### Summary table

| Book lesson | Status | Key evidence |
|---|---|---|
| Deformation gradient F, Green strain E=½(FᵀF−I) | **Implemented** | `Tet10SolidTL`, `Tet4NeoHookean`, `TrussTL2D`, `Beam2DCorotational` all form F and E per the book's H-based recipe |
| Large-rotation / geometric-nonlinear elements | **Implemented** | `TrussTL2D`, `Beam2DCorotational`, `Tet4NeoHookean`, `Tet10SolidTL`, `Shell4MITCCorotational` |
| Total Lagrangian ∫B₀ᵀP dΩ₀ formulation | **Implemented** | `Tet10SolidTL`/`Tet4NeoHookean` internal force, docstrings cite this book's Ch. 4 by name |
| Updated Lagrangian formulation | **Not Implemented** | No element integrates σ over the current configuration with spatial B |
| TL/UL equivalence exploited as a design principle | **Partial** | Corotational elements implicitly use the "redefine reference = current" trick but never state it as such |
| Explicit stress-measure conversion (S↔P↔σ) | **Partial** | Conversion is baked into `F@S@dN_dX`, never exposed as a standalone utility or used for stress recovery/output |
| Objective stress rates (Jaumann/Truesdell/Green-Naghdi) | **Not Implemented** | Zero hits repo-wide; no hypoelastic/rate-form material exists to need them |
| Rate-of-deformation D vs. total strain | **Not Implemented** | No velocity-gradient computation anywhere; everything is total-Lagrangian |
| Voigt kinematic vs. kinetic convention | **Implemented** | Consistently applied across every B-matrix / stress-vector pairing |
| DOF-transformation rule (f̂=Tᵀf, K̂=TᵀKT) | **Partial** | Used pervasively for corotational transforms; master-slave tie/MPC use (Example 4.5) has no counterpart at all |
| Constitutive-routine architecture (converged-state-only updates) | **Implemented** | `material.j2_radial_return_3d`, `TrussPlastic2D`, enforced structurally via `FESystem.commit_all_states()` |
| Return-mapping / radial return for J2 | **Implemented** | Closed-form Box-5.14-style scalar radial return for 3-D/plane-strain J2 and the 1-D truss case; plane-stress J2 explicitly out of scope |
| Algorithmic (consistent) tangent modulus | **Implemented** for J2 plasticity; **inconsistent elsewhere** | `C^alg = C^ep − 2μβÎ`-style closed form in `j2_radial_return_3d`; hyperelastic elements (`Tet4NeoHookean`) instead use finite-difference/complex-step tangents |
| Incremental objectivity (Hughes-Winget) | **Not Implemented** | No large-strain/large-rotation plasticity element exists to need it |
| Multiplicative decomposition F=Fe·Fp | **Not Implemented** | Plasticity is additive small-strain only; hyperelasticity is purely elastic with no plastic coupling |
| Neo-Hookean hyperelasticity | **Implemented** | PK2/Green-strain-based, correct isochoric/volumetric split, but elastic-only (no Fe·Fp path to plasticity) |
| Non-associative / kinematic hardening | **Implemented** (Wave 2 item 14, 2026-09-08) | `PlasticMaterialJ2Kinematic`/`j2_radial_return_3d_kinematic()` (material.py): combined linear-isotropic + Armstrong-Frederick nonlinear kinematic hardening, driven by a scalar local Newton on `dgamma` derived from a direction-consistency argument (full derivation in the function's docstring); local-Newton slope and consistent tangent are both finite-difference past that point, the same choice `Tet4NeoHookean.tangent_stiffness()` already makes for an analogous reason. `Hex8PlasticJ2Kinematic(Hex8PlasticJ2)` drives it; plain `Hex8PlasticJ2`/`j2_radial_return_3d()` untouched. `tests/test_plasticity_kinematic.py` (6 checks, including an independently-derived closed-form cross-check at `gamma_AF=0` and the AF saturation-limit asymptote) |
| Crystal plasticity (Ch. 13) | **Not Implemented** | Zero trace of slip systems, Schmid tensor, dislocation-density hardening |
| Newton's method, full and modified | **Implemented** | Full Newton is the default; modified/chord Newton exists in the Koiter-Newton driver family |
| Convergence criteria (scaled) | **Implemented** (Wave 1 item 9, 2026-09-07) | Force-scaled residual-norm stays the default/primary check; displacement-increment (`du_tol`) and energy-error (`energy_tol`, Belytschko & Schoeberle 1975) added as OPTIONAL, AND-combined criteria to `solve_nonlinear_static`, `solve_nonlinear_displacement_control`, `solve_nonlinear_arc_length`, `solve_nonlinear_koiter_newton`, and `solve_nonlinear_static_koiter_newton`'s corrector, via the shared `nonlinear_solver._extra_convergence_ok()` helper. `solve_nonlinear_koiter_newton_generic`'s two correctors deliberately excluded — see that function's own docstring for why |
| Line search | **Implemented** (Wave 1 item 10, 2026-09-07) | Armijo backtracking (the same algorithm already validated as the dynamic Newmark driver's own fallback) extended, as a FALLBACK-ONLY retry after plain Newton exhausts `max_iter`, to `solve_nonlinear_static`, `solve_nonlinear_displacement_control`, `solve_nonlinear_arc_length` (backtracking the coupled (du,dlambda) Crisfield correction), and `solve_nonlinear_static_koiter_newton`'s frozen-tangent corrector, via the shared `nonlinear_solver._armijo_line_search_step()` helper. Deliberately NOT added to `solve_nonlinear_koiter_newton`'s or `solve_nonlinear_koiter_newton_generic`'s bordered continuation correctors (a real, previously-debugged sign-error history lives in that exact code, and they already have their own, different globalization -- shrinking the predictor's perturbation parameter) — left as documented future work |
| Material/geometric tangent stiffness split | **Mostly Implemented** | Analytic split for `TrussTL2D`, `Beam2DCorotational`, `beams3d.py`, `Tet10SolidTL`, `Shell4MITCCorotational`; `Tet4NeoHookean` and the curved-contact element deliberately use numerical tangents instead |
| Algorithmic tangent in the global Newton loop | **Implemented** | `Hex8PlasticJ2` assembles `K_T` directly from the consistent `D_ep`, following the "secant Newton"/last-converged-state discipline |
| Arc-length / Riks continuation | **Implemented (variant)** | Crisfield cylindrical arc-length (omits the load-factor term from the book's spherical constraint — a documented simplification), extended by a genuine multi-mode Koiter-Newton bifurcation tracker |
| Critical-point / stability detection | **Implemented** | `solve_linear_buckling()` generalized eigenproblem matches the book's one-shot buckling estimate; an online near-critical-eigenvalue check drives the Koiter-Newton bifurcation trigger; no literal det(A)=0 sign test |
| Explicit central-difference integration + CFL | **Partial** | Genuine lumped-mass central-difference integrator with a spectral (global) critical-timestep estimate — but linear-only (no nonlinear/contact explicit driver) and no local per-element wavespeed CFL formula |
| Material stability / localization / regularization (§6.7) | **Not Implemented** | No acoustic tensor, no ellipticity check, no softening/localization regularization of any kind (gradient, nonlocal, viscoplastic, or Hillerborg) |
| ALE formulation | **Not Implemented** | Zero trace of independent mesh motion, mesh smoothing/rezoning, or convective-term stabilization |
| Contact: gap/Kuhn-Tucker formulation | **Implemented** (Wave 3 item 16, 2026-09-08) | `GapContactPenalty`/`GapContactCurvedFriction` (untouched) still give a real unilateral-contact status switch against a fixed/analytic obstacle; `NodeToSegmentContact2D` (elements/contact.py) adds a genuine two-deformable-body node-to-segment discretization, registered via the existing `FESystem.add_contact_element()` mechanism with no solver changes. `tests/test_node_to_segment_contact.py` CHECK 5: a real `TrussTL2D`-post-vs-`Quad4PlaneStress`-block system, both sides measurably deform |
| Contact enforcement (penalty/Lagrange/augmented Lagrangian) | **Implemented** (Wave 3 item 15, 2026-09-08) | Penalty (`GapContactPenalty`) and the exact single-constraint pure Lagrange-multiplier solve (`solve_contact_lagrange_static()`) both still exist unchanged; `solve_contact_augmented_lagrange_static()` adds a genuine augmented-Lagrangian scheme (outer Uzawa multiplier update wrapping an ordinary inner penalty-style Newton solve). Perturbed Lagrangian remains absent. `tests/test_augmented_lagrangian_contact.py`: at a matched `k_p`, AL's relative error vs. the exact reaction is `4.7e-9` vs. plain penalty's `2.1e-2` |
| Coulomb friction (stick-slip return mapping) | **Implemented** (Wave 3 item 17, 2026-09-08) | `GapContactCurvedFriction` (untouched) still implements stick/slip for one node against a rigid circle; `NodeToSegmentContact2DFriction` generalizes the identical stick/slip decision structure to a node sliding along a (possibly deforming) master segment. `tests/test_node_to_segment_contact.py` CHECK 6 |
| General contact search (node-to-segment, broad-phase) | **Implemented (narrow scope)** (Wave 3 item 16, 2026-09-08) | `find_contact_pairs_2d()` (elements/contact.py): an exhaustive O(n_slave*n_segments) closest-point DISTANCE FILTER, not a spatial-hash/BVH broad phase -- documented as adequate for small-to-moderate contact zones, a drop-in replacement point for a real spatial index if large-scale contact is ever needed |
| XFEM (enrichment, level sets, phantom nodes) | **Not Implemented** | Zero trace anywhere in source, examples, or tests |
| Multiresolution continuum theory | **Not Implemented** | No micromorphic/higher-grade kinematics, no RVE/homogenization machinery |
| Patch tests | **Implemented (solids only)** | Genuine near-machine-precision patch tests for Tri3/Quad4/Quad8/Tet4/Tet10/Hex8/Hex20 and a mixed mesh; no patch test exists for any beam or shell element |
| Volumetric locking cures (B-bar/SRI/mean-dilatation) | **Implemented** (Wave 2 item 11, 2026-09-08) | `Hex8SolidBbar(Hex8Solid3D)` (elements/solids.py): Hughes (1980) mean-dilatation B-bar at full 2×2×2 integration (no reduced integration, no hourglass risk). Proven and verified to reduce exactly to `Hex8Solid3D` on affine fields; `tests/test_hex8_bbar.py` shows plain `Hex8Solid3D`'s cantilever-bending ratio collapsing 0.71→0.18 as ν→0.4999 while `Hex8SolidBbar` stays at 0.73→0.74. Plain `Hex8Solid3D` untouched |
| Reduced integration + hourglass control | **Implemented** (Wave 2 item 12, 2026-09-08) | `Element.hourglass_stabilized_stiffness()` (elements/base.py): a generic DEFLATED-EIGENPROBLEM stabilizer (not the literature's tabulated Flanagan-Belytschko/Belytschko-Bindeman shape vectors, deliberately, to avoid hand-transcription risk) -- eigendecomposes `ke_reduced`'s null space, projects `ke_full` onto it, and separates genuine rigid-body modes from spurious/hourglass ones by their energy in `ke_full`. Mathematically proven and numerically verified to vanish exactly on affine fields; works on any `Element` subclass with no per-element code. Wired via `FESystem.assemble_stiffness(D, method="full"\|"reduced"\|"hourglass_stabilized")`, default `"full"` unchanged. `tests/test_hourglass_stabilization.py` shows the `examples/main.py` cantilever's full-integration ratio 0.71 improving to 1.01 (hourglass-stabilized), much closer to the Euler-Bernoulli reference |
| Shear-locking cures for beams/shells (MITC/ANS) | **Implemented** | `Shell4MITC` implements genuine Dvorkin-Bathe assumed-natural-strain tying-point shear interpolation; `Quad4MindlinPlate` uses SRI |
| Membrane-locking cure (MITC4+, Ko/Lee/Bathe 2017) | **Investigated, not implemented** (Wave 4 item 22, 2026-09-08) | A working implementation was built and numerically validated against the paper's own flat-element equivalence guarantee, then deliberately reverted: proved that the scope needed to avoid membrane-bending DOF coupling (displacement kept purely in-plane) makes the fix an EXACT no-op for warped elements too, not just flat ones -- a warped element's out-of-plane geometry can only ever reach the membrane strain calculation through a displacement component that isn't there. A real fix needs genuine w-coupling into the membrane block, carrying the same dead-end risk documented for `Shell4MITCCorotational`'s large-rotation work below. Full finding recorded in `elements/shells.py`'s module docstring |
| Continuum-based (director/fiber) beam/shell kinematics | **Not Implemented** | Beam/shell elements use classical resultant/structural theory (Hermite beam, membrane+Mindlin-plate composition), not the book's degenerated-solid director approach — acknowledged candidly in the code's own docstrings |
| Large-rotation update (Rodrigues/Hughes-Winget) | **Mostly Implemented** (Wave 4 items 18+23, 2026-09-08) | Exact for the 2-D corotational beam (trivial there); the corotational shell's DRILLING frame now uses an EXACT single-axis rotation (item 18, `Shell4MITCCorotational._exact_drill_rotation()`) instead of the first-order truncation, verified to reduce a documented ~1.6%-at-30-degrees error to floating-point-exact zero at every angle tested up to 90 degrees -- the BENDING block remains uncorrected (item 20, investigated and deferred, see the roadmap doc). `Beam3DCorotational` (item 23) is now implemented: full 3-D corotational frame (minimal-rotation transport + exact single-axis twist correction), validated to exact-zero rigid-motion force about ANY axis, not just the beam's own -- built via real central-FD tangent (Phase A, analytic Battini-Pacoste tangent deferred as documented future work) rather than a hand-derived exact-SO(3) tangent |
| Drilling-DOF / flat-shell singularity | **Implemented** | `Shell4MITC` uses an Allman-type artificial drilling stiffness, with a docstring naming the exact failure mode from the book |
| Hu-Washizu / mixed variational formulations | **Not Implemented (attempted and rejected)** | A mixed Hellinger-Reissner-style extension to the corotational shell was tried and explicitly abandoned after Newton-divergence failures — a documented negative result, not a silent omission. Revisited (Wave 4 item 19, 2026-09-08): an implicit-function-theorem argument shows the "sub-iterate the internal unknown to convergence" follow-up path reproduces the original failed tangent exactly, by construction (the internal unknown's governing equation is affine, so its "convergence" is a one-step algebraic solve, not an iterative process) — not a new experiment, a proof from the element's own already-established force/tangent split. The other follow-up path (true joint global DOFs, a monolithic block-saddle-point solve) is not ruled out by this argument but is deferred as disproportionate scope for one non-load-bearing coupling term. A separate new hypothesis for the underlying coupling term itself (Wave 4 item 21: element-constant, not bilinear, absolute rotation) was also tried and found to reproduce the same Newton-convergence failure on a real assembled-mesh solve (Dead End 8) |

### What this means in practice

`fea_engine` is not a general-purpose nonlinear FE code in the sense the Belytschko book covers — it's a focused Total-Lagrangian / corotational structural-and-solid-element code with real, working large-deformation kinematics, a correct J2 return-mapping plasticity implementation with a genuine consistent tangent, Neo-Hookean hyperelasticity, Newton/arc-length/Koiter-Newton bifurcation-tracking solvers, a linear buckling eigensolver, and MITC-style shear-locking cures for its shell element. That is a substantial and mostly textbook-faithful slice of the book (roughly Chapters 1–6, 8–9, minus their most advanced corners).

What it deliberately does not attempt: anything requiring a mesh that moves independently of material (ALE, Ch. 7), general deformable-body contact with real geometric search (Ch. 10 — the existing contact elements only handle a single node against a fixed rigid obstacle), fracture/discontinuity representation (XFEM, Ch. 11), sub-grid microstructural physics (multiresolution continuum theory, Ch. 12), or crystallographic plasticity (Ch. 13). It also has two "half-finished" areas worth flagging for anyone extending the package: volumetric locking has no cure at all in the 3-D solids (full integration everywhere, no B-bar/SRI/mean-dilatation), and hourglass control has a detector but no stabilizer, so reduced integration exists in the code but isn't safe to actually use. Convergence checking and line search are also both narrower than the book recommends — residual-norm-only tolerances, and Armijo backtracking scoped to a single dynamic-driver fallback rather than available everywhere Newton is used.
