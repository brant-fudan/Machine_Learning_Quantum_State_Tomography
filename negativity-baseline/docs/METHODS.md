# Mathematical specification and guarantees

## 1. Physical model, features and identifiable quantities

A two-qubit state is `rho >= 0`, `Tr rho=1` in the ordered computational basis |00>, |01>, |10>, |11>. We use the *standard* (not logarithmic) negativity:

`N(rho) = (||rho^(Gamma_B)||_1 - 1)/2 = sum_{lambda_i<0} |lambda_i|`, where `Gamma_B` transposes the second qubit. For two qubits, `N=0` iff the state is separable (Peres--Horodecki PPT criterion).

For a local Pauli measurement pair `(A,B)`, A,B ∈ {X,Y,Z}, and a,b ∈ {±1}, each joint outcome POVM effect is

`E_{A,B}^{a,b} = (I + a A) ⊗ (I + b B) / 4`;

`p_{A,B}(a,b) = [1 + a <A⊗I> + b <I⊗B> + ab <A⊗B>]/4`.

Hence nine settings span all 15 nonidentity Pauli coefficients; `XX` and `ZZ` reveal only six (`XI, IX, XX, ZI, IZ, ZZ`). Incomplete data cannot in general determine Negativity, even in the infinite-shot limit. The code uses independent multinomial sampling per **setting** and pooled marginal samples across settings where the same single-qubit Pauli is measured.

## 2. Exact counterexample and irreducible prediction risk

Let

`rho_E(p) = p |Phi+><Phi+| + (1-p) I_4/4 = (I_4 + p XX - p YY + p ZZ)/4`,

`rho_S(p) = (I_4 + p XX + p ZZ)/4`.

Here `XX = X⊗X`, etc. For `1/3 < p <= 1/2`, `rho_S` has eigenvalues `(1+2p)/4`, `1/4`, `1/4`, `(1-2p)/4` (all nonnegative), and its partial transpose is identical to itself (so it is separable by 2×2 PPT). By contrast `N(rho_E)= (3p-1)/4 >0`.

Both states have zero one-qubit Pauli expectations and `⟨XX⟩=⟨ZZ⟩=p`, so they generate exactly the same four-outcome probability tables for both measured settings. However, `⟨YY⟩_E=-p` versus `⟨YY⟩_S=0`. For *any* estimator f of just the XX,ZZ data, an identical feature distribution must lead to the same output distribution. At p=0.45 the two true Negativities differ by 0.0875. Any **deterministic estimate for the population probabilities** has worst-case absolute error ≥ 0.04375; with equal prior weights its mean absolute error is ≥0.04375. The impossibility is not a model-capacity or training issue.

## 3. Point estimates: different scientific meanings

**Direct MLP/GRU:** minimizes a supervised blend of negativity MSE and binary BCE on synthetic states. It learns conditional regularities of the *training prior*. Classification probabilities/MAE are empirical; they cannot serve as entanglement witnesses under distribution shift or observational ambiguity.

**PSD-constrained LS reconstruction:** minimize

`0.5 * sum_{i in observed} n_eff_i * (Tr(rho P_i) - mu_hat_i)^2`

subject to `rho>=0, Tr(rho)=1`. We solve by projected FISTA; each iteration projects the Hermitian matrix onto the PSD trace-one spectrahedron by simplex projection of eigenvalues. This is a convex **least-squares** estimator, not the multinomial maximum-likelihood estimator. Incomplete measurements admit multiple equally data-fitting physical states, so its selected completion has no special entitlement to the true Negativity. More iterations improve convergence; the output always remains physical after its final projection.

## 4. Data-consistent minimum Negativity is an SDP

For an expectation-value confidence box `l_i <= Tr(rho P_i) <= u_i` (observed i), define

`N_min(D) = min_{rho, Q} Tr(Q)`

subject to `rho>=0, Tr(rho)=1, Q>=0, rho^(Gamma_B)+Q>=0` and the box constraints. Q is Hermitian (the inequality enforces it). For any fixed physical rho, the minimum over Q equals `sum_{lambda<0}|lambda|`, so this is exactly the minimum compatible Negativity. If a confidence region contains the true rho, mathematically `N_min(D)<=N(rho_true)`; if `N_min(D)>0`, every compatible state is entangled. The derivation does not assume that the unknown state was drawn from the ML training prior.

The supplied **CVXPY** path is the clean conic-SDP implementation. The standard CPU runner without CVXPY uses `scipy.optimize.minimize(SLSQP)` on an *equivalent* formulation with eigenvalue nonnegativity and analytic eigenvalue gradients. Although the mathematical task remains an SDP, SLSQP is a general constrained nonlinear optimizer, **not an SDP conic solver**; it can fail or give a slightly optimistic minimum. We record failures and constraint violations. Report its numbers as `numerical SDP objective`, never as an independently rigorous certified dual lower bound. A publication-quality claim needs conic-solver checking, primal/dual gaps, and conservative numerical certificate verification (interval arithmetic if strict rigor is required).

## 5. Finite-shot confidence semantics

Each Pauli moment is estimated by the sample mean of `n_eff_i` iid ±1 samples from compatible local setting batches. Hoeffding implies

`Pr(|mu_hat_i-mu_i|>epsilon_i) <= 2 exp(-n_eff_i * epsilon_i^2 /2)`.

If m distinct nontrivial Pauli moments are observed, choose

`epsilon_i = sqrt(2 log(2m/alpha)/n_eff_i)`.

A union bound implies all true moments lie in the simultaneous box with probability ≥`1-alpha` for **each fixed state** and its iid measurement experiment. The moments' estimates need not be mutually independent; marginal shot-sharing is allowed. This does NOT cover calibration errors, non-iid memory/drift, or an adversarial quantum device, and it is not a DI bound. Multiple states in the CSV each have a marginal guarantee; they do not automatically share an experiment-wide `1-alpha` event.

## 6. Two independent finite-shot guarantees (no numeric SDP needed)

### Pair-Pauli certificate

For distinct `A,C in {X,Y,Z}` and distinct `B,D in {X,Y,Z}`, the observables `A⊗B` and `C⊗D` anticommute locally at **both** parties and commute globally. A local-unitary change of frames maps them, up to signs, onto `X⊗X` and `Z⊗Z`. Bell twirling is an LOCC channel preserving those two correlations. For the resulting Bell-diagonal state with correlators a,c, their joint Bell-basis distribution must have an atom of probability at least `(|a|+|c|)/2`. A Bell-diagonal state's negativity equals `max(0,p_max-1/2)`; negativity does not increase under LOCC. Consequently, universally:

`N(rho) >= max(0, (|<A⊗B>|+|<C⊗D>|-1)/2)`.

If the simultaneous confidence event holds, replace each absolute correlator by `max(0, |mu_hat|-epsilon)` to obtain a valid `1-alpha` lower bound. Maximizing over all eligible measured pairs is still valid on the same confidence event. The partial setting protocol has one pair (XX,ZZ); full measurements offer many pairs.

### Data-adaptive spectral witness

For **every** Hermitian operator Q with `0<=Q<=I`:

`N(rho) >= -Tr(Q rho^(Gamma_B)) = -Tr(Q^(Gamma_B) rho)`.

Take Q as the projector onto the negative eigenspace of the partial transpose of a PSD-LS reconstruction. Expand `Q^(Gamma_B)` in the 16 Pauli basis. Evaluate the **worst-case expectation** over every observed `l_i/u_i`, and for missing moments use the unconditional interval `[-1,1]`. The worst-case expectation yields a conservative, valid lower confidence bound. The fact that Q is selected *using the same observed data* does not invalidate the guarantee: the simultaneous moment box constrains the true state for all such affine witnesses at once. We clip only a negative lower bound to zero, never increase it by a numerical SDP optimum.

The final guarantee is `max(pair_bound, spectral_bound)`, since both are valid on the same simultaneous confidence event. This often certifies more entangled states than either individually; **it need not equal the optimum SDP bound**.

## 7. Tests and what they do not establish

Tests cover Bell/Werner analytical Negativity, partial transpose involution, Pauli reconstruction, indistinguishable pair, real finite shots, physicality of LS, an exact-measurement SDP limiting case, MLP/GRU forward output, and numerical certificate inequality in seeded samples. A single observed 100% empirical coverage on 768 synthetic samples is not a proof; the **Hoeffding derivation** supplies the conditional guarantee. They also do not establish that the NN has optimal capacity or generalizes to realistic hardware.
