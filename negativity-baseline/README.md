# Two-Qubit Negativity Learning: predictions vs certificates

**A reproducible AI-for-Quantum student baseline (October 2026).** Independent of the legacy GHZ/RNN tomography data file. It uses the **ideas**, not a verbatim copy, of the GRU autoregressive generator in [Machine_Learning_Quantum_State_Tomography](https://github.com/brant-fudan/Machine_Learning_Quantum_State_Tomography).

中文说明见 `docs/STUDENT_GUIDE.md`；原理、认证保证及复现注意事项见 `docs/METHODS.md`；独立创新性判断见 `docs/RESEARCH_ASSESSMENT.md`。

## Quick start

Python 3.10+ recommended; tested with Python 3.13.5 (CPU), NumPy 2.3.5, SciPy 1.17.0 and PyTorch 2.10.0.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py --quick --sdp-engine slsqp
```

On Windows PowerShell: run `python run.py --quick --sdp-engine slsqp` without the Unix environment-variable prefix.

For the actual SDP **conic** solver path (CVXPY + CLARABEL/SCS), install `requirements-sdp.txt` and run `python run.py --quick --sdp-engine cvxpy`. The CPU tested environment had no CVXPY and no package-download access. **The reference quick-run numbers use SciPy SLSQP to solve the equivalent eigenvalue-constrained SDP numerically. They are not rigorously certified conic optima.** A separate fully valid (assumption-conditional) finite-shot lower confidence bound is computed without an SDP solver.

For a longer run: `python run.py --train 10000 --validation 1200 --test 200 --epochs 70 --sdp-subset 40 --seed 42`. The baseline neural architecture is an MLP; to exercise the adapted legacy recurrent architecture: `python run.py --quick --model gru` (this overwrites the default result tables, so make a copy first).

## Dataset and methods

* **Physics:** two-qubit density matrices; Negativity is the sum of negative eigenvalues of the partial transpose in absolute value, between 0 and 0.5. In 2×2, PPT exactly detects separability.
* **Measurements:** `full` = all 9 Pauli setting pairs `{X,Y,Z} × {X,Y,Z}` (informationally complete); `partial` = `XX`, `ZZ` (informationally incomplete). Each setting has four joint outcomes `(++,+−,−+,−−)`.
* **Statistical model:** multinomial outcomes, fixed *n shots per setting*, n ∈ {64,256,1024}; full uses `9n` copies, partial `2n` copies. **Comparisons at the same n do not have matched total measurement budgets.** No device miscalibration, drift or correlated shots.
* **Features:** 15 non-identity Pauli moments (pooled single-qubit marginals), 15 observation masks, and 15 inverse-square-root effective-shot indicators. Missing moments are NOT treated as known zeros.
* **Supervised predictor:** MLP with regression + separate classification head (optional GRU analog of the previous code, but not its generative log-likelihood).
* **Physical tomography:** convex positive-semidefinite constrained **least squares** in Pauli coordinates, solved by projected FISTA (not MLE, and not the unique true state under incomplete measurements).
* **Measurement-constrained SDP:** minimum compatible Negativity over physical states and simultaneous confidence intervals for observed moments; implemented with CVXPY and with a small 4×4 SciPy numerical fallback.
* **Guaranteed certificate:** a two-setting pair-correlation inequality and a data-adaptive spectral witness, each minimized over simultaneous Bonferroni-Hoeffding confidence boxes. This is an **at-least-95% one-state confidence statement under trusted iid measurement assumptions**, not a 95% claim that every sample in the CSV simultaneously satisfies it.
* **Distribution shift:** training + IID testing are 50% Hilbert-Schmidt Wishart full-rank mixed states and 50% explicit separable mixtures. OOD testing uses 50% Bell-isotropic/Werner-type states and 50% locally rotated, nearly pure entangled states. Different distributions have very different Negativity prevalence.

## Reproduced CPU quick run (seed=42)

Training 2,400 states; IID validation 320; 64 held-out states per (IID/OOD)×(full/partial)×(shots=64/256/1024) condition (768 test entries); 36 training epochs; 768 SDP attempts; 763 SLSQP runs reported success. SLSQP attempts were run for **all** 768 entries, not just a favorable subset. Reference results are preliminary (single seed, deliberately small NN, no hyperparameter search).

| Test | Settings | Shots/setting | MLP Negativity MAE | Physical LS Negativity MAE | Provably positive confidence-bound fraction |
|---|---|---:|---:|---:|---:|
| IID | `XX`, `ZZ` | 1,024 | 0.0326 | 0.0291 | 0.0% |
| IID | all nine | 1,024 | 0.0334 | 0.0047 | 7.8% |
| OOD | `XX`, `ZZ` | 1,024 | 0.2103 | 0.1419 | 23.4% |
| OOD | all nine | 1,024 | 0.2120 | 0.0061 | 64.1% |

For 64 and 256 shots/setting consult `results/metrics.csv` and `results/certificate_summary.csv`. **The MLP is weaker than physical LS in this limited run**, especially under distribution shift. Do not interpret the performance of this deliberately small baseline as a statement about all neural networks.

The sample-by-sample CSV also reports `sdp_numerical_lb` and `sdp_success`. They must not be confused with a mathematically valid numerical dual SDP certificate. For certified results use `certified_combined_lb = max(certified_pair_lb, certified_spectral_lb)`.

## The crucial non-identifiability example

Define |Φ+⟩=(|00⟩+|11⟩)/√2 and

- ρ_E(p)=p|Φ+⟩⟨Φ+|+(1−p)I/4, with 1/3 < p ≤ 1/2;
- ρ_S(p)=[I⊗I+p X⊗X+p Z⊗Z]/4.

The `XX` and `ZZ` four-outcome distributions, **not just two scalar correlations**, are identical, yet ρ_S is separable while ρ_E is entangled for 1/3 < p ≤ 1/2. At p=0.45: N(ρ_E)=0.0875 and N(ρ_S)=0. The same feature vector produces the same network output 0.05611; the ideal SDP compatible-state lower bound is 0. Any point predictor must err by at least 0.04375 on one of the pair. A `YY` measurement disambiguates the states (difference in ⟨YY⟩=0.45). Details, including proof, in `docs/METHODS.md`.

## Outputs

- `results/predictions.csv`: all 768 states, continuous and binary point predictions, finite-shot guaranteed witnesses, numerical SDP status.
- `results/metrics.csv`: MAE, RMSE, bias, AUC, F1, balanced accuracy, Brier score, false positives and prevalence.
- `results/certificate_summary.csv`: 95% bound detection and empirical coverage, and numerical SDP audit.
- `results/ambiguity.json`: exact indistinguishable-pair demonstration.
- `results/run_metadata.json`: seed, package versions, timing and design.
- `results/model_mlp_seed42.pt`: torch model weights for the seeded run.
- `figures/01_*.png` to `figures/05_*.png`: accuracy, OOD shift, valid bound detection, ambiguity and false positives, plus editable vector SVGs.
- `tests/test_physics.py`: independent physics/statistics/SDP/regression smoke tests.

All experiments are in calibrated, trusted local measurement models. No DI certification is claimed; the SDP's optimal value is a formal lower bound only when the confidence model and numerical optimization conditions are met.

## References

See `docs/RESEARCH_ASSESSMENT.md` for explicit overlap with Carrasquilla et al. (Nat. Mach. Intell., 2019), Gühne–Reimpell–Werner (PRL 2007; PRA 2008), Huang–Kueng–Preskill (Nat. Phys. 2020), Koutný et al. (*Science Advances*, 2023), Pan et al. (*Scientific Reports*, 2024), Seong *et al.* (*APS Open Science*, 2026) on SDP-optimized witnesses under incomplete measurements, and other 2025–2026 entanglement-ML works. **This package is a teaching/research baseline, not a novel method or a ready-to-submit paper.**
