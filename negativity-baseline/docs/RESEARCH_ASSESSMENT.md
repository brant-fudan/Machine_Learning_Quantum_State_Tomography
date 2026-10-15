# Literature, novelty gate and a 6-week undergraduate plan

## Research status as of 2026-10-15

**Most important overlap:** Koutný, Ginés, et al., *Deep learning of quantum entanglement from incomplete measurements*, *Science Advances* **9**, eadd7131 (2023), DOI https://doi.org/10.1126/sciadv.add7131. They directly infer entanglement-related quantities from incomplete local projective measurements, compare to maximum-likelihood tomography and investigate unseen Werner families. Their paper quantifies **concurrence** (two qubits) and **mutual information** (larger systems); it does not make the point-estimate-versus-guaranteed-Negativity distinction developed as an *educational baseline* here. None of our current components individually warrants a novelty claim.

**Earlier relevant theory:** Gühne, Reimpell & Werner, *Estimating Entanglement Measures in Experiments*, *PRL* **98**, 110502 (2007), DOI https://doi.org/10.1103/PhysRevLett.98.110502; and *Lower bounds on entanglement measures from incomplete information*, *PRA* **77**, 052317 (2008), DOI https://doi.org/10.1103/PhysRevA.77.052317. Lower-bounding entanglement with incomplete measurements and witness/convex methods is **not new**.

**Earlier ML/tomography:** Carrasquilla, Torlai, Melko & Aolita, *Reconstructing quantum states with generative models*, *Nature Machine Intelligence* **1**, 155–161 (2019), DOI https://doi.org/10.1038/s42256-019-0028-1 (motivation for legacy RNN). Huang, Kueng & Preskill, *Predicting many properties of a quantum system from very few measurements*, *Nature Physics* **16**, 1050–1057 (2020), DOI https://doi.org/10.1038/s41567-020-0932-7 (classical shadows/property learning; linear-observable concentration results do not automatically guarantee an arbitrary nonlinear Negativity predictor).

**Nearby 2024–2026 work:** Pan et al., *Quantifying entanglement for unknown quantum states via artificial neural networks*, *Scientific Reports* **14**, 26267 (2024), DOI https://doi.org/10.1038/s41598-024-76978-7, explicitly uses partial Pauli correlations as NN features. Huang et al., *Direct entanglement detection of quantum systems using machine learning*, *npj Quantum Information* **11**, 29 (2025), DOI https://doi.org/10.1038/s41534-025-00970-w. *Neural quantum states for entanglement-depth certification from randomized Pauli measurements*, *Physical Review Research* **8**, 023255 (2026), DOI https://doi.org/10.1103/29hj-7w41, uses constrained neural ansätze and likelihood gaps rather than our two-qubit Negativity task. *Machine Learning of Quantum Entanglement from Noisy Measurements*, *Electronics* **15** (2026) 3197, DOI https://doi.org/10.3390/electronics15143197, considers simulated noisy two-qubit SIC-POVM features; it is not a controlled benchmark against our Pauli-box certificates.

**Especially relevant current 2026 prior art (important novelty constraint):** Seong, Kim, Seo *et al.*, *Practical and efficient verification of entanglement with incomplete measurement settings*, *APS Open Science* **1**, 000008 (published **29 April 2026**), DOI https://doi.org/10.1103/sgbs-7228; preprint https://arxiv.org/abs/2512.09856. This paper already designs multiple entanglement witnesses from incomplete measurement settings, **uses SDP optimization to find useful witnesses**, and gives a two-photon polarization experiment. Therefore **"SDP + incomplete measurements + entanglement detection" is demonstrably not novel**, even beyond the Gühne--Reimpell--Werner predecessors. An adaptive neural proposal must be checked against this paper's best witnesses as well as against existing optimal measurement-allocation literature. Seong *et al.*, *Mirrored entanglement witnesses for multipartite and high-dimensional quantum systems*, *Quantum Science and Technology* **11**, 015037 (2026), DOI https://doi.org/10.1088/2058-9565/ae30a3, supplies another recent advanced witness benchmark. Their objectives are not automatically the same as guaranteed **Negativity magnitudes**, so claims of exact equivalence also require care.

These were verified using accessible publication pages/abstracts. The list is selective rather than an exhaustive systematic review; in particular, no claim of primacy is made for adapting a learned witness or finite-statistics certificate.

## What the reproducible run supports

The training set (Wishart mixed+explicit separable) and OOD family (Bell mixtures+rotated near-pure) are deliberately different, making it a stress test, not an i.i.d. accuracy leaderboard. At n=1024 per setting, the direct MLP's OOD MAE is ≈0.21 whether measurements are full or incomplete; the physical LS MAE is ≈0.006 with full data but ≈0.142 with incomplete data. This is **one seed, small MLP, finite training**, not a proof that quantum-state property regression is intrinsically inferior. The empirical results highlight distribution prior dependence and information constraints.

The analytically indistinguishable pair demonstrates a stronger, unconditional negative conclusion: in the XX/ZZ protocol, arbitrary model complexity or shots cannot resolve its differing Negativities. The compatible-state SDP minimum correctly returns zero on exact partial measurements and positive on exact full measurements. This counterexample is a didactic use of a known identifiability principle, not a new impossibility theorem on its own.

For finite-statistics certification, the pair-correlation and spectral witnesses give per-experiment `1-alpha` guarantees conditional on trusted iid Pauli devices. Empirical nonviolations observed in the run are sanity checks, not independent proof. The **numerical SDP path still needs a high-precision conic solver and dual-certificate audit**. Conflating point estimates and such certificates would be a substantive scientific error.

## Specific paper-worthy questions (not accomplishments yet)

**A. Finite-shot design of *certifiable* information (not a novelty claim by itself).** Given a fixed **total shots** budget and a library of local Pauli settings, choose measurements adaptively to maximize the valid bound or minimize worst-case uncertainty (not just improve neural MAE). Compare principled SDP-dual sensitivities, simple witness heuristics, uncertainty sampling, **Seong et al. 2026 incomplete-measurement witness optimization**, and matched-budget random baselines. Novelty requires a provable property, a nontrivial separation, or broad credible experimental superiority beyond known adaptive tomography and measurement-selection works.

**B. Risk-controlled NN-assisted certification.** Let an NN cheaply propose a measurement setting or a feasible witness operator, then use simultaneous fresh finite-shot confidence boxes or an anytime-valid martingale to preserve coverage even with adaptive selection. A guarantee under sequential stopping/adaptivity and quantified certification *power* versus classical baseline could be interesting. Pure plug-in witness plus Hoeffding, as implemented here, is standard and insufficient by itself.

**C. Geometry of incomplete measurement equivalence classes.** Characterize the diameter of the Negativity interval for physically compatible states, the sample complexity of shrinking the interval, and which measurement choices separate ambiguous state families. Exact tight two-qubit results are more scientifically valuable than adding a more complex network. The current XX/ZZ ambiguity example is an entry point only.

Avoid projects limited to changing MLP to CNN/Transformer/GRU and reporting a 1–2% MAE gain, especially against unoptimized linear-inversion tomography; Pan et al. 2024 and Koutný et al. 2023 already cover the broad empirical idea.

## Suggested 6-week undergraduate trajectory

| Week | Verifiable deliverable |
|---|---|
| 1 | Derive partial transpose, XX/ZZ indistinguishability and tomography rank by hand; run physics+ambiguity tests. |
| 2 | Run 3 seeds, audit shot allocation, reproduce all 5 figures; understand F1/Brier, especially prevalence shift. |
| 3 | Install CVXPY + a conic solver; compare SLSQP/CVXPY on all 768 instances with primal/dual diagnostics; label solver failures. |
| 4 | Build **matched-total-budget** protocols; test a range of finite-shot entanglement strengths near PPT boundary. |
| 5 | Prototype active selection with *honestly split* pilot and certification shots, benchmark against analytic witness-aware allocation. |
| 6 | Quantify gains over strong baselines with 5–10 seeds, bootstrap CI **across independent states**, review nearest prior art and formulate one modest proposition or falsifiable hypothesis. |

**Go/no-go:** A publishable manuscript should offer a theorem, a genuinely new reliable statistical protocol, or systematic improvements over competitive prior methods with careful costs and coverage. A working GRU/MLP baseline and a known ambiguity example alone do not merit a physics journal submission. Potential venue cannot be assessed responsibly until that extra contribution is established.
