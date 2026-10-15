# Legacy code audit (read from original GitHub repo, October 15, 2026)

Source: https://github.com/brant-fudan/Machine_Learning_Quantum_State_Tomography, branch `main`. Review covered root README and `RNN-based/{rnn.py,main.py,args.py,fid.py,utils.py,ghz_mps.py}`. This is a **static code/dependency audit**, not a successful end-to-end re-run of the old script (the tracked tree has no `data.txt`).

| Legacy part | Verified behavior/problem | Baseline choice |
|---|---|---|
| `rnn.py:RNN` | Learnable input embedding, BOS vector, trainable hidden initializer, `torch.nn.GRU`, categorical sampling and `log_prob`. The log-probability factors the **joint POVM outcome distribution**, not `N(rho)`. | Preserve recurrent bias through an optional GRU property-predictor (`models.py`) with measurement features and separate regression/classification heads. Do **not** blindly import the old RNN. |
| `main.py:prepare_data` | Loads literal `./data.txt`; the GitHub tree contains no such file. Uses removed `np.int`; fails on NumPy 2.x. | Generate seeded physically valid density matrices and finite-shot multinomial Pauli counts in memory. |
| `main.py:training` | `int(Ns/batch_size)` silently drops a remainder, walks fixed sequential batches; evaluation calls GHZ-only `classical_fidelity`, with early stop based on this reference state. | Shuffled training, explicit IID/OOD splits and three distinct evaluation objectives. |
| `args.py` | Parses process argv and changes global default torch tensor type at **module import**, risking unintended dtype/device effects; `--epoch` default 100000. | Encapsulated CLI under `if __name__ == '__main__'`, explicit NumPy/Torch dtypes, bounded quick-run sizes. |
| `rnn.py:generate` | Uses module-global `args.device` rather than its own `self.device` for sample tensor; imports GHZ/ncon even though core GRU class does not require them. | Device-neutral PyTorch inputs; no GHZ/ncon dependency. |
| `fid.py` | `classical_fidelity` evaluates POVM generative modeling against GHZ via Monte Carlo; some unused helpers also use `np.int`. | Negativity MAE, classification metrics, finite-shot confidence bounds and solver audit. |
| Dependencies | No pinned environment or structured unit tests evident in the inspected files; older code depends on NumPy, PyTorch, custom `ncon.py`, matplotlib. | Minimal new requirements, pytest regression suite, independent self-contained package. |

**Attribution:** Original repository cites Carrasquilla et al., *Nature Machine Intelligence* (2019) and has an MIT license (copyright notice: Hong-Ye Hu, 2021). This student baseline is new code inspired by its GRU design, not a claim that the original tensor-network POVM code already learns physical entanglement.
