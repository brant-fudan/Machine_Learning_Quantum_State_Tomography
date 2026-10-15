"""Two-qubit states and entanglement; fixed computational basis |00>,|01>,|10>,|11>."""
from __future__ import annotations
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)
SINGLE = {"I": I, "X": X, "Y": Y, "Z": Z}
LABELS = [a + b for a in "IXYZ" for b in "IXYZ"]
PAULIS = np.stack([np.kron(SINGLE[a], SINGLE[b]) for a, b in [tuple(s) for s in LABELS]])
NONID = LABELS[1:]
P0 = PAULIS[0]
PT_SIGNS = np.array([-1. if key[1] == 'Y' else 1. for key in LABELS])
BELL = np.array([1., 0., 0., 1.], complex) / np.sqrt(2)
BELL_RHO = np.outer(BELL, BELL.conj())


def partial_transpose(rho: np.ndarray) -> np.ndarray:
    """Transpose the SECOND qubit. Supports (...,4,4) arrays."""
    a = np.asarray(rho)
    return a.reshape(a.shape[:-2] + (2, 2, 2, 2)).swapaxes(-3, -1).reshape(a.shape)


def negativity(rho: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(partial_transpose(rho))
    return np.maximum(0., -np.minimum(vals, 0.).sum(axis=-1))


def moments(rho: np.ndarray) -> np.ndarray:
    return np.einsum('...ij,kji->...k', rho, PAULIS).real[..., 1:]


def from_moments(t: np.ndarray) -> np.ndarray:
    return (P0 + np.einsum('...k,kij->...ij', t, PAULIS[1:])) / 4


def haar_qubits(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def haar_unitary(rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    return q * np.exp(-1j * np.angle(np.diag(r)))[None, :]


def generate_states(n: int, family: str, rng: np.random.Generator) -> np.ndarray:
    if family == 'iid':
        # Mixed distribution: half Hilbert--Schmidt Wishart, half explicitly separable.
        modes = rng.integers(0, 2, size=n)
    elif family == 'ood':
        # Deliberately shifted: isotropic Bell mixtures and locally rotated pure states.
        modes = rng.integers(2, 4, size=n)
    else:
        raise ValueError(family)
    out = np.zeros((n, 4, 4), complex)
    for idx, mode in enumerate(modes):
        if mode == 0:
            a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            m = a @ a.conj().T
            out[idx] = m / np.trace(m).real
        elif mode == 1:
            x, y = haar_qubits(rng, 3), haar_qubits(rng, 3)
            v = np.einsum('ki,kj->kij', x, y).reshape(3, 4)
            w = rng.dirichlet(np.ones(3))
            out[idx] = np.einsum('k,ki,kj->ij', w, v, v.conj())
        elif mode == 2:
            p = rng.uniform(0, 1)
            out[idx] = p * BELL_RHO + (1-p) * np.eye(4) / 4
        else:
            theta = rng.uniform(.03, .785)
            v = np.array([np.cos(theta), 0, 0, np.sin(theta)], complex)
            v = np.kron(haar_unitary(rng), haar_unitary(rng)) @ v
            white = rng.uniform(0, .08)
            out[idx] = (1-white)*np.outer(v, v.conj())+white*np.eye(4)/4
    return out


def ambiguous_pair(p: float = .45) -> tuple[np.ndarray, np.ndarray]:
    """Same XX, ZZ joint outcome laws, different negativity (for 1/3 < p <= 1/2)."""
    entangled = p * BELL_RHO + (1-p) * np.eye(4) / 4
    separable = (P0 + p * PAULIS[LABELS.index('XX')] + p * PAULIS[LABELS.index('ZZ')]) / 4
    return entangled, separable
