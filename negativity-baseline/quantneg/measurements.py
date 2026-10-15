"""Shot-level local Pauli (X,Y,Z) x (X,Y,Z) projective measurements."""
from __future__ import annotations
import numpy as np
from .physics import PAULIS, LABELS, I, SINGLE

SETTINGS = [(a,b) for a in 'XYZ' for b in 'XYZ']
FULL = tuple(range(9))
PARTIAL = (0, 8)  # XX and ZZ only
OUTCOMES = ((1,1),(1,-1),(-1,1),(-1,-1))
POVMS = np.stack([np.stack([np.kron((I+a*SINGLE[s[0]])/2,(I+b*SINGLE[s[1]])/2)
                                     for a,b in OUTCOMES]) for s in SETTINGS])


def simulate_counts(rhos: np.ndarray, protocol: str, shots: int,
                    rng: np.random.Generator) -> np.ndarray:
    indices = FULL if protocol == 'full' else PARTIAL if protocol == 'partial' else None
    if indices is None or shots < 1:
        raise ValueError('protocol=full/partial, shots >= 1 required')
    # [state, setting, outcome]
    probs = np.einsum('nij,soji->nso', rhos, POVMS).real
    probs = np.clip(probs, 0, None)
    probs /= probs.sum(axis=-1, keepdims=True)
    counts = np.zeros((len(rhos), 9, 4), dtype=np.int32)
    for s in indices:
        counts[:,s] = rng.multinomial(shots, probs[:,s])
    return counts


def probability_table(rho: np.ndarray) -> np.ndarray:
    return np.einsum('ij,soji->so', rho, POVMS).real


def collect_moments(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (15 observed means, 15 effective independent shots per mean).

    Single-qubit marginals are pooled across different, disjoint setting batches.
    Cross-moment estimates reuse their shots; simultaneous Bonferroni coverage
    DOES NOT require independence across different observables.
    """
    counts = np.asarray(counts)
    if counts.ndim == 2:
        counts = counts[None]
    total = np.zeros((len(counts), 16), float)
    exposure = np.zeros((len(counts), 16), int)
    a_sign = np.array([1,1,-1,-1]); b_sign=np.array([1,-1,1,-1])
    for s,(a,b) in enumerate(SETTINGS):
        row = counts[:,s]
        n = row.sum(axis=-1)
        for name, weights in ((a+'I', a_sign), ('I'+b, b_sign), (a+b,a_sign*b_sign)):
            idx=LABELS.index(name)
            total[:,idx]+=row@weights
            exposure[:,idx]+=n
    observed=exposure[:,1:] > 0
    estimate=np.divide(total[:,1:], exposure[:,1:],
                       out=np.zeros_like(total[:,1:]), where=observed)
    return estimate, exposure[:,1:]


def features(means: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    observed=(exposure>0).astype(float)
    # A dimensionless shot-noise descriptor; zero means this observable is missing.
    noise=np.divide(1., np.sqrt(exposure.astype(float)),
                    out=np.zeros_like(means), where=exposure>0)
    return np.concatenate((means, observed, noise), axis=-1).astype('float32')


def hoeffding_intervals(means: np.ndarray, exposure: np.ndarray, alpha: float=.05):
    """Simultaneous fixed-state confidence box (>=1-alpha), assuming iid shots.

    For ±1 samples: P(|mean - true|>eps)<=2 exp(-n*eps^2/2).
    """
    mask=exposure>0
    m=int(mask.sum())
    if not 0 < alpha < 1:
        raise ValueError('alpha should be in (0,1)')
    eps=np.zeros_like(means, dtype=float)
    if m:
        eps[mask]=np.sqrt(2*np.log(2*m/alpha)/exposure[mask])
    return np.maximum(-1., means-eps), np.minimum(1., means+eps), mask
