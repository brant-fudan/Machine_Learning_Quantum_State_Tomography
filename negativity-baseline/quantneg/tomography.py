"""Physical-state, positivity-constrained least squares via projected FISTA.

Convex objective: 0.5 sum_i exposure_i*(Tr(rho P_i)-sample_mean_i)^2
subject to rho >= 0 and trace(rho)=1. The objective differs from
multinomial MLE; it is intentionally labeled *least squares*, NOT MLE.
"""
from __future__ import annotations
import numpy as np
from .physics import PAULIS, moments


def project_density(m: np.ndarray) -> np.ndarray:
    m=(m + m.swapaxes(-1,-2).conj())/2
    vals,vec=np.linalg.eigh(m)
    # Euclidean projection of each eigenvalue vector onto the probability simplex.
    order=np.sort(vals,axis=-1)[...,::-1]
    u=np.cumsum(order,axis=-1)-1
    ranks=np.arange(1,5)
    active=order-u/ranks>0
    k=active.sum(axis=-1)
    cut=np.take_along_axis(u,(k-1)[...,None],axis=-1)[...,0]/k
    vals=np.maximum(vals-cut[...,None],0)
    return (vec*vals[...,None,:])@vec.swapaxes(-1,-2).conj()


def physical_least_squares(means: np.ndarray, exposure: np.ndarray,
                           iterations: int=240) -> np.ndarray:
    if means.ndim==1:
        means,exposure=means[None], exposure[None]
    n=len(means)
    z=np.broadcast_to(np.eye(4, dtype=complex)/4,(n,4,4)).copy()
    old=z.copy()
    weights=exposure.astype(float)
    scale=np.maximum(1., weights.max(axis=-1,keepdims=True))
    w=weights/scale
    mom=PAULIS[1:]
    factor=1./4.  # Lipschitz constant of grad is at most 4*max(w)=4
    t=1.
    for _ in range(iterations):
        zmean=moments(z)
        grad=np.einsum('ni,ijk->njk', w*(zmean-means), mom)
        candidate=project_density(z - factor*grad)
        tnext=(1+np.sqrt(1+4*t*t))/2
        z=candidate+(t-1)/tnext*(candidate-old)
        old=candidate
        t=tnext
    return old
