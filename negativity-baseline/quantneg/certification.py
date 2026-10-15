"""Partial information/finite-shot negativity lower bounds.

SDP: minimize Tr(Q) subject to rho>=0, Tr(rho)=1, Q>=0,
     rho^(Gamma_B)+Q>=0 and marginal Pauli moment confidence box.
All programs are *trusted-device* (calibrated Pauli observables), NOT DI.
Numerical SDP optima are NOT themselves rigorously certified dual bounds.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, Bounds
from .physics import LABELS, PAULIS, P0, PT_SIGNS, from_moments, moments, partial_transpose
from .measurements import hoeffding_intervals


def confidence_pair_witness(means: np.ndarray, exposure: np.ndarray, alpha: float=.05) -> float:
    """Rigorous 1-alpha lower confidence bound, assuming trusted iid Pauli shots.

    For orthogonal local axes (i != k, j != l):
      N(rho) >= (|<sigma_i sigma_j>|+|<sigma_k sigma_l>|-1)/2.
    Multiple pair choices are safe: every observed moment is in its
    Bonferroni/Hoeffding interval on ONE simultaneous event.
    """
    mask=exposure>0
    m=int(mask.sum()); result=0.
    if not m: return result
    radius=np.zeros(15)
    radius[mask]=np.sqrt(2*np.log(2*m/alpha)/exposure[mask])
    for a in 'XYZ':
        for b in 'XYZ':
            ii=LABELS.index(a+b)-1
            if not mask[ii]: continue
            for c in 'XYZ':
                if c==a: continue
                for d in 'XYZ':
                    if d==b: continue
                    jj=LABELS.index(c+d)-1
                    if not mask[jj]: continue
                    result=max(result,(abs(means[ii])-radius[ii]+abs(means[jj])-radius[jj]-1)/2)
    return max(0.,float(result))

def confidence_spectral_witness(means: np.ndarray, exposure: np.ndarray,
                                alpha: float=.05, reference_rho=None) -> float:
    """Data-adaptive, *mathematically valid* lower confidence bound.

    For any 0 <= Q <= I, N(rho) >= -Tr(Q rho^{Gamma_B}).
    Select Q as negative spectral projector of a physical LS fit, then
    minimize its affine witness over the simultaneous Hoeffding box.
    For UNMEASURED moments, use the unconditional physical range [-1,1].
    Selecting Q using the same data is safe: simultaneous box coverage
    validates the linear inequality for every (data-dependent) Q at once.
    """
    from .physics import partial_transpose
    if reference_rho is None:
        from .tomography import physical_least_squares
        reference_rho=physical_least_squares(means,exposure,iterations=150)[0]
    e,v=np.linalg.eigh(partial_transpose(reference_rho))
    choose=e < -1.e-8
    if not choose.any():
        return 0.
    V=v[:,choose]
    Q=V@V.conj().T
    # Eigenprojector ensures 0<=Q<=I (up to harmless floating error).
    low,high,mask=hoeffding_intervals(means,exposure,alpha)
    low=np.where(mask,low,-1.)
    high=np.where(mask,high,1.)
    # Coefficients in Tr(Q^Gamma rho) expansion.
    coeff=np.einsum('ij,kji->k',partial_transpose(Q),PAULIS).real/4
    upper=coeff[0] + np.maximum(coeff[1:]*low,coeff[1:]*high).sum()
    return float(max(0.,-upper))


@dataclass
class SDPResult:
    value: float
    success: bool
    engine: str
    violation: float
    iterations: int
    status: str


def _spectral_constraint(x):
    t=x[:15]; q=x[15:]
    rho=from_moments(t)
    pt=from_moments(t*PT_SIGNS[1:])
    Q=np.einsum('k,kij->ij',q,PAULIS)/4
    eigs=[]; grad=[]
    for k,M in enumerate((rho,Q,pt+Q)):
        e,v=np.linalg.eigh((M+M.conj().T)/2)
        eigs.append(e)
        basis_t=(PAULIS[1:]*(PT_SIGNS[1:,None,None] if k==2 else 1)) if k!=1 else None
        der=np.zeros((4,31))
        if basis_t is not None:
            der[:,:15]=np.einsum('aj,mab,bj->jm',v.conj(),basis_t,v).real/4
        if k!=0:
            der[:,15:]=np.einsum('aj,mab,bj->jm',v.conj(),PAULIS,v).real/4
        grad.append(der)
    return np.concatenate(eigs),np.concatenate(grad,axis=0)


def sdp_lower_bound(means, exposure, alpha=.05, *, engine='auto', maxiter=160,
                    exact=False) -> SDPResult:
    """Solve underlying SDP numerically. 'exact' means exact input means, not proof.

    If CVXPY unavailable, SciPy SLSQP optimizes the *same convex SDP*
    represented through minimum eigenvalue constraints. This is a numerical
    fallback (not a conic solver or dual certificate), for 4x4 only.
    """
    means=np.asarray(means,float); exposure=np.asarray(exposure,int)
    if exact:
        lo=hi=means.copy(); seen=exposure>0
    else:
        lo,hi,seen=hoeffding_intervals(means,exposure,alpha)
    if engine in ('auto','cvxpy'):
        try:
            import cvxpy as cp
        except ImportError:
            if engine=='cvxpy':
                raise
        else:
            t=cp.Variable(15)
            Q=cp.Variable((4,4),hermitian=True)
            rho=(P0+sum(t[i]*PAULIS[i+1] for i in range(15)))/4
            pt=(P0+sum(t[i]*PT_SIGNS[i+1]*PAULIS[i+1] for i in range(15)))/4
            constraints=[rho>>0,Q>>0,pt+Q>>0]
            if seen.any(): constraints.extend((t[seen]>=lo[seen],t[seen]<=hi[seen]))
            prob=cp.Problem(cp.Minimize(cp.real(cp.trace(Q))),constraints)
            available=cp.installed_solvers()
            solver=next((s for s in ('CLARABEL','SCS','CVXOPT') if s in available),None)
            if solver is None:
                raise RuntimeError('No supported SDP conic solver available')
            try:
                val=prob.solve(solver=solver,verbose=False)
                if prob.status in ('optimal','optimal_inaccurate') and val is not None:
                    tv=np.asarray(t.value).reshape(-1)
                    rho_v=from_moments(tv)
                    qv=np.asarray(Q.value)
                    violation=min(np.linalg.eigvalsh(rho_v).min(),np.linalg.eigvalsh(qv).min(),
                                  np.linalg.eigvalsh(partial_transpose(rho_v)+qv).min(),
                                  float((tv[seen]-lo[seen]).min()) if seen.any() else 0.,
                                  float((hi[seen]-tv[seen]).min()) if seen.any() else 0.)
                    return SDPResult(float(val), prob.status=='optimal', 'cvxpy-'+solver,
                                     float(max(0.,-violation)), -1, prob.status)
            except Exception as e:
                if engine=='cvxpy': raise RuntimeError('SDP solver failed') from e
            if engine=='cvxpy':
                return SDPResult(float('nan'),False,'cvxpy-'+solver,float('nan'),-1,prob.status)
    if engine not in ('auto','slsqp','cvxpy'):
        raise ValueError(engine)
    # Startup from PSD LS estimate (unmeasured moments set to zero).
    from .tomography import physical_least_squares
    rho0=physical_least_squares(means,exposure,iterations=70)[0]
    t0=moments(rho0)
    pt=partial_transpose(rho0)
    e,v=np.linalg.eigh(pt)
    Q=(v*np.maximum(-e,0)[None,:])@v.conj().T + .005*np.eye(4)
    q0=np.einsum('ij,kji->k',Q,PAULIS).real
    x0=np.concatenate((t0,q0))
    bounds=Bounds(np.concatenate((np.where(seen,lo,-1.),np.full(16,-np.inf))),
                  np.concatenate((np.where(seen,hi,1.),np.full(16,np.inf))))
    cache={}
    def cached(x):
        if 'x' not in cache or not np.array_equal(x,cache['x']):
            e,j=_spectral_constraint(x);cache.update(x=x.copy(),e=e,j=j)
        return cache['e'],cache['j']
    g=np.zeros(31);g[15]=1.
    res=minimize(lambda x:x[15],x0,method='SLSQP',jac=lambda x:g,
                 bounds=bounds,
                 constraints=[{'type':'ineq','fun':lambda x:cached(x)[0],
                               'jac':lambda x:cached(x)[1]}],
                 options={'maxiter':maxiter,'ftol':1.e-8,'disp':False})
    ev=cached(res.x)[0]
    boxlo=np.min(res.x[:15][seen]-lo[seen]) if seen.any() else 0
    boxhi=np.min(hi[seen]-res.x[:15][seen]) if seen.any() else 0
    violation=max(0.,-float(min(ev.min(),boxlo,boxhi)))
    success=bool(res.success and violation<2.e-5 and res.fun>-2.e-5)
    return SDPResult(float(max(0.,res.fun)),success,'scipy-SLSQP',
                     violation,int(res.nit),str(res.message))
