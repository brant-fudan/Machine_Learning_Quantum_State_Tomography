import numpy as np
import torch
from quantneg.physics import (BELL_RHO,I, PAULIS, LABELS, negativity,partial_transpose,
                              ambiguous_pair,generate_states,moments,from_moments)
from quantneg.measurements import (FULL,PARTIAL,collect_moments,simulate_counts,
                                   probability_table,hoeffding_intervals,features)
from quantneg.tomography import physical_least_squares
from quantneg.certification import confidence_pair_witness,confidence_spectral_witness,sdp_lower_bound
from quantneg.models import PropertyNet


def test_negativity_known():
    assert abs(negativity(BELL_RHO)-.5)<1e-11
    assert negativity(np.eye(4)/4)<1e-11
    for p in (.1,.3,.4,.7,1.):
        rho=p*BELL_RHO+(1-p)*np.eye(4)/4
        assert abs(negativity(rho)-max(0.,(3*p-1)/4))<1e-10


def test_partial_transpose_involution_and_basis():
    x=np.random.default_rng(1);rho=generate_states(12,'iid',x)
    assert np.allclose(partial_transpose(partial_transpose(rho)),rho)
    assert np.allclose(from_moments(moments(rho)),rho)


def test_ambiguous_pair_is_same_measured_data():
    ent,sep=ambiguous_pair(.45)
    assert np.linalg.eigvalsh(sep).min()>0
    assert abs(negativity(ent)-.0875)<1e-12
    assert negativity(sep)<1e-12
    assert np.allclose(probability_table(ent)[list(PARTIAL)],
                       probability_table(sep)[list(PARTIAL)])
    assert not np.allclose(probability_table(ent)[list(FULL)],
                           probability_table(sep)[list(FULL)])


def test_counts_features_tomography():
    rng=np.random.default_rng(18)
    rho=generate_states(9,'iid',rng)
    counts=simulate_counts(rho,'partial',64,rng)
    means,exp=collect_moments(counts)
    assert counts.shape==(9,9,4)
    assert (counts.sum(axis=(1,2))==128).all()
    assert (exp>0).sum(axis=1).tolist()==[6]*9
    ff=features(means,exp)
    assert ff.shape==(9,45)
    rec=physical_least_squares(means,exp,iterations=200)
    assert np.all(np.linalg.eigvalsh(rec)>=-1e-8)
    assert np.allclose(np.trace(rec,axis1=1,axis2=2),1.)


def test_finite_shot_joint_box_and_certificate():
    ent,sep=ambiguous_pair(.45)
    rng=np.random.default_rng(55)
    c=simulate_counts(np.stack([ent,sep]),'full',6000,rng)
    t,n=collect_moments(c)
    tr=moments(np.stack([ent,sep]))
    for k in range(2):
        l,h,seen=hoeffding_intervals(t[k],n[k],.05)
        assert (l[seen]-.00001<=tr[k,seen]).all()
        assert (h[seen]+.00001>=tr[k,seen]).all()
        assert confidence_pair_witness(t[k],n[k])<=negativity([ent,sep][k])+1e-6


def test_pair_bound_never_above_true_for_random_states():
    rng=np.random.default_rng(77)
    for family in ('iid','ood'):
        r=generate_states(150,family,rng)
        c=simulate_counts(r,'full',10000,rng)
        t,n=collect_moments(c)
        # Population moments with the measured n_eff: no sampling fluctuation.
        for ti,ni,truth in zip(moments(r),n,negativity(r)):
            assert confidence_pair_witness(ti,ni)<=truth+1e-8


def test_nn_model_shapes():
    x=torch.zeros(5,45)
    for kind in ('mlp','gru'):
        v,p=PropertyNet(kind)(x)
        assert v.shape==(5,) and p.shape==(5,)
        assert (v>=0).all() and (v<=.5).all()


def test_sdp_exact_pair_and_full():
    ent,sep=ambiguous_pair(.45)
    for rho,protocol,target in [(ent,'partial',0.),(sep,'partial',0.),(ent,'full',.0875)]:
        c=simulate_counts(np.array([rho]),protocol,50,np.random.default_rng(2))
        t,n=collect_moments(c)
        tv=moments(rho)
        out=sdp_lower_bound(tv,n[0],exact=True,engine='slsqp',maxiter=190)
        assert out.success, out
        assert abs(out.value-target)<.002, out


def test_finite_shot_dual_witness_and_same_data_adaptation():
    rng=np.random.default_rng(155)
    r=generate_states(20,'ood',rng)
    for protocol in ('full','partial'):
        c=simulate_counts(r,protocol,2500,rng)
        tt,nn=collect_moments(c)
        for t,n,rho in zip(tt,nn,r):
            bound=confidence_spectral_witness(t,n)
            assert bound>=0
            assert bound<=negativity(rho)+1e-8
    c=simulate_counts(np.array([BELL_RHO]),'full',30000,rng)
    t,n=collect_moments(c)
    assert confidence_spectral_witness(t[0],n[0])>.35
