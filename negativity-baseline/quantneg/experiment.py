"""Run exact-seeded experiments: synthetic finite-shot Pauli data (no GHZ data.txt)."""
from __future__ import annotations
import argparse, json, platform, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, torch
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             balanced_accuracy_score, brier_score_loss)
from .physics import negativity,generate_states,ambiguous_pair,moments,LABELS
from .measurements import (simulate_counts,collect_moments,features, FULL,PARTIAL,
                           probability_table)
from .tomography import physical_least_squares
from .certification import confidence_pair_witness,confidence_spectral_witness,sdp_lower_bound
from .models import fit_network,predict

ROOT=Path(__file__).resolve().parent.parent


def make_examples(n, family, rng, protocols, shots_list):
    rhos=generate_states(n,family,rng)
    targets=negativity(rhos)
    ms=np.zeros((n,15));ex=np.zeros((n,15),int)
    protocols_ix=rng.integers(0,len(protocols),n)
    shots_ix=rng.integers(0,len(shots_list),n)
    for i,p in enumerate(protocols):
        for j,shots in enumerate(shots_list):
            ix=np.flatnonzero((protocols_ix==i)&(shots_ix==j))
            if not len(ix):continue
            co=simulate_counts(rhos[ix],p,shots,rng)
            means,exposure=collect_moments(co)
            ms[ix]=means;ex[ix]=exposure
    return features(ms,ex),targets


def brier(y,prob):
    return float(brier_score_loss(y.astype(int),prob))


def metrics(y, pred, prob):
    yy=(y>1.e-8).astype(int)
    pred_label=prob>=.5
    return {'mae':float(np.mean(np.abs(y-pred))),
            'rmse':float(np.sqrt(np.mean((y-pred)**2))),
            'bias':float(np.mean(pred-y)),
            'accuracy':float(accuracy_score(yy,pred_label)),
            'balanced_accuracy':float(balanced_accuracy_score(yy,pred_label)),
            'f1':float(f1_score(yy,pred_label,zero_division=0)),
            'brier':brier(yy,prob),
            'auc':float(roc_auc_score(yy,prob)) if len(np.unique(yy))>1 else np.nan,
            'fpr_sep':float(np.mean(pred_label[yy==0])) if (yy==0).any() else np.nan,
            'recall_ent':float(np.mean(pred_label[yy==1])) if (yy==1).any() else np.nan,
            'n_states':len(y),'prevalence':float(np.mean(yy))}


def run(args):
    start=time.perf_counter()
    rng=np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.quick:
        n_train, n_val, n_test, epochs, n_sdp=2400,320,64,36,64
    else:
        n_train,n_val,n_test,epochs,n_sdp=args.train,args.validation,args.test,args.epochs,args.sdp_subset
    protocols=('partial','full');shot_list=(64,256,1024)
    print(f'train={n_train} val={n_val} per_test_cell={n_test}; epochs={epochs}; SDP/cell={n_sdp}',flush=True)
    x_train,y_train=make_examples(n_train,'iid',rng,protocols,shot_list)
    x_val,y_val=make_examples(n_val,'iid',rng,protocols,shot_list)
    model,hist=fit_network(x_train,y_train,kind=args.model,seed=args.seed,epochs=epochs)
    val_pred,val_score=predict(model,x_val)
    print('IID validation:',metrics(y_val,val_pred,val_score),flush=True)
    predictions=[]; summaries=[];certificate=[]
    for family in ('iid','ood'):
        # Identical underlying states across settings and budgets: paired comparison.
        state_rhos=generate_states(n_test,family,rng)
        target=negativity(state_rhos)
        for protocol in protocols:
            for shots in shot_list:
                c=simulate_counts(state_rhos,protocol,shots,rng)
                means,ex=collect_moments(c)
                inp=features(means,ex)
                nn_val,nn_prob=predict(model,inp)
                fit=physical_least_squares(means,ex,iterations=280)
                ls_val=negativity(fit)
                ls_prob=(ls_val>1.e-4).astype(float)
                analytic=np.array([confidence_pair_witness(means[i],ex[i],args.alpha)
                                   for i in range(n_test)])
                spectral=np.array([confidence_spectral_witness(means[i],ex[i],args.alpha,fit[i])
                                   for i in range(n_test)])
                sdp_values=np.full(n_test,np.nan)
                sdp_ok=np.zeros(n_test,bool)
                sdp_violation=np.full(n_test,np.nan)
                sdp_engine='not_run'
                # Deliberately small random audit; never silently count failed solvers.
                ix_sdp=rng.choice(n_test,size=min(n_sdp,n_test),replace=False)
                for i in ix_sdp:
                    result=sdp_lower_bound(means[i],ex[i],args.alpha,
                                           engine=args.sdp_engine,maxiter=155)
                    sdp_values[i]=result.value
                    sdp_ok[i]=result.success
                    sdp_violation[i]=result.violation
                    sdp_engine=result.engine
                context=dict(family=family,protocol=protocol,shots_per_setting=shots,
                             n_settings=9 if protocol=='full' else 2,
                             total_shots=(9 if protocol=='full' else 2)*shots)
                for i in range(n_test):
                    predictions.append({**context,'state_id':i,'true_negativity':float(target[i]),
                        'entangled':bool(target[i]>1e-8),'nn_negativity':float(nn_val[i]),
                        'nn_ent_probability':float(nn_prob[i]),'ls_negativity':float(ls_val[i]),
                        'sdp_numerical_lb':float(sdp_values[i]),
                        'sdp_success':bool(sdp_ok[i]),
                        'sdp_violation':float(sdp_violation[i]),
                        'certified_pair_lb':float(analytic[i]),
                        'certified_spectral_lb':float(spectral[i]),
                        'certified_combined_lb':float(max(analytic[i],spectral[i]))})
                for key,pred,score in (('direct_nn',nn_val,nn_prob),
                                       ('physical_LS',ls_val,ls_prob)):
                    summaries.append({**context,'method':key,**metrics(target,pred,score)})
                passed=sdp_ok & np.isfinite(sdp_values)
                audit={**context,'alpha':args.alpha,
                       'empirical_pair_lb_coverage':float(np.mean(analytic<=target+1.e-9)),
                       'pair_detection_rate':float(np.mean(analytic>1.e-6)),
                       'pair_mean_lb':float(np.mean(analytic)),
                       'spectral_detection_rate':float(np.mean(spectral>1.e-6)),
                       'spectral_mean_lb':float(np.mean(spectral)),
                       'combined_detection_rate':float(np.mean(np.maximum(analytic,spectral)>1.e-6)),
                       'combined_empirical_coverage':float(np.mean(np.maximum(analytic,spectral)<=target+1.e-9)),
                       'mean_true_negativity':float(np.mean(target)),
                       'sdp_attempted':len(ix_sdp),'sdp_success':int(np.sum(passed)),
                       'sdp_engine':sdp_engine,
                       'sdp_mean_lb_success':float(np.mean(sdp_values[passed])) if passed.any() else np.nan,
                       'sdp_detection_success':float(np.mean(sdp_values[passed]>1.e-5)) if passed.any() else np.nan,
                       'sdp_gt_true_success':int(np.sum(sdp_values[passed] > target[passed]+.0001)),
                       'sdp_max_psd_violation_success':float(np.max(sdp_violation[passed])) if passed.any() else np.nan}
                certificate.append(audit)
                print(f'{family:3s} {protocol:7s} n={shots:4d}: '
                      f'NN_MAE={summaries[-2]["mae"]:.4f}, LS_MAE={summaries[-1]["mae"]:.4f}, '
                      f'pair_detect={audit["pair_detection_rate"]:.3f}, '
                      f'SDP_ok={audit["sdp_success"]}/{audit["sdp_attempted"]}',flush=True)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(certificate).to_csv(out/'certificate_summary.csv',index=False)
    np.savetxt(out/'training_loss.csv',np.asarray(hist),delimiter=',',header='loss',comments='')
    torch.save(model.state_dict(),out/f'model_{args.model}_seed{args.seed}.pt')
    # Exact indistinguishable-state example: predictor sees identical features.
    ent,sep=ambiguous_pair(.45)
    both=np.stack((ent,sep));truth=negativity(both)
    # Use *exact population* partial means, with arbitrarily high exposure to
    # demonstrate structural ambiguity even in the infinite-shot limit.
    real_mom=moments(both)
    example_counts=simulate_counts(both,'partial',1,np.random.default_rng(7))
    _,example_exp=collect_moments(example_counts)
    exact_means=np.zeros((2,15));exact_means[example_exp>0]=real_mom[example_exp>0]
    fake_ex=np.where(example_exp>0,100000000,0)
    predicted,_=predict(model,features(exact_means,fake_ex))
    ls=negativity(physical_least_squares(exact_means,fake_ex,iterations=300))
    sdp=[sdp_lower_bound(exact_means[i],fake_ex[i],engine=args.sdp_engine,
                          exact=True,maxiter=190) for i in range(2)]
    ambiguous=dict(p=.45,truth_entangled=float(truth[0]),truth_separable=float(truth[1]),
         same_partial_probabilities=bool(np.allclose(probability_table(ent)[[0,8]],
                                                     probability_table(sep)[[0,8]])),
         full_YY_difference=float(abs(real_mom[0,LABELS.index('YY')-1]-real_mom[1,LABELS.index('YY')-1])),
         nn_outputs=predicted.tolist(),ls_outputs=ls.tolist(),
         sdp_exact_outputs=[x.value for x in sdp],sdp_success=[x.success for x in sdp],
         unavoidable_max_error_half_gap=float((truth[0]-truth[1])/2))
    (out/'ambiguity.json').write_text(json.dumps(ambiguous,indent=2)+'\n')
    runtime={'seed':args.seed,'mode':'quick' if args.quick else 'standard',
             'model':args.model,'train_states':n_train,'validation_states':n_val,
             'test_states_per_cell':n_test,'epochs':epochs,'alpha':args.alpha,
             'sdp_attempts_per_cell':n_sdp,'sdp_engine':args.sdp_engine,
             'python':platform.python_version(),'numpy':np.__version__,
             'scipy':scipy.__version__,'torch':torch.__version__,
             'elapsed_seconds':round(time.perf_counter()-start,1),
             'notes':'Simulated trusted Pauli projectors; shot allocation is PER setting; '
                     '9n vs 2n overall shots. SLSQP values are numerical, not dual-certified.'}
    (out/'run_metadata.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print('AMBIGUITY:',json.dumps(ambiguous,indent=2),flush=True)
    from .plotting import make_figures
    make_figures(out,ROOT/'figures')
    return pd.DataFrame(summaries),pd.DataFrame(certificate)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true',help='Run CPU-sized reproducible baseline')
    p.add_argument('--model',choices=['mlp','gru'],default='mlp')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--train',type=int,default=8000)
    p.add_argument('--validation',type=int,default=1000)
    p.add_argument('--test',type=int,default=160)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--sdp-subset',type=int,default=24)
    p.add_argument('--sdp-engine',choices=['auto','slsqp','cvxpy'],default='auto')
    return p.parse_args()

if __name__=='__main__':
    run(parse_args())
