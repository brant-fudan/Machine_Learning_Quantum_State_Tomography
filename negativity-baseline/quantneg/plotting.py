"""Plain matplotlib outputs; chart generation is deterministic from saved CSVs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def make_figures(result_folder:Path, figure_folder:Path):
    figure_folder.mkdir(exist_ok=True)
    stats=pd.read_csv(result_folder/'metrics.csv')
    cert=pd.read_csv(result_folder/'certificate_summary.csv')
    states=pd.read_csv(result_folder/'predictions.csv')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for family in ('iid','ood'):
        for protocol in ('partial','full'):
            for method in ('direct_nn','physical_LS'):
                d=stats[(stats.family==family)&(stats.protocol==protocol)&(stats.method==method)]
                ax.plot(d.shots_per_setting,d.mae,marker='o',
                        label=f'{family} / {protocol} / {method}')
    ax.set_xscale('log',base=2);ax.set_xticks([64,256,1024]);ax.set_xticklabels(['64','256','1024'])
    ax.set(xlabel='Shots per local measurement setting',ylabel='MAE in Negativity',
           title='Point prediction accuracy (not a certification measure)')
    ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(figure_folder/'01_mae_vs_shots.png',dpi=180);fig.savefig(figure_folder/'01_mae_vs_shots.svg');plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,5))
    for protocol in ('partial','full'):
        d=states[(states.family=='ood')&(states.protocol==protocol)&(states.shots_per_setting==1024)]
        ax.scatter(d.true_negativity,d.nn_negativity,alpha=.65,label=f'Neural {protocol}')
    ax.plot([0,.5],[0,.5],linestyle='--',label='perfect')
    ax.set(xlabel='True negativity',ylabel='Predicted negativity',
           title='Distribution shift: training IID, testing OOD')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(figure_folder/'02_shift_predictions.png',dpi=180);fig.savefig(figure_folder/'02_shift_predictions.svg');plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.5))
    for family in ('iid','ood'):
        for protocol in ('partial','full'):
            d=cert[(cert.family==family)&(cert.protocol==protocol)]
            ax.plot(d.shots_per_setting,d.combined_detection_rate,marker='s',
                    label=f'{family}/{protocol}')
    ax.set_xscale('log',base=2);ax.set_xticks([64,256,1024]);ax.set_xticklabels(['64','256','1024'])
    ax.set(xlabel='Shots per setting',ylabel='Fraction with positive valid witness bound',
           ylim=(-.02,1.02),title='Conservative 95% finite-shot certificate (Hoeffding)')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(figure_folder/'03_certificate_detection.png',dpi=180);fig.savefig(figure_folder/'03_certificate_detection.svg');plt.close(fig)

    amb=json.loads((result_folder/'ambiguity.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4.3))
    labels=['Entangled / same XX,ZZ','Separable / same XX,ZZ']
    xx=np.arange(2);width=.24
    ax.bar(xx-width,[amb['truth_entangled'],amb['truth_separable']],width,label='True Negativity')
    ax.bar(xx,amb['nn_outputs'],width,label='NN prediction (same input)')
    ax.bar(xx+width,amb['sdp_exact_outputs'],width,label='SDP exact-measurement lower limit')
    ax.set_xticks(xx,labels);ax.set(ylabel='Negativity',title='Information-theoretic non-identifiability')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(figure_folder/'04_indistinguishable_pair.png',dpi=180);fig.savefig(figure_folder/'04_indistinguishable_pair.svg');plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.3))
    for family in ('iid','ood'):
        for protocol in ('partial','full'):
            d=stats[(stats.family==family)&(stats.protocol==protocol)&(stats.method=='direct_nn')]
            ax.plot(d.shots_per_setting,d.fpr_sep,marker='o',label=f'{family}/{protocol}')
    ax.set_xscale('log',base=2);ax.set_xticks([64,256,1024]);ax.set_xticklabels(['64','256','1024'])
    ax.set(xlabel='Shots per setting',ylabel='NN false-positive fraction on separable states',
           title='A high classification score is not an entanglement certificate')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(figure_folder/'05_nn_false_positives.png',dpi=180);fig.savefig(figure_folder/'05_nn_false_positives.svg');plt.close(fig)
