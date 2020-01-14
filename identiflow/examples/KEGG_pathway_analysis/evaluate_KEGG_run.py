#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Torsten Gross
"""

import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

save_plots = False

#%%load data
analysis_folder=os.path.dirname(__file__)
plot_folder=os.path.join(analysis_folder, 'plots/')
KEGG_results_folder=os.path.join(analysis_folder, 'results/KEGG/')

pathways = pd.read_pickle(os.path.join(analysis_folder,'KEGG_pathways.pkl'))

KEGG_results = []
for filename in os.listdir(KEGG_results_folder):
    if not filename[-3:] == 'pkl':
        continue
    with open(os.path.join(KEGG_results_folder,filename), 'rb') as f:
        KEGG_results.append( pickle.load(f) )

KEGG_results = pd.DataFrame(KEGG_results)

KEGG_results = KEGG_results.join(pathways[['N','tc_score']],on='KEGG_id')
KEGG_results = KEGG_results.rename(columns={'ident_AUC':'Identifiability AUC'})

#%%compute fraction of required perturbations

KEGG_results['n_required_perts'] = KEGG_results['best_sum_dis'].apply(len)
n_required_perts = KEGG_results[['KEGG_id','strategy','n_required_perts']].groupby(
        ['KEGG_id','strategy']).mean().reset_index()

random_n_required_perts = n_required_perts[n_required_perts['strategy']=='random'].drop(columns='strategy').set_index('KEGG_id')
fraction_required_perts = n_required_perts[n_required_perts['strategy']!='random'].join(random_n_required_perts, on='KEGG_id',rsuffix='_random')
fraction_required_perts['Fraction of required perturbations'] = fraction_required_perts['n_required_perts'] / fraction_required_perts['n_required_perts_random']
fraction_required_perts = fraction_required_perts.drop(columns=['n_required_perts','n_required_perts_random'])

fraction_required_perts = fraction_required_perts.join(pathways['tc_score'], on = 'KEGG_id')
pearson_r_req_frac   = fraction_required_perts.groupby('strategy')[ 'Fraction of required perturbations', 'tc_score'].apply(
        lambda col: scipy.stats.pearsonr(col[ 'Fraction of required perturbations'],col['tc_score']))
spearman_r_req_frac = fraction_required_perts.groupby('strategy')[ 'Fraction of required perturbations', 'tc_score'].apply(
        lambda col: scipy.stats.spearmanr(col[ 'Fraction of required perturbations'],col['tc_score']))

#%%compute Delta identifiability AUC
mean_AUC = KEGG_results[['KEGG_id','strategy','Identifiability AUC']].groupby(
        ['KEGG_id','strategy']).mean().reset_index(level='strategy')
delta_AUC = mean_AUC[mean_AUC.strategy!='random'].join( 
        mean_AUC.loc[mean_AUC.strategy=='random','Identifiability AUC'],
                    rsuffix='_random')
delta_AUC['Delta Identifiability AUC'] = delta_AUC['Identifiability AUC'] - delta_AUC['Identifiability AUC_random']
delta_AUC = delta_AUC.drop(columns=['Identifiability AUC','Identifiability AUC_random'])
delta_AUC = delta_AUC.join(pathways['tc_score'])

pearson_r   = delta_AUC.groupby('strategy')['Delta Identifiability AUC', 'tc_score'].apply(
        lambda col: scipy.stats.pearsonr(col['Delta Identifiability AUC'],col['tc_score']))
spearman_r = delta_AUC.groupby('strategy')['Delta Identifiability AUC', 'tc_score'].apply(
        lambda col: scipy.stats.spearmanr(col['Delta Identifiability AUC'],col['tc_score']))


#%%for multi-target strategy compute the distribution of the number of perturbation targets

path_npert = []
for i,r in KEGG_results.loc[KEGG_results.strategy=='multi_target',['KEGG_id' , 'best_pert_seqs']].iterrows():
    for pert_seq  in r['best_pert_seqs']:
        for pert_combi in pert_seq:
            path_npert.append( [r['KEGG_id'] , len(pert_combi)])
            
path_npert = pd.DataFrame(path_npert,columns=['KEGG_id','Number of targets'])

path_npert = path_npert.join(pathways['N'],on='KEGG_id')

def cut_size_groups(r):
    if r['N']<25:
        size_group = '< 25'
    elif (r['N']>=25) and (r['N']<=75):
        size_group = '25 - 75'
    elif r['N']>75:
        size_group = '> 75'
    return size_group

path_npert['size_group'] = path_npert.apply(cut_size_groups,axis=1)

path_npert = path_npert.groupby(['size_group','Number of targets'])[['KEGG_id']].count().rename(
        columns={'KEGG_id':'n_experiments'})
path_npert = path_npert.join(path_npert.groupby(level = 'size_group')[['n_experiments']].sum().rename(
        columns={'n_experiments':'total_n_experiments'}),on='size_group').reset_index()

path_npert['Fraction of experiments'] = path_npert['n_experiments']/path_npert['total_n_experiments']


#%%plot results

strategy_order = ['random','naive','greedy','multi_target']
palette = sns.color_palette(n_colors=5)
palette = [palette[i] for i in [0,1,2,4]]
#%%

fig,ax = plt.subplots(figsize=[3.25,2])
sns.boxplot(x='strategy',y='Identifiability AUC',data=KEGG_results,order = strategy_order, ax=ax,
            palette=palette,width=.55)
sns.despine()
plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join(plot_folder, 'identifiabilty_AUC.pdf'))
plt.show()



#%%
fig,ax = plt.subplots(figsize=[2.85,2])
sns.boxplot(x='strategy',y= 'Fraction of required perturbations',data=fraction_required_perts,
            order = strategy_order[1:], ax=ax,
            palette=palette[1:],width=.5)
sns.despine()
ax.set_ylabel('Fraction of required\nperturbations')
#ax.set_ylim(top=1)
plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join(plot_folder, 'fraction_required_perturbation.pdf'))
plt.show()


#%%
delta_order=['delta_naive_random','delta_greedy_random','delta_opti_random']
sns.lmplot(x='tc_score',y='Delta Identifiability AUC',col='strategy',hue='strategy',
                data=delta_AUC,
                height=2.,aspect=.9,
                col_order=strategy_order[1:],hue_order=strategy_order[1:],
                scatter_kws={'s':5,'linewidth':.25},
                line_kws={'linewidth':1.},
                sharey=True,
                palette=palette[1:],)
sns.despine()
#ax.set_ylabel('Delta Identifiability AUC')
plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join(plot_folder, 'Delta_ident_AUC_corr.pdf'))
plt.show()


#%%

delta_order=['delta_naive_random','delta_greedy_random','delta_opti_random']
a = sns.lmplot(x='tc_score',y='Fraction of required perturbations',col='strategy',hue='strategy',
                data=fraction_required_perts,
                height=2.,aspect=.9,
                col_order=strategy_order[1:],hue_order=strategy_order[1:],
                scatter_kws={'s':5,'linewidth':.25},
                line_kws={'linewidth':1.},
                sharey=True,
                palette=palette[1:],)
sns.despine()
a.set_ylabels('Fraction required\nperturbations')
plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join(plot_folder, 'Frac_req_pert_corr.pdf'))
plt.show()

#%%
sns.catplot(y='Fraction of experiments',x='Number of targets',col='size_group',col_order=['< 25','25 - 75','> 75'],
            data=path_npert,kind='bar',
            height=2.,aspect=1.1,sharex=False,
            color='grey',
            ci='sd',
            )

plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join(plot_folder, 'n_pert_targets.pdf'))
plt.show()