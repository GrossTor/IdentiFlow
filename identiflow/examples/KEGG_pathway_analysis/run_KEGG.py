#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Torsten Gross
Note that depending on your architecture, running all networks might take hours
to days. Consider reducing max_net_size to 20 or below for a quick test run.
"""

nbr_used_cores = 7
max_net_size = np.inf

from multiprocessing import Pool
import os
import networkx as nx
import pickle
import pandas as pd
import numpy as np

from experimental_design import optimize_experimental_design

analysis_folder=os.path.dirname(__file__)
results_folder=os.path.join(analysis_folder, 'results/')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

with open (os.path.join(analysis_folder,'KEGG_pathways_comprehensive.pkl'), 'rb') as fp:
    pathways = pd.DataFrame(pickle.load(fp))

pathways['N']=pathways['nodes'].apply(len)
cropped_pathways=[]
for i,pathway in pathways.iterrows():
    #I only want to look at connected components
    largest_cc = max(nx.connected_components(nx.Graph(pathway.net)), key=len)
    N = len(largest_cc)
    if N<5: continue
    net = pathway.net.subgraph( largest_cc ).copy()
    tc_score = 1 - nx.number_of_edges(nx.algorithms.transitive_closure(net))/(N*(N-1))

    cropped_pathways.append({
            'KEGG_id':pathway['KEGG_id'],
            'nodes' : largest_cc,
            'net' : net,
            'N': N,
            'tc_score' : tc_score,
            })


pathways = pd.DataFrame(cropped_pathways)
pathways = pathways[['KEGG_id','N', 'net', 'tc_score']].set_index('KEGG_id')

pathways.to_pickle(os.path.join(analysis_folder,'KEGG_pathways.pkl'))

#%%
def parallel_Kegg(KEGG_input):
    net = KEGG_input['net']
    strategy = KEGG_input['strategy']
    sampling = KEGG_input['sampling']
    n_samples = KEGG_input['n_samples']
    KEGG_id = KEGG_input['KEGG_id']
    run_id = KEGG_input['run_id']

    nodes = list(net.nodes)
    perturbations = {'P{0}'.format(i):{nodes[i]} for i in range(len(nodes))}

    result = optimize_experimental_design(net, perturbations, strategy=strategy,
                                          sampling=sampling,n_samples=n_samples)

    result.update({'KEGG_id' : KEGG_id, 'strategy': strategy, 'sampling': sampling,
                   'n_samples':n_samples, 'run_id': run_id})

    file_name = 'KEGG_{0}_strategy_{1}_sampling_{2}_nsamples_{3}_runid_{4}.pkl'.format(
            KEGG_id, strategy, sampling, n_samples, run_id)

    with open(os.path.join(results_folder, 'KEGG/', file_name), 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    return result

n_samples = 10
run_id = 0
KEGG_inputs = []
for KEGG_id,pathway in pathways.iterrows():
    if pathway.N > max_net_size:
        continue

    run_id +=1
    KEGG_inputs.append(
            {'KEGG_id' : KEGG_id, 'net': pathway.net, 'strategy': 'greedy',
             'sampling': True, 'n_samples': n_samples, 'run_id':run_id})

    run_id +=1
    KEGG_inputs.append(
            {'KEGG_id' : KEGG_id, 'net': pathway.net, 'strategy': 'naive',
             'sampling': True, 'n_samples': n_samples, 'run_id':run_id})

    run_id +=1
    KEGG_inputs.append(
            {'KEGG_id' : KEGG_id, 'net': pathway.net, 'strategy': 'multi_target',
             'sampling': True, 'n_samples': n_samples, 'run_id':run_id})

    for i in range(n_samples):
        #in the random case I don't want to select the best out of the n_samples
        #run but take the mean over them. Therefore I run the individually.

        run_id +=1
        KEGG_inputs.append(
                {'KEGG_id' : KEGG_id, 'net': pathway.net, 'strategy': 'random',
                 'sampling': True, 'n_samples': 1, 'run_id':run_id})

    #uncomment this to run the exhaustive search! This will not finish for
    #networks larger than 20.

#    run_id +=1
#    KEGG_inputs.append(
#            {'KEGG_id' : KEGG_id, 'net': pathway.net, 'strategy': 'exhaustive',
#             'sampling': False, 'n_samples': n_samples, 'run_id':run_id})


#
if __name__ == '__main__':
    with Pool(nbr_used_cores) as p:
        KEGG_results=p.map(parallel_Kegg, KEGG_inputs)



#%%run single pathways
#pathway= pathways.loc['path:hsa01523'] #small pathway
##pathway= pathways.loc['path:hsa04114'] #medium pathway
##pathway= pathways.loc['path:hsa05200'] #large pathway
#result = parallel_Kegg({'KEGG_id' : pathway.name, 'net': pathway.net, 'strategy': 'greedy',
#               'sampling': True, 'n_samples': 10, 'run_id':1})