#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Torsten Gross
"""

import networkx as nx
import identiflow

#%%Define network topology (as networkx DiGraph) and perturbation targets

edges = [('node 0', 'node 1'), 
         ('node 0', 'node 2'), 
         ('node 0', 'node 3'), 
         ('node 1', 'node 3'), 
         ('node 2', 'node 3'), 
         ('node 3', 'node 0'), 
         ('node 4', 'node 3'), 
         ('node 4', 'node 5'), 
         ('node 5', 'node 4')]

perturbations = {'P0': {'node 0', 'node 3'},
                 'P1': {'node 2'}, 
                 'P2': {'node 3', 'node 4'}}


net = nx.DiGraph(edges)

#There must be no self_loops. The next line ensures it.
net.remove_edges_from(nx.classes.selfloop_edges(net))

#%%Infer identifiability

sol_space_dims, identifiability = identiflow.infer_identifiability(net,perturbations)
sol_space_dims_simu, identifiability_simu = identiflow.infer_identifiability_by_simulation(net,perturbations)

fig,ax = identiflow.draw_identifiability_graph(identifiability)
#fig.savefig('identi_net.pdf')

#%%Infer relationships between non-identifiable paras

cyclic_flats_dict = identiflow.infer_identifiability_relationships(net,perturbations)
for node in cyclic_flats_dict:
    if sol_space_dims[node]>0:
        fig,ax=identiflow.draw_lattice(cyclic_flats_dict[node])
        #fig.savefig('matroid_{0}.pdf'.format(node),bbox_inches = "tight")
        
#%%Optimize experimental design 
#Depending on the network one might want to try different strategies. For
#explanations see function documentation.

nodes = list(net.nodes)
perturbations = {'P{0}'.format(i):{nodes[i]} for i in range(len(nodes))}

exhaustive = identiflow.optimize_experimental_design(net, perturbations,
                                strategy='exhaustive',sampling=False)

greedy = identiflow.optimize_experimental_design(net, perturbations,
                                strategy='greedy',sampling=False)

multi_target = identiflow.optimize_experimental_design(net, perturbations,
                                strategy='multi_target',sampling=False)

naive = identiflow.optimize_experimental_design(net, perturbations,
                                strategy='naive',sampling=False)

random = identiflow.optimize_experimental_design(net, perturbations,
                                strategy='random',sampling=True, n_samples=1)

#print some results
import pprint
print('\nPerformance:\n\n   exhaustive: {0}\n   greedy: {1}\n   multi_target: {2}\n   naive: {3}\n   random: {4}'.format(
        exhaustive['ident_AUC'],greedy['ident_AUC'],multi_target['ident_AUC'],naive['ident_AUC'], random['ident_AUC']))

print('\nBest perturbation sequences:\n\n   greedy:\n')
pprint.pprint(greedy['best_pert_seqs'])
print('\n   multi-target:\n')
pprint.pprint([tuple(set(combi) for combi in seq) for seq in multi_target['best_pert_seqs']])

