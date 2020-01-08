#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Torsten Gross
"""

__all__=('infer_identifiability', 'infer_identifiability_by_simulation',
         'infer_identifiability_relationships','draw_identifiability_graph',
         'draw_lattice'
         )

import numpy as np
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D


#%%Identifiability from maximum flow problems


def infer_identifiability(net, perturbations, di_only=False):
    '''
    Given the network topology (provided as networkx DiGraph object 'net') and 
    perturbation targets (provided as dictionary 'perturbations') it returns 
    the identifiability status of the unknown network parameters (identifiability) 
    and the dimensionalities of the underlying solution spaces (d_i).
    
    If di_only=True only the dimensionality of the solution spaces is computed.
    This reduces computational effort.
    '''

    d_i={}
    identifiability = {'edges':{},'perturbations':{}}
    
    for node in net:
        consol_net = net.copy() #build the consolidated net for each node on which to compute node connectivity
        consol_net.add_nodes_from(['source','sink'])
        
        predecessors = list(net.predecessors(node))
        consol_net.add_edges_from([(predecessor,'sink') for predecessor in predecessors])
        
        known_perts, unknown_perts = [],[]
        for pert,targets in perturbations.items():
            if node in targets:
                unknown_perts.append(pert)
            else:
                known_perts.append(pert)
        
        for pert in known_perts:
            new_node_name = 'p{}'.format(pert)
            consol_net.add_node(new_node_name)
            consol_net.add_edge('source',new_node_name)
            for target in perturbations[pert]:
                consol_net.add_edge(new_node_name, target)
        
        node_cut = nx.minimum_node_cut(consol_net,'source','sink')
        min_ver_cut_size = len(node_cut)
        
        d_i[node] = len(predecessors) - min_ver_cut_size
        
        if di_only: continue

        #to check identifiability of edges one needs to solve a reduced max-flow problem.
        #Probably, I could just take previous result and reduce it. Not sure how to do it yet,
        #So at the expense of an additional order of complexity, for now I solve the max-flow problem
        #anew for every edge.
        for predecessor in predecessors:
            consol_net.remove_edge(predecessor,'sink')
            node_cut_squeezed = nx.minimum_node_cut(consol_net,'source','sink')
            min_ver_cut_size_squeezed = len(node_cut_squeezed)
            if min_ver_cut_size_squeezed < min_ver_cut_size:
                identifiability['edges'][(predecessor,node)] = True
            elif min_ver_cut_size_squeezed == min_ver_cut_size:
                identifiability['edges'][(predecessor,node)] = False
            else:
                raise(Exception('The size of the minimal vertex cut must not increase if I remove an edge.'))
            
            consol_net.add_edge(predecessor,'sink')
        
        #to check identifiability of sensitivity entries one needs to solve a max-flow problem with an additional source.
        
        for pert in unknown_perts:
            new_node_name = 'p{}'.format(pert)
            new_edges = [('source',new_node_name)]
            for target in perturbations[pert]:
                new_edges.append((new_node_name, target))
            consol_net.add_node(new_node_name)            
            consol_net.add_edges_from(new_edges)
            
            node_cut_expanded = nx.minimum_node_cut(consol_net,'source','sink')
            min_ver_cut_size_expanded = len(node_cut_expanded)

            if min_ver_cut_size_expanded == min_ver_cut_size:
                identifiability['perturbations'][(pert, node)] = True
            elif min_ver_cut_size_expanded > min_ver_cut_size:
                identifiability['perturbations'][(pert, node)] = False
            else:
                raise(Exception('The size of the minimal vertex cut must not decrease if I add a node.'))
            
            consol_net.remove_node(new_node_name) #this also removes all new_edges
            
        
    return d_i,identifiability


def infer_identifiability_by_simulation(net, perturbations):
    '''
    This function performs the same task as infer_identifiability but instead
    of solving a series of maximum flow problems it simulates a network response
    (based on random network parameters) and determines identifiability through
    computation of numerical ranks of the according matrices. This is mostly 
    just for verification. Yet, it can sometimes be more computationally 
    efficient. This comes at the expense of possible errors that occur because 
    ranks are have to be estimated by whether singular values cross a certain 
    threshold which might not be robust if the random matrices are nearly singular.
    '''
    
    nodes = list(net.nodes)
    perturbation_names = list(perturbations.keys())
    Jac_known = nx.to_numpy_array(net).T
    Sen_known = np.zeros([len(net), len(perturbations)])
    N,P=Sen_known.shape
    
    for row in range(N):
        for col in range(P):
            if nodes[row] in perturbations[perturbation_names[col]]:
                Sen_known[row,col] = 1

    Sen_trial = Sen_known.copy()
    Sen_trial[Sen_trial!=0]=np.random.rand(np.sum(Sen_trial!=0))
    Pexpt = np.eye(P)
    Jac_trial=Jac_known.copy()
    Jac_trial[Jac_trial!=0]=np.random.rand(np.sum(Jac_trial!=0))
    np.fill_diagonal(Jac_trial,-1.)
    try: 
        J_inv_S=np.linalg.lstsq(Jac_trial,Sen_trial,rcond=None)[0]
    except np.linalg.LinAlgError:
        return []
    
    Rexpt = - J_inv_S.dot(Pexpt)
#    Jac_identifiable, Sen_identifiable, d_i = determine_identifiable_parameters(
#        Rexpt,Jac_known,Sen_known,Pexpt,error_scale=max(500/(N*P),1) )
#    
    error_scale=max(500/(N*P),1)
    Jac_identifiable=np.full(Jac_known.shape,np.nan)
    Sen_identifiable=np.full(Sen_known.shape,np.nan)
    
    _,Sig,_=np.linalg.svd(Rexpt)
        
    round_err = error_scale * Sig.max() * np.finfo(Rexpt.dtype).eps
    d_i={}
    for i in range(N):
        Jaci_unknown_mask = Jac_known[i,:]!=0
        Seni_unknown_mask = Sen_known[i,:]!=0
        
        R_tilde=Rexpt.T[:,Jaci_unknown_mask]
        P_tilde=Pexpt.T[:,Seni_unknown_mask]
        Psi=np.concatenate((R_tilde, P_tilde),axis=1)
        if Psi.size==0:
            continue
        #determine rank of Psi
        U,Sig,VT=np.linalg.svd(Psi)        
        d_i[i]= VT.shape[1] - (Sig > (round_err * max(Rexpt.shape)) ).sum()
        
        if d_i[i]==0:
            Vi=np.empty([VT.shape[0],0])
        else:
            Vi=VT[-d_i[i] :,:].T
        
        identifiable_paras = np.sum(np.abs(Vi),1) < round_err

        Jac_identifiable[i,Jaci_unknown_mask] = identifiable_paras[:np.sum(Jaci_unknown_mask)]
        Sen_identifiable[i,Seni_unknown_mask] = identifiable_paras[np.sum(Jaci_unknown_mask):]
    
    
    identifiability = {'edges':{},'perturbations':{}}
    for i in range(N):
        for j in range(N):
            if Jac_known[i,j]!=0:
                if Jac_identifiable[i,j]:
                    identifiability['edges'][(nodes[j],nodes[i])]=True
                else:
                    identifiability['edges'][(nodes[j],nodes[i])]=False

    for i in range(N):
        for j in range(P):
            if Sen_known[i,j]!=0:
                if Sen_identifiable[i,j]:
                    identifiability['perturbations'][(perturbation_names[j], nodes[i])]=True
                else:
                    identifiability['perturbations'][(perturbation_names[j], nodes[i])]=False
        
    return d_i,identifiability

#%%Identifiability relationships from matroid computations


class dual_network_oracle:
    '''
    This dependency oracle is required to run the Boros algorithm that finds
    all matroid circuits. The oracle infers dependency of an index set (I) 
    based on according maximum-flow problem.
    '''
    def __init__(self, net, perturbations, node):
        net = net.copy()
        net.add_nodes_from(['source','sink'])
        predecessors = list(net.predecessors(node))
        
        net.add_edges_from([(predecessor,'sink') for predecessor in predecessors])
        
        known_perts, unknown_perts = [],[]
        for pert,targets in perturbations.items():
            if node in targets:
                unknown_perts.append(pert)
            else:
                known_perts.append(pert)
        for pert in known_perts:
            new_node_name = 'p{}'.format(pert)
            net.add_node(new_node_name)
            net.add_edge('source',new_node_name)
            for target in perturbations[pert]:
                net.add_edge(new_node_name, target)

        node_cut = nx.minimum_node_cut(net,'source','sink')
        min_ver_cut_size = len(node_cut)
        
        self.net = net
        self.predecessors = predecessors
        self.unknown_perts = unknown_perts
        self.u_i_J = len(predecessors)
        self.u_i_S = len(unknown_perts)
        self.u_i = self.u_i_J + self.u_i_S
        self.d_i = len(predecessors) - min_ver_cut_size
        
        self.eps_i = range(self.u_i) #the matroid ground set
        
        self.perturbations = perturbations
        return

    def test_independence(self,I):
        len_I = len(I)
        
        consol_net = self.net.copy()
        
        #alter the consol_net according to the index-subset I
        dual_I = [dual_i for dual_i in self.eps_i if not dual_i in I]
        
        dual_I_J = [i for i in dual_I if i < self.u_i_J]
        dual_I_S = [i-self.u_i_J for i in dual_I if i >= self.u_i_J]
        
        removal_edges = []
        for i in range(self.u_i_J):
            if i not in dual_I_J:
                removal_edges.append((self.predecessors[i],'sink'))
        consol_net.remove_edges_from(removal_edges)
        
        new_nodes = []
        new_edges = []
        for i in range(self.u_i_S):
            if i not in dual_I_S:  #"not" because I want to retain the perturbations not chosen by dual_index set
                pert = self.unknown_perts[i]
                new_node_name = 'p{}'.format(pert)
                new_nodes.append(new_node_name)
                new_edges.append(('source',new_node_name))
                for target in self.perturbations[pert]:
                    new_edges.append((new_node_name, target))
                    
        consol_net.add_nodes_from(new_nodes)
        consol_net.add_edges_from(new_edges)

        node_cut = nx.minimum_node_cut(consol_net,'source','sink')
        
        dual_rank =len(dual_I_S) + len(node_cut)
        
        rank = dual_rank + len_I - (self.u_i - self.d_i)
        
        if rank == len_I:
            return True
        elif rank < len_I:
            return False
        else:
            raise(Exception('The rank cannot be larger than the number of columns.'))

        
def enumerate_circuits(S,rank_M,test_independence):
    '''
    Returns the matroid circuits of matrix W. Uses the Boros algorithm.
    (Algorithms for Enumerating Circuits in Matroids, Boros et al. 2003)
    
    S is the matroid groundset.
    rank_M is the rank of the matroid.
    test_independence is a function that takes as input an index subset and 
    returns True if it is independent and False otherwise. The validity of the 
    dependence oracle is assumed and not checked.
    '''
    
    if rank_M==0:
        #at least in my application this is the case when all individual columns are considered
        #linearly dependent. So they are all loops, that is single-element circuits.
        return [set([i]) for i in S]
        

    
    #Determine fundamental circuits      
    base=[]
    for x in S:
        if test_independence(base+[x]):
            base.append(x)
        if len(base)==rank_M:
            break

    funda_circs=[]
    for x in S:
        if x in base: continue
        funda_circ=base+[x]
        ind_shift=0
        for del_ind in range(rank_M+1):
            reduced_funda_circ=funda_circ[:del_ind-ind_shift]+funda_circ[del_ind-ind_shift+1:]
            if not test_independence(reduced_funda_circ):
                funda_circ=reduced_funda_circ
                ind_shift+=1
        funda_circs.append(set(funda_circ))
    
    #Determine remaining circuits
    circuits=funda_circs
    open_circ_pairs=list(it.combinations(range(len(circuits)),r=2))
    while open_circ_pairs:
        circ_pair=open_circ_pairs.pop()
        circ_pair_union=circuits[circ_pair[0]].union(circuits[circ_pair[1]])
        for e in circuits[circ_pair[0]].intersection(circuits[circ_pair[1]]):
            circ_pair_union_reduced=circ_pair_union.difference(set([e]))
            for circ in circuits:
                if circ.issubset(circ_pair_union_reduced):
                    break
            else:
                #circuit axiom not closed, need to identify the missing circuit
                new_circ=list(circ_pair_union_reduced)
                ind_shift=0
                for del_ind in range(len(circ_pair_union_reduced)):
                    reduced_new_circ=new_circ[:del_ind-ind_shift]+new_circ[del_ind-ind_shift+1:]
                    if not test_independence(reduced_new_circ):
                        new_circ=reduced_new_circ
                        ind_shift+=1
                        
                #the new circuit must also be checked for circuit axiom closesness
                open_circ_pairs.extend(it.product(range(len(circuits)),[len(circuits)]))
                circuits.append(set(new_circ)) 
    return circuits




def circuits2cyclic_flats(circuits):
    '''
    Given a valid set of matroid circuits it returns the set of circuit flats
    and their ranks.
    '''
    
    circuits_r={}
    for circ in circuits:
        try:
            circuits_r[len(circ)-1].append(circ.copy())
        except KeyError:
            circuits_r[len(circ)-1]=[circ.copy()]
        
    
    circ_closures={}
    ranks=sorted(circuits_r.keys())
    for rank in ranks:
        circ_closures[rank]=[]
        circuits_r_curr=[circ.copy() for circ in circuits_r[rank]]
        while circuits_r_curr:
            base=circuits_r_curr.pop()
            circ_closure=base.copy()
            base.pop()
            #find the complete closure by checking which other circuits the base spans
            for rank_it in ranks:
                if rank_it>rank:
                    break
                for circ in circuits_r[rank_it]:
                    if len(base.intersection(circ))==(len(circ)-1):
                        circ_closure.update(circ)
    
            circ_closures[rank].append(circ_closure)
            #sort out all circuits that belong to previously found closure
            circuits_r_curr[:] = [circ for circ in circuits_r_curr if not circ.issubset(circ_closure)]
    
    return circ_closures




def infer_identifiability_relationships(net,perturbations):
    '''
    Given the network topology (net) and the targets of the perturabtions
    (perturbations) it returns the identifiability relationship between different
    non-identifiable network parameters in the form of the cyclic flats.
    '''
    
    cyclic_flats={}
    for node in net:
        oracle = dual_network_oracle(net, perturbations, node)
        circuits = enumerate_circuits(oracle.eps_i,oracle.d_i,oracle.test_independence)

        para_names = ['({0} -> {1})'.format(pre_n,node) for pre_n in net.predecessors(node)] + \
                     ['({0} -> {1})'.format(pert,node) for pert,targets in perturbations.items() if node in targets]
        para_names={i:para_name for i,para_name in enumerate(para_names)}
        circuits=[{para_names[c] for c in circuit} for circuit in circuits]
        
        cyclic_flats[node]=circuits2cyclic_flats(circuits)
                    
    return cyclic_flats

#%%Plotting

def draw_identifiability_graph(identifiability):
    color_nnode = 'lightgrey'
    color_pnode = (1.00, 212./255.,  42./255.)
    color_pnode = 'gold'
    
    font_size = 10
    
    color_identifiable_edge = (0.23, 0.56, 0.23)
    color_nonidentifiable_edge = (0.74, 0.24, 0.24)
    
    nn_edges, nn_edge_colors = [], []
    n_nodes = set([])
    for nn_edge,identifiable in identifiability['edges'].items():
        n_nodes.add(nn_edge[0])
        n_nodes.add(nn_edge[1])
        nn_edges.append(nn_edge)
        if identifiable:
            nn_edge_colors.append( color_identifiable_edge )
        else:
            nn_edge_colors.append( color_nonidentifiable_edge )

    pn_edges, pn_edge_colors = [], [] 
    p_nodes = set([])
    for pn_edge,identifiable in identifiability['perturbations'].items():
        p_nodes.add(pn_edge[0])
        pn_edges.append(pn_edge)
        if identifiable:
            pn_edge_colors.append( color_identifiable_edge )
        else:
            pn_edge_colors.append( color_nonidentifiable_edge )

    comprehensive_net = nx.DiGraph()
    comprehensive_net.add_edges_from(nn_edges)
    comprehensive_net.add_edges_from(pn_edges)

    pos = nx.spring_layout(comprehensive_net)  # positions for all nodes
    
    fig, ax = plt.subplots(figsize=[8,8])   
    
    nx.draw_networkx(comprehensive_net, pos,
                     nodelist=n_nodes,
                     node_color=color_nnode,
                     node_size=500,
                     with_labels = False,
                     edgelist = nn_edges,
                     edge_color = nn_edge_colors,
                     #arrowstyle='simple',
                     arrowsize=25,
                     width=2.,
                     alpha=1.,
                     ax=ax)
    
    nx.draw_networkx(comprehensive_net, pos,
                     nodelist=p_nodes,
                     node_color=color_pnode,
                     node_size=500,
                     node_shape = 's',
                     edgelist = pn_edges,
                     edge_color = pn_edge_colors,
                     arrowstyle='-',
                     width=2.,
                     #style='dotted',
                     alpha=1.,
                     with_labels = True,
                     font_size = font_size,
                     font_weight = 1000,
                     ax=ax)
    
    custom_lines = [Line2D([0], [0], color=color_identifiable_edge, lw=2),
                    Line2D([0], [0], color=color_nonidentifiable_edge, lw=2)]
    ax.legend(custom_lines, ['Identifiable', 'Non-identifiable'])

    plt.axis('off')
    return fig, ax

#circ_closures is a synonym for cyclic_flats
def draw_lattice(cyclic_flats):
    '''
    Create a graphical representation of the lattice of cyclic flats.
    '''
    
    lattice=nx.DiGraph()
    flat_pool=set([])

    for rank in sorted(cyclic_flats.keys()):
        for flat2 in cyclic_flats[rank].copy():
            lattice.add_node(frozenset(flat2))
            to_be_removed=[]
            for flat1 in flat_pool:
                if flat1.issubset(flat2):
                    lattice.add_edge(frozenset(flat1),frozenset(flat2))
                    to_be_removed.append(frozenset(flat1))
        flat_pool.difference_update(to_be_removed)
        flat_pool.update({frozenset(flat) for flat in cyclic_flats[rank]})
        
    pos={}
    for rank,flats in cyclic_flats.items():
        for x_pos,flat in enumerate(flats):
            pos[frozenset(flat)]=[x_pos,rank]
    labels={node:i for i,node in enumerate(lattice.nodes)}
    
    fig,ax=plt.subplots(figsize=[3,3])
    nx.draw_networkx(lattice,pos=pos,labels=labels,label=list(labels.keys()),ax=ax)
    ax.set_xticks([])
    ax.set_ylabel('rank')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.box(on=False)
    #plt.tight_layout()
    plt.text(ax.get_xlim()[0],ax.get_ylim()[1],''.join(['{0}: {1}\n'.format(i,', '.join(node)) for node,i in labels.items()]))

    plt.show()
    

    return fig,ax
