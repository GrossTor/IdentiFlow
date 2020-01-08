#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Torsten Gross
"""
__all__=('optimize_experimental_design',)

import numpy as np
import networkx as nx
import itertools as it
from .identifiability import infer_identifiability


def get_performance(net, perturbations, di_only=False):#, counter = None):
    '''
    Returns the sum of solution spaces dimnesionalities and the number of identifiable edges.
    '''
#    if not counter is None:
#        counter['inference_counter'] +=1
        
    d_i, identifiability = infer_identifiability(net, perturbations, di_only)
    return {'sum_dis':sum(d_i.values()),'n_idable_edges': sum(identifiability['edges'].values())}


def get_perturbations_subset_single_target(all_perturbations, pert_set):
    return {pert : all_perturbations[pert] for pert in pert_set}
    
def get_perturbations_subset_multi_target(all_perturbations, pert_set):
    '''
    In the multi_target case pert_set is a set of sets (and not an integer set
    as in the single_target case) each of which describes 
    the perturbation combinations. Thus I need to build the perturbation subset
    from the full perturbation set as the union over the perturbation targets
    over the pert combis.
    '''
    
    perturbations_subset = {}
    for i,pert_combi in enumerate(pert_set):
        perturbations_subset['pert_combi_'+str(i)] = set([])
        for pert in pert_combi:
            perturbations_subset['pert_combi_'+str(i)].update(all_perturbations[pert])
    return perturbations_subset

def next_perts_greedy_strategy(curr_pert_set,model):
    
    best_performance = ( 0, np.inf)
    next_perts = set([])
    for pert in (model['perturbation_set'] - curr_pert_set):
        next_pert_set = frozenset([pert]).union(curr_pert_set) #this creates a frozenset
        if not next_pert_set in model['pert_set_performance']:
            perturbations_subset = get_perturbations_subset_single_target(model['perturbations'], next_pert_set)
            performance = get_performance(model['net'],  perturbations_subset)
            model['pert_set_performance'][next_pert_set] = performance
        else:
            performance = model['pert_set_performance'][next_pert_set]
        
        #select as next_perts those that maximize the number of identifiable links, and
        #have least sum of solution space dimensionality
        #take all that tie.
        
        if best_performance[0] < performance['n_idable_edges']:
            next_perts = set([pert])
            best_performance = ( performance['n_idable_edges'],
                                 performance['sum_dis'] )
        elif best_performance[0] == performance['n_idable_edges']:
            if best_performance[1] > performance['sum_dis']:
                next_perts = set([pert])
                best_performance = ( performance['n_idable_edges'],
                                     performance['sum_dis'] )
            elif best_performance[1] == performance['sum_dis']:
                next_perts.add(pert)
    
    return next_perts



def prepare_naive_strategy(model):    
    trans_closure = nx.transitive_closure( model['net'] )
    
    naive_score_perts = {}    
    for pert in model['perturbation_set']:
        targets = model['perturbations'][pert]
        naive_perturbation_score = sum( len(list(trans_closure.successors(target))) for target in targets )

        if naive_perturbation_score in naive_score_perts:
            naive_score_perts[naive_perturbation_score].add(pert) 
        else:
            naive_score_perts[naive_perturbation_score] = {pert}
    
    model['naive_score_perts'] = naive_score_perts
    model['sorted_scores'] = sorted(naive_score_perts.keys(), reverse=True)
    
    return
    
def next_perts_naive_strategy(curr_pert_set, model):    
    for curr_score in model['sorted_scores']:
        next_perts = model['naive_score_perts'][curr_score] - curr_pert_set
        if len( next_perts ) > 0:
            break
    return next_perts


def next_perts_random_strategy(curr_pert_set, model):
    next_perts = set([ np.random.choice(list(model['perturbation_set'] - curr_pert_set)) ])
    return next_perts

def next_perts_exhaustive_strategy(curr_pert_set, model):
    next_perts = model['perturbation_set'] - curr_pert_set
    return next_perts

def next_perts_multi_target_strategy(curr_pert_set, model, multi_target_exhaustive = False ):
    
    best_performance = ( 0, np.inf)
    last_pert_combis = [frozenset([])]
    
    for combi_size in range(1,len(model['perturbation_set'])+1):
        #print(combi_size)
        continue_extending = False
        next_pert_combis = set([])
        if multi_target_exhaustive:
            last_pert_combis_selection = last_pert_combis
        else:
            last_pert_combis_selection = [np.random.choice(list(last_pert_combis))]
        for last_pert_combi in last_pert_combis_selection:
            for pert in model['perturbation_set'].difference(last_pert_combi):
                #print('pert: ', pert)
                pert_combi = last_pert_combi.union(frozenset([pert]))
                if pert_combi in curr_pert_set:
                    continue
                pert_set = curr_pert_set.union( [ pert_combi ] )
                if pert_set in model['pert_set_performance']:
                    performance = model['pert_set_performance'][pert_set]
                else:        
                    perturbations_subset = get_perturbations_subset_multi_target(model['perturbations'], pert_set)
                    performance = get_performance(model['net'],  perturbations_subset)
                    model['pert_set_performance'][pert_set] = performance
                
                if best_performance[0] < performance['n_idable_edges']:
                    continue_extending = True
                    next_pert_combis = set([pert_combi])
                    best_performance = ( performance['n_idable_edges'],
                                         performance['sum_dis'] )
                elif best_performance[0] == performance['n_idable_edges']:
                    if best_performance[1] > performance['sum_dis']:
                        continue_extending = True
                        next_pert_combis = set([pert_combi])
                        best_performance = ( performance['n_idable_edges'],
                                             performance['sum_dis'] )
                    elif (best_performance[1] == performance['sum_dis']) and continue_extending:
                        #The second condition ensures that I don't consider a 
                        #larger combination if it does not increase the performance
                        next_pert_combis.add(pert_combi)
            
        if continue_extending:
            last_pert_combis = next_pert_combis
        else:
            break

    return list(last_pert_combis)
            
            

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(1,len(s)+1))
    
def next_perts_exhaustive_multi_target_strategy(curr_pert_set, model):
    next_perts = frozenset((frozenset(pert_combi) for pert_combi in powerset(model['perturbation_set']))).difference(
            curr_pert_set)
    return next_perts



def dfs(curr_pert_set, curr_pert_seq, curr_pert_seq_performance, model, 
        next_perts_strategy, get_perturbations_subset):
    '''
    This is a recursively implemented depth-first-search. Nodes are perturbation
    sets. Depending on the strategy (as implemented by "next_perts_strategy") 
    every node leads to node(s) whose according perturbations set is larger by one. 
    The search first finds the node that either fully determines the network or
    exhausts the perturbations. Then recursively jumps back to find other 
    perturbation sequences and stores those with higher performance.
    '''
    
    #print('Deepness : ',len(curr_pert_set),curr_pert_seq_performance)
    
    next_perts = next_perts_strategy(curr_pert_set,model)
    
    for next_pert in next_perts:
        next_curr_pert_set = curr_pert_set.union(frozenset([next_pert]))
        next_curr_pert_seq = curr_pert_seq + (next_pert,)
        if next_curr_pert_set in model['pert_set_performance']:
            performance = model['pert_set_performance'][next_curr_pert_set]
        else:
            perturbations_subset = get_perturbations_subset(model['perturbations'], next_curr_pert_set)
            performance = get_performance(model['net'],  perturbations_subset)#,counter = model)

            model['pert_set_performance'][next_curr_pert_set] = performance
        next_curr_pert_seq_performance = curr_pert_seq_performance + performance['n_idable_edges']
        
        if  (performance['sum_dis'] == 0) or (len(next_curr_pert_seq) == model['P']):
            #print('I went deep!')
            #the network is fully identifiable or the next pert_seq is the full set of perts. 
            #either way, I can stop the reccursion and see whether the sequence
            #is best performning.            
            
            rest_performance = ( model['P'] - len(next_curr_pert_seq) ) * performance['n_idable_edges']
            final_seq_performance = next_curr_pert_seq_performance + rest_performance
            #print(next_curr_pert_seq,final_seq_performance)
            if model['best_pert_seq_performance']<final_seq_performance:
                model['best_pert_seq_performance'] = final_seq_performance
                model['best_pert_seqs'] = [next_curr_pert_seq]
            elif model['best_pert_seq_performance'] == final_seq_performance:
                model['best_pert_seqs'].append(next_curr_pert_seq)
            
            continue
            
        else:
            dfs(next_curr_pert_set, next_curr_pert_seq, next_curr_pert_seq_performance, model,
                next_perts_strategy, get_perturbations_subset)


def randomly_sample_optimal_design_seq(model,next_perts_strategy,get_perturbations_subset):#(curr_pert_set, curr_pert_seq, curr_pert_seq_performance, pert_set_performance, model):
    '''
    In contrast to dfs, the choice on which perturbation amongst next_perturbations 
    to add to current perturbation set is done randomly. There is no recursive
    backtracking.
    '''
    
    
    curr_pert_set = frozenset([])
    curr_pert_seq = ()
    curr_pert_seq_performance = 0
    
    finished=False
    while not finished:        
        next_perts = next_perts_strategy(curr_pert_set,model)
    
        next_pert = np.random.choice(list(next_perts))
        curr_pert_set = curr_pert_set.union(frozenset([next_pert]))
        curr_pert_seq = curr_pert_seq + (next_pert,)

        if curr_pert_set in model['pert_set_performance']:
            performance = model['pert_set_performance'][curr_pert_set]
        else:
            perturbations_subset = get_perturbations_subset(model['perturbations'], curr_pert_set)            
            performance = get_performance(model['net'],  perturbations_subset)#,counter = model)
            model['pert_set_performance'][curr_pert_set] = performance



        curr_pert_seq_performance = curr_pert_seq_performance + performance['n_idable_edges']
        
        #print('Deepness : ',len(curr_pert_set), performance['sum_dis'])
        
        if  (performance['sum_dis'] == 0) or (len(curr_pert_seq) == model['P']):
            #print('I went deep!')
            #the network is fully identifiable or the next pert_seq is the full set of perts. 
            #either way, I can stop the reccursion and see whether the sequence
            #is best performning.            
            
            rest_performance = ( model['P'] - len(curr_pert_seq) ) * performance['n_idable_edges']
            final_seq_performance = curr_pert_seq_performance + rest_performance
            #print(next_curr_pert_seq,final_seq_performance)
            if model['best_pert_seq_performance']<final_seq_performance:
                model['best_pert_seq_performance'] = final_seq_performance
                model['best_pert_seqs'] = [curr_pert_seq]
            elif (model['best_pert_seq_performance'] == final_seq_performance) and\
                not (curr_pert_seq in model['best_pert_seqs']):
                model['best_pert_seqs'].append(curr_pert_seq)
            
            finished=True
            break
        
    return


def optimize_experimental_design(net,perturbations,strategy='greedy', 
                                 sampling=False,n_samples=None, multi_target_exhaustive=False):
    '''
    Given the network topology (provided as networkx DiGraph object **net**) and 
    perturbation targets (provided as dictionary **perturbations**), this function
    identifies perturbation sequence(s) that maximize the number of identifiable 
    network parameters with a minimal number of required perturbations. 
    
    strategy
    --------
    Generally, perturbation sequences are constructed stepwise. At each step a 
    set of next possible perturbations is suggested according to the chosen 
    strategy.
    
    **'greedy'** : Chooses all perturbations that maximize the number of 
    identifiable network paramters and minimize 
    the sum of the dimensionality of the solution space. This does not guarantee
    to identify the optimal perturbation sequence but typically does and has 
    good computational performance. This is the **recommended strategy**.
    
    **'exhaustive'** : Chooses all remaining perturbations. This guarantees to find
    the optimal sequence but is often computationally intractable.
    
    **'naive'** : Chooses the most upstream remaining perturbations. This is 
    computationally efficient, but typically shows worse performance.
    
    **'random'** : Chooses a random remaining perturbation.
    
    **'multi_target'** : Similar to the greedy strategy, except that the set of 
    possible perturbations now also consists of any combination of the original 
    perturbations. See below for more details.
        
    **'exhaustive_multi_target'** : Similar to the exhaustive strategy except that
    it considers all remaining perturbation combinations as possible next
    perturbations. It is therefore rarely computationally tractable.
    
    sampling
    --------
    
    If set to **False** optimal perturbation sequences are identified via a 
    depth-first search such that all perturbation sequences that conform to the
    chosen strategy are considered.
    
    If set to **True**, the search is restarted from an empty set of 
    perturbations **n_sample** times. At each step the next perturbation is
    chosen at random from the set of possible next perturbations. While this no
    longer guarantees to find the optimal sequence, it allows to treat larger
    networks where the number of permissible sequences becomes to large.
    
    In either case, only the best performing sequences are retained.
    
    multi_target_exhaustive
    -----------------------
    This is only relevant if the **multi_target** strategy was chosen. In that case
    the set of perturbation combinations is also built-up step-wise with increasing
    cardinality of the combinations. If set to **False** a randomly chosen permissible
    perturbation is added to the combination at each step. If set to **True**
    all permissible perturbations are considered. The optimal combination is 
    then found via another (nested) depth-first search. This is only possible 
    for small networks.
    
    Output
    ------
    
    The function returns a dictionary with the following items.
    
    *ident_AUC* : The identifiability area under the curve -  a measure of
    performance bounded between zero and one.
    
    *best_pert_seqs* : A set of the best performing perturbation sequences.
    If a multi-target strategy was chosen the elements of the sequences are sets
    that indicate a perturbation combination.
    
    *best_sum_dis* : An integer list indicating the sum of solution space 
    dimensionalities at each step along the optimal sequence of perturbations.
    
    *best_n_idable_edges* : An integer list indicating the number of identifiable 
    network edges at each step along the optimal sequence of perturbations.
    '''
    
    model = {}
    model['net'] = net
    model['perturbations'] = perturbations
    #N = len(net)
    P = len(perturbations)
    model['P'] = P
    model['perturbation_set'] = set(perturbations.keys())
    #model['inference_counter']=0
    model['best_pert_seq_performance'] = 0
    model['best_pert_seqs'] = 0
    model['pert_set_performance']={}

    
    if strategy=='greedy':
        next_perts_strategy = next_perts_greedy_strategy
        get_perturbations_subset = get_perturbations_subset_single_target

    elif strategy=='naive':
        prepare_naive_strategy(model)
        next_perts_strategy = next_perts_naive_strategy
        get_perturbations_subset = get_perturbations_subset_single_target
    
    elif strategy=='random':
        next_perts_strategy = next_perts_random_strategy
        get_perturbations_subset = get_perturbations_subset_single_target
    
    elif strategy=='exhaustive':
        next_perts_strategy = next_perts_exhaustive_strategy
        get_perturbations_subset = get_perturbations_subset_single_target
        
    elif strategy=='multi_target':
        if multi_target_exhaustive == False:
            next_perts_strategy = next_perts_multi_target_strategy
        else:
            next_perts_strategy = lambda x,y: next_perts_multi_target_strategy(x,y,multi_target_exhaustive=True)
        get_perturbations_subset = get_perturbations_subset_multi_target

    elif strategy=='exhaustive_multi_target':
        next_perts_strategy = next_perts_exhaustive_multi_target_strategy
        get_perturbations_subset = get_perturbations_subset_multi_target
    
    if not sampling:
        curr_pert_set = frozenset([])
        curr_pert_seq = ()
        curr_pert_seq_performance = 0
        
        dfs(curr_pert_set, curr_pert_seq, curr_pert_seq_performance, model, 
            next_perts_strategy, get_perturbations_subset)
    else:
        for i in range(n_samples):
            randomly_sample_optimal_design_seq(model,next_perts_strategy, get_perturbations_subset)
    AUC_normalization = P*len(net.edges)
    
    best_sum_dis, best_n_idable_edges = [], []
    best_pert_seq = model['best_pert_seqs'][0]
    for seq_len in range(1,len(best_pert_seq)+1):
        pert_set = frozenset(best_pert_seq[:seq_len])
        performance = model['pert_set_performance'][pert_set]
        best_sum_dis.append(performance['sum_dis'])
        best_n_idable_edges.append(performance['n_idable_edges'])   
    
    result = {'ident_AUC':  model[ 'best_pert_seq_performance']/AUC_normalization,
              'best_pert_seqs':model['best_pert_seqs'],
              'best_sum_dis':best_sum_dis,
              'best_n_idable_edges':best_n_idable_edges}
    
    return result

