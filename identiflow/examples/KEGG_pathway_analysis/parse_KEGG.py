#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Torsten Gross

Before using this script please ensure that you are authorized to access the 
KEGG databse. More information can be found on:
https://www.kegg.jp/kegg/legal.html
"""

import xml.etree.ElementTree as ET
import requests
import io
import pandas as pd
import networkx as nx
import os

analysis_folder=os.path.dirname(__file__)

human_pathways =  requests.get('http://rest.kegg.jp/list/pathway/hsa').text
human_pathways = pd.read_csv(io.StringIO(human_pathways), sep = '\t',
                             header=None, names = ['KEGG_id', 'description'])


#%%
pathways = []
for KEGG_id in human_pathways['KEGG_id']:
    print(KEGG_id)
    r = requests.get('http://rest.kegg.jp/get/{0}/kgml'.format(KEGG_id))
    pathway = ET.fromstring(r.text)
    edge_list = []
    for relation in pathway.iter('relation'):
        edge_list.append(( relation.attrib['entry1'],  relation.attrib['entry2']))
    if not edge_list: continue #empty pathway
    net = nx.from_edgelist(edge_list,nx.DiGraph())
    #I only want to look at connected components
    largest_cc = max(nx.connected_components(nx.Graph(net)), key=len)
    N = len(largest_cc)
    #filter out very small nets
    if N<5: continue
    net = net.subgraph( largest_cc ).copy()
    #compute isolation score
    tc_score = 1 - nx.number_of_edges(nx.algorithms.transitive_closure(net))/(N*(N-1))
    pathways.append({
            'KEGG_id':KEGG_id,
            'net' : net,
            'N': N,
            'tc_score' : tc_score,
            })

pathways = pd.DataFrame(pathways).set_index('KEGG_id')
pathways.to_pickle(os.path.join(analysis_folder,'KEGG_pathways.pkl'))


