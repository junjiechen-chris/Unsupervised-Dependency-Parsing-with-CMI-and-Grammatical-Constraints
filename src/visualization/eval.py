# %%
import os
from json import load
import torch
from torch import nn
import sys
sys.path.append('/home/chris/projects/dep_syntax-MI-wisteria')
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Type, Set
import h5py

from src.common.data import DataArguments
from model import EsinerAgent

from transformers import(
    HfArgumentParser,
)

from data import UDDatasetForVisualization, UDSentenceForVisualization

import pandas as pd
import seaborn as sns

import numpy as np

import networkx
from matplotlib import pyplot as plt

from scipy.stats import ttest_ind

from copy import deepcopy

from collections import Counter
import pickle
from model import decoding as decode_pm
# import models.decoding as decode_pm
import scipy
from functools import partial

# %%
@dataclass
class DataArgumentsForVisualization(DataArguments):
    # fn_hdf5: str = field(default=None)
    # dir_hdf5:str = field(default=None)
    dataset: str = field(default=None)
    section: str = field(default=None)
    num_samples: int = field(default = None)
    with_pos:bool = field(default=True)
    printRB:bool = field(default=False)
    
    mode: str = field(default = "full")
    
working_dir = '/home/chris/projects/dep_syntax-MI-wisteria'
argparse_args = [
    "--dataset", "en_ewt",
    "--section", "test-w10",
    "--num_samples", "128",
    "--mode", "none"
]


parser = HfArgumentParser(
    [DataArgumentsForVisualization]
)
data_args = parser.parse_args_into_dataclasses()[0]

upos_mark = "upos" if data_args.with_pos else "wwm"

data_args.dev_data_file = f"data/{data_args.dataset}/{data_args.dataset}_extv2-ud-{data_args.section}.conllu"
data_args.fn_hdf5 = f"scores.{data_args.dataset}_extv2-ud-{data_args.section}.conllu.tar.qt1.5_{upos_mark}_fast.samples_{data_args.num_samples}"
data_args.dir_hdf5 = f"cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/{data_args.dataset}"



# %%

import importlib
import data
importlib.reload(data)
UDDatasetForVisualization = data.UDDatasetForVisualization
dev_dataset = UDDatasetForVisualization(os.path.join(working_dir, data_args.dev_data_file))
ADMISSIBLE_POS = set(['PRON', 'AUX', 'DET', 'NOUN', 'ADP', 'PROPN', 'VERB', 'NUM', 'ADJ', 'CCONJ', 'ADV', 'PART', 'INTJ', 'SYM', 'SCONJ', 'X', '_'])


# %%

filter_dep = lambda sentence, p: sentence._is_dependency(p.word_idx[0], p.word_idx[1]) or sentence._is_dependency(p.word_idx[1], p.word_idx[0])
filter_nondep = lambda sentence, p: not filter_dep(sentence, p)
filter_no_punct = lambda sentence, p: p.upos[0] != 'PUNCT' and p.upos[1] != 'PUNCT'
filters = [filter_dep]

# %%

h5 = h5py.File(os.path.join(working_dir, data_args.dir_hdf5, data_args.fn_hdf5), 'r')
measurement = 'energy-delta'
dev_dataset = UDDatasetForVisualization(os.path.join(working_dir, data_args.dev_data_file))

# %%
def fmi_bost_0_rows_fn(h5, measurement, sid, sent=None, heuristic_value = 1e-2, flag_bias_mode = False, flag_softmax = False, h5mode = 'abs'):
    if h5mode == 'abs':
        base_measurement = np.abs(h5[measurement][str(sid)][:])
    elif h5mode == 'raw':
        base_measurement = -h5[measurement][str(sid)][:]
    else:
        raise NotImplementedError(f'h5mode {h5mode} not implemented')
    # print(base_measurement)
    fill_mask = np.logical_not(deepcopy(sent.uposmask)) # True -> already filled, False -> not filled
    max_len = base_measurement.shape[0]
    # heuristic_value = 1e-
    
    for i in range(1, max_len):
        fill_mask = fill_mask[:-1]
        # if i>1:
        if flag_bias_mode:
            base_measurement_nk = np.diag(np.where(np.abs(h5[measurement][str(sid)][:]).sum(1)<9999., heuristic_value, 0)[:-i] * (sent.uposmask[i:])* (sent.uposmask[:-i]) * np.logical_not(fill_mask), i)
        else:
            base_measurement_nk = np.diag(np.where(np.abs(h5[measurement][str(sid)][:]).sum(1)<heuristic_value, heuristic_value, 0)[:-i] * (sent.uposmask[i:])* (sent.uposmask[:-i]) * np.logical_not(fill_mask), i)
        heuristic_value *= 0.95
        fill_mask = np.logical_or((np.logical_not(sent.uposmask[i:])* np.logical_not(sent.uposmask[:-i])).astype(bool), fill_mask)
        base_measurement += base_measurement_nk
        
    
    return base_measurement# + base_measurement_n1 + base_measurement_n2 + base_measurement_n3 + base_measurement_n4
    


    # np.diag(np.where(np.abs(h5[measurement][str(sid)][:]).sum(1)<1e-2, 1e-2, 0)[:-1] * sent.uposmask[1:]* sent.uposmask[:-1]
fmi_retrival_fn = lambda h5, measurement, sid, sent=None: np.abs(h5[measurement][str(sid)][:])
pmi_retrival_fn = lambda h5, measurement, sid, sent=None: np.triu(h5[measurement][str(sid)][:] + np.transpose(h5[measurement][str(sid)][:]))
pmi_abs_fn = lambda h5, measurement, sid, sent=None: np.triu(np.abs(h5[measurement][str(sid)][:]) + np.abs(np.transpose(h5[measurement][str(sid)][:])))

# %%

import model
import scipy
importlib.reload(model)
EsinerAgent = model.EsinerAgent
from tqdm import tqdm
decode_pm = model.decoding
EdmondAgentWithPOSConstraint = model.EdmondAgentWithPOSConstraint
esiner = EsinerAgent()
edmond = EdmondAgentWithPOSConstraint()

# %%

def parsing_accuracy(h5, measurement, h5_retrival_fn, flag_relax_assert = False, flag_printRB = False, flag_printRandom = False, flag_esiner = False, flag_apply_fn_unheading = True, flag_softmax = False):
    dev_dataset.sample_sentence([filter_dep, filter_no_punct], ADMISSIBLE_POS)
    tpfpfns = {
        'pred': np.array([0, 0, 0]),
        'pred_esiner': np.array([0, 0, 0]),
        'rb-tree': np.array([0, 0, 0]),
        'random': np.array([0, 0, 0]),
    }

    tptt_labeled = {
        'pred': {},
        'pred_esiner': {},
        'rb-tree': {},
        'random': {},
    }
    
    tptt_pred = {
        'pred': {},
        'pred_esiner': {},
        'rb-tree': {},
        'random': {},
    }
    
    label_set = set()

    tpfpfn_per_lineard = {'pred': {}, 'rb-tree': {}, 'random': {}, 'pred_esiner': {}}

    def update_tpfpfn(tpfpfn, pred, ref):
        tpfpfn[0] += len(pred.intersection(ref))
        tpfpfn[1] += len(pred.difference(ref))
        tpfpfn[2] += len(ref.difference(pred))
    
    def update_tptt_labeled(tptt, pred, ref):
        # print(ref)
        for dep, label in ref.items():
            label_set.add(label)
            if dep in pred:
                tptt[label] = tptt.get(label, np.array([0, 0])) + np.array([1, 1])
            else:
                tptt[label] = tptt.get(label, np.array([0, 0])) + np.array([0, 1])
                
    def update_tppred(tppred, pred, ref, upos_mask):
        # print(upos_mask)
        for dep in pred:
            dep_lineard = abs(upos_mask[:dep[1]].sum() - upos_mask[:dep[0]].sum())
            if dep in ref.keys():
                tppred[dep_lineard] = tppred.get(dep_lineard, np.array([0, 0])) + np.array([1, 1])
            else:
                tppred[dep_lineard] = tppred.get(dep_lineard, np.array([0, 0])) + np.array([0, 1])
    
    def update_tpfpfn_per_lineard(tpfpfn, pred, ref, upos_mask):
        for dep in pred:
            dep_lineard = abs(upos_mask[:dep[1]].sum() - upos_mask[:dep[0]].sum())
            if dep in ref:
                tpfpfn[dep_lineard] = tpfpfn.get(dep_lineard, np.array([0, 0, 0])) + np.array([1, 0, 0])
            else:
                tpfpfn[dep_lineard] = tpfpfn.get(dep_lineard, np.array([0, 0, 0])) + np.array([0, 1, 0])
        for dep in ref.difference(pred):
            dep_lineard = abs(upos_mask[:dep[1]].sum() - upos_mask[:dep[0]].sum())
            # lineard = abs(edge[1]-edge[0])
            tpfpfn[dep_lineard] = tpfpfn.get(dep_lineard, np.array([0, 0, 0])) + np.array([0, 0, 1]) 
        
        
                    
        
    def print_stat(tpfpfn, remark):
        print(f'##########tpfpfn-uuas-{remark}#############')
        tp, fp, fn = tpfpfn
        print('precision', tp/(tp+fp))
        print('f1', 2*tp/(2*tp+fp+fn))
    
    def print_stat_per_lineard(tpfpfn, remark):
        print(f'##########tpfpfn-uuas-{remark}#############')
        for dep_lineard, (tp, fp, fn) in sorted(tpfpfn.items(), key=lambda x: x[0]):
            print(f'{dep_lineard}', 2*tp/(2*tp+fp+fn),  tp+fn)

        
    def print_labeled_recall(tptt, remark):
        print(f'##########tptt-{remark}#############')
        for label, (tp, tt) in tptt.items():
            if tt>50 or True:
                print(label, tp, tt,  tp/tt)
                
    def print_precision(tppred, remark):
        print(f'##########tppred-{remark}#############')
        for dep_lineard, (tp, pred) in sorted(tppred.items(), key=lambda x: x[0]):
            if pred>20:
                print(dep_lineard, tp/pred)
        
        # print("tp, fp, fn", tp, fp, fn)
        # print('accuracy', tp/(tp+fp))
        # print('f1', 2*tp/(2*tp+fp+fn))
        
        
    intersect_pred_rb = 0
    for id, sent in enumerate(tqdm(dev_dataset.data)):
        # print(sent)
        # print(sent.root, sent.sentence.meta_data.sid, sum(sent.uposmask[:sent.root]))
        if len(sent) == 0: continue
        sid = sent.sentence.meta_data.sid
        measure = h5_retrival_fn(h5, measurement, sid, sent)#np.abs(h5[measurement][str(sid)][:])
        if flag_softmax:
            measure = scipy.special.softmax(measure + -9999. * np.tril(np.ones(measure.shape)), axis=1)
        # print(measure)
        
        # pred_fn = edmond.decode(measure, uposmask=sent.uposmask, upos_fn_mask=sent.upos_fn_mask)
        # print(g_esiner.edges)
        # return
        
        # print(measure)
        # g = networkx.from_numpy_array(measure, create_using=networkx.Graph).to_undirected()
        # pred = set(networkx.maximum_spanning_tree(g).edges())
        if flag_esiner:
            # measure = measure + measure.T
            print(sum(sent.uposmask[:sent.root]))
            # print(sent.sentence.words)
            print(sent.sentence._raw_tokens)
            print(sent.uposmask)
            # print(pred)
            # print(ref)
            print('----')
            pred = set(esiner.decode(measure, uposmask=sent.uposmask, root = sum(sent.uposmask[:sent.root])).edges)
        elif flag_apply_fn_unheading:
            pred = set(edmond.decode(measure, uposmask=sent.uposmask, upos_fn_mask=sent.upos_fn_mask))
        else: 
            pred = set(edmond.decode(measure, uposmask=sent.uposmask, upos_fn_mask=[False for i in range(len(measure))]))
            # g = networkx.from_numpy_array(measure, create_using=networkx.Graph).to_undirected()
            # pred = set(networkx.maximum_spanning_tree(g).edges())

        deps = sent.deps
        # print(pred)
        ref = set([tuple(sorted(dep.word_idx)) for dep in deps])
        ref_labeled = {tuple(sorted(dep.word_idx)): dep.dependency_label for dep in deps}
        # print(ref_labeled)
        # print(pred.intersection(ref), ref.difference(pred))

        
        if len(ref)!=len(pred) and not flag_relax_assert:
            print(sent.sentence._upos)
            print(sent.sentence.meta_data.sid)
            # print(sent.sentence.words)
            print(ref)
            
            # print(esiner.decode(measure, uposmask=sent.uposmask).raw_edges)
            print(pred) 
        if not flag_relax_assert:
            assert len(ref) == len(pred), 'the number of predicted edge must be equal to the number of gold edges'
            # assert len(ref) == len(pred_esiner), 'the number of esiner edge must be equal to the number of gold edges'
        if len(ref) == 0: continue
        rb = set([(i, i+1) for i in range(len(ref))])

        # print(sent.uposmask)
        # print(sent.sentence._upos)
        # print(sent.sentence.__dict__)
        random_tree = networkx.random_tree(n=sent.uposmask.sum())
        # print(np.where(sent.uposmask)[0])
        relabel_map = {i: j for i, j in enumerate(np.where(sent.uposmask)[0])}
        random_tree = set(networkx.relabel_nodes(random_tree, relabel_map).edges())

        rb_tree = networkx.from_edgelist([(i, i+1) for i in range(sent.uposmask.sum()-1)])
        rb_tree = set(networkx.relabel_nodes(rb_tree, relabel_map).edges())
        

        # print(ref)
        update_tpfpfn(tpfpfns['pred'], pred, ref)
        # update_tpfpfn(tpfpfns['pred_esiner'], pred_esiner, ref)
        update_tpfpfn(tpfpfns['rb-tree'], rb_tree, ref)
        update_tpfpfn(tpfpfns['random'], random_tree, ref)
        
        # update_tptt_labeled(tptt_labeled['pred_esiner'], pred_esiner, ref_labeled)
        update_tptt_labeled(tptt_labeled['pred'], pred, ref_labeled)
        update_tptt_labeled(tptt_labeled['rb-tree'], rb_tree, ref_labeled)
        update_tptt_labeled(tptt_labeled['random'], random_tree, ref_labeled)
        
        update_tppred(tptt_pred['pred'], pred, ref_labeled, sent.uposmask)
        
        update_tpfpfn_per_lineard(tpfpfn_per_lineard['pred'], pred, ref, sent.uposmask)
        update_tpfpfn_per_lineard(tpfpfn_per_lineard['rb-tree'], rb_tree, ref, sent.uposmask)
        update_tpfpfn_per_lineard(tpfpfn_per_lineard['random'], random_tree, ref, sent.uposmask)
        # update_tpfpfn_per_lineard(tpfpfn_per_lineard['pred_esiner'], pred_esiner, ref, sent.uposmask)
        
        intersect_pred_rb += len(pred.intersection(rb_tree).intersection(ref))
        
    print_stat(tpfpfns['pred'], 'pred')
    # print_stat_per_lineard(tpfpfn_per_lineard['pred'], 'pred')
    # print_labeled_recall(tptt_labeled['pred'], 'pred')
    print('###############')
    # print_stat(tpfpfns['pred_esiner'], 'pred_esiner')
    # print_labeled_recall(tptt_labeled['pred_esiner'], 'pred_esiner')
    # print('###############')
    # print('pred-rb intersection', intersect_pred_rb)
    if flag_printRB:
        print_stat(tpfpfns['rb-tree'], 'rb-tree')
        # print_labeled_recall(tptt_labeled['rb-tree'], 'rb-tree')
        # print_stat_per_lineard(tpfpfn_per_lineard['rb-tree'], 'rb-tree')
        
    if flag_printRandom:
        print_stat(tpfpfns['random'], 'random')
        print_labeled_recall(tptt_labeled['random'], 'random')
        print_stat_per_lineard(tpfpfn_per_lineard['random'], 'random')
    
    # print('####################')
        

# %%
if data_args.mode == "full":
    parsing_accuracy( h5, 'energy-delta', partial(fmi_bost_0_rows_fn, heuristic_value=0.5, flag_bias_mode = False), flag_relax_assert=False, flag_esiner=False, flag_printRB=data_args.printRB, flag_apply_fn_unheading=True, flag_softmax = False)
elif data_args.mode == "norb":
    parsing_accuracy( h5, 'energy-delta', partial(fmi_bost_0_rows_fn, heuristic_value=1e-7, flag_bias_mode = False), flag_relax_assert=False, flag_esiner=False, flag_printRB=data_args.printRB, flag_apply_fn_unheading=True, flag_softmax = False)
elif data_args.mode == 'nofnuh':
    parsing_accuracy( h5, 'energy-delta', partial(fmi_bost_0_rows_fn, heuristic_value=0.5, flag_bias_mode = False), flag_relax_assert=False, flag_esiner=False, flag_printRB=data_args.printRB, flag_apply_fn_unheading=False, flag_softmax = False)
elif data_args.mode == "none":
    parsing_accuracy( h5, 'energy-delta', partial(fmi_bost_0_rows_fn, heuristic_value=1e-7, flag_bias_mode = False), flag_relax_assert=False, flag_esiner=False, flag_printRB=data_args.printRB, flag_apply_fn_unheading=False, flag_softmax = False)


# %%
