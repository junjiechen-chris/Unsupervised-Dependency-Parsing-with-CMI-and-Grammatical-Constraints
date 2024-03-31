import networkx as nx
from typing import Dict, Tuple, List, Set
import numpy as np
import torch
from torch.utils.data import DataLoader 
from .data import UDSentence
import datetime
from tqdm import tqdm
import pandas as pd



def compute_tp_fp_fn(gt: nx.DiGraph, pred:nx.DiGraph, use_directed_score = True, filter_fn = None):
    gt_edges: Set[Tuple[int]] = set(gt.edges())
    pred_edges: Set[Tuple[int]] = set(pred.edges())
    # print(pred_edges)
    if filter_fn is not None:
        # assert 
        gt_edges = set([item for item in gt_edges if filter_fn(item)])
        pred_edges = set([item for item in pred_edges if filter_fn(item)])
    # print(pred_edges)
        # pred_edges = set(filter(filter_fn, pred_edges))
    if not use_directed_score:
        # print([sorted(item) for item in gt_edges])
        gt_edges:Set[Tuple[int]] = set([tuple(sorted(item)) for item in gt_edges])
        pred_edges:Set[Tuple[int]] = set([tuple(sorted(item)) for item in pred_edges])
    tp = gt_edges.intersection(pred_edges)
    fp = pred_edges.difference(gt_edges)
    fn = gt_edges.difference(pred_edges)
    return np.array(list(map(len, [tp, fp, fn])))

def compute_tp_fp_fn_with_edges(gold, pred, use_directed_score = True, filter_fn = None):
    gt_edges =  set(gold)
    pred_edges = set(pred)
    # print(pred_edges)
    if filter_fn is not None:
        # assert 
        gt_edges = set([item for item in gt_edges if filter_fn(item)])
        pred_edges = set([item for item in pred_edges if filter_fn(item)])
    # print(pred_edges)
        # pred_edges = set(filter(filter_fn, pred_edges))
    if not use_directed_score:
        # print([sorted(item) for item in gt_edges])
        gt_edges:Set[Tuple[int]] = set([tuple(sorted(item)) for item in gt_edges])
        pred_edges:Set[Tuple[int]] = set([tuple(sorted(item)) for item in pred_edges])
    tp = gt_edges.intersection(pred_edges)
    fp = pred_edges.difference(gt_edges)
    fn = gt_edges.difference(pred_edges)
    return np.array(list(map(len, [tp, fp, fn])))
def compute_f_beta(tp_fp_fn , beta = 1):
    # print(tp_fp_fn)
    tp, fp, fn = np.split(tp_fp_fn, 3)
    return (1+beta**2)*tp/((1+beta**2)*tp + (beta**2)*fn +fp)

def compute_dda(tp_fp_fn: List[int]):
    tp, fp, fn = np.split(tp_fp_fn, 3)
    return tp/(tp+fp)

def compute_recall(tp_fp_fn: np.array):
    tp, fp, fn = np.split(tp_fp_fn, 3)
    return tp/(tp+fn)


def compute_tp_fp_fn_over_dataset(dset, use_directed_score = True, graph_strategy = None, ref_strategy = None):
    accumulator_tp_fp_fn = np.zeros(3)
    assert graph_strategy is not None
    for ud_item in tqdm(dset.examples):
        if not ud_item.overlength:
            parse = UDSentence.construct_graph_from_cache(ud_item, strategy = graph_strategy)
            ref = UDSentence.construct_graph_from_cache(ud_item, strategy = ref_strategy)
        else:
            parse = nx.DiGraph()
            ref = nx.DiGraph()
        # parse = UDSentence.decode_vinfo_maximum_spanning_tree(vinfo_graph, min_max='max')
        # ref = UDSentence.construct_graph_from_cache(ud_item, strategy = ref_strategy)
        accumulator_tp_fp_fn += compute_tp_fp_fn(ref, parse, use_directed_score=use_directed_score)
        # print(compute_tp_fp_fn(ref, parse, use_directed_score=use_directed_score))
    return accumulator_tp_fp_fn
    # print(accumulator_tp_fp_fn)

def compute_graph_vinfo_over_dataset(dset, graph_strategy):
    vinfo_list = []
    for ud_item in dset.examples:
        g = ud_item.construct_graph_from_cache(ud_item, strategy = graph_strategy)
        # parse = UDSentence.decode_vinfo_maximum_spanning_tree(vinfo_graph, min_max='max')
        # print(parse.edges.data('weight'))
        vinfo = np.array([e[-1] for e in g.edges.data('weight')])
        # print(vinfo)
        vinfo_list.append(vinfo)
    print(list(map(lambda x: (np.mean(x), np.std(x)),  vinfo_list)))

def compute_vinfos_for_graphs(dataset:object, precompute_strategy: Dict, vinfo_null_model, vinfo_informed_model, CollateFnForVinfoModel, batch_size = 32):
    vinfo = []
    dataset.prep_for_graph_decoding(dataset, strategy=precompute_strategy)#{'strategy_type':'mst', 'window_size':16, 'compute_missing_vinfo': True})
    if len(dataset) == 0:
        # print("NO AVAIABLE DATA")
        return 0., 0.
    loader = DataLoader(dataset, batch_size, collate_fn=partial(CollateFnForVinfoModel, tokenizer=tokenizer))
    with torch.no_grad():
        for batch_ndx, sample in enumerate(loader):
            x_idx = sample['i_choices']
            x_idx_mask = sample['i_choice_mask']
            y_idx = sample['j_choices']
            y_idx_mask = sample['j_choice_mask']
            true_ys = sample['encodings']['input_ids'].gather(1, y_idx)
            # print(y_idx)

            batch_size, max_y_seq_len = y_idx.size()

            null_logit = vinfo_null_model(x_idx, y_idx, sample['encodings'])['logits']
            null_y_log_probs = torch.log_softmax(null_logit, dim=-1).gather(2, true_ys.unsqueeze(-1))

            informed_logit = vinfo_informed_model(x_idx, y_idx, sample['encodings'])['logits']
            informed_y_log_probs = torch.log_softmax(informed_logit, dim=-1).gather(2, true_ys.unsqueeze(-1))

            joint_null_y_log_probs = (null_y_log_probs.squeeze(-1) * y_idx_mask).sum(-1)
            joint_informed_y_log_probs = (informed_y_log_probs.squeeze(-1) * y_idx_mask).sum(-1)

            vinfo.append(-(joint_null_y_log_probs-joint_informed_y_log_probs).cpu())
            if batch_ndx % 250 == 0:
                print("{}: current batch: {}".format(datetime.now(), batch_ndx), flush=True)
    vinfo = torch.cat(vinfo, dim=0)
    dataset.save_vinfos(dataset, vinfo)
    # print(vinfo.size())



def save_recall_by_syn_rel(dset, graph_strategy, output_dir):
    df = {'syn_rel': [], 'dependency_recall': []}
    for syn_rel in dset.syn_rels:
        if syn_rel == 'root': continue
        print("syn_rel: {}".format(syn_rel))
        df['syn_rel'].append(syn_rel)
        recall = compute_recall(compute_tp_fp_fn_over_dataset(dset, graph_strategy = graph_strategy, ref_strategy={**ud_strategy, 'relation_list_filter': [syn_rel]}))
        df['dependency_recall'].append(recall)
    pd.DataFrame(df).set_index('syn_rel').to_csv(os.path.join(output_dir, '{}.csv'.format(json.dumps(graph_strategy))))