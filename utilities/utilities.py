from typing import Any, Optional, List, Dict, Tuple, Union, Type, Set
import networkx as nx
import os
from .data import UDSentence
import numpy as np
from torch import nn
from .data import CoNLLDataset 
import torch
import matplotlib.pyplot as plt
from .auxobjs import ListObj, DictObj
from random import random

# from unsupervised_parsing.utilities.data import UDSentence

upos_set = ['PRON', 'AUX', 'DET', 'NOUN', 'ADP', 'PROPN', 'VERB', 'NUM', 'ADJ', 'CCONJ', 'ADV', 'PART', 'INTJ', 'SYM', 'SCONJ', 'PUNCT', 'X', '_']
uposid2upos = {id:upos for id, upos in enumerate(upos_set)}
upos2uposid = {upos:id for id, upos in enumerate(upos_set)}


def printDependencyGraph(g, example, save_dir, id = 1):
    print(example._raw_tokens)
    token_counts = len(example._raw_tokens)
    posx=list(range(len(example._raw_tokens)+1))
    posy=len(posx)*[0]
    pos={i:[posx[i],posy[i]] for i in range(len(posx))}
    node_labels = {id+1: tok for id, tok in enumerate(example._raw_tokens)}
    fig = plt.figure(1, figsize=(0.5 * token_counts, 3), dpi=600)
    nx.draw(g, pos=pos, with_labels = True, connectionstyle='arc3,rad=-0.7', labels=node_labels)
    plt.savefig(os.path.join(save_dir, 'example_{}.jpg'.format(id)), bbox_inches='tight')
    plt.show() 

def printHeatmap_v2(example, save_dir, vinfo, ref_g=None, pred_g = None, id = 1, baseline = None, remark = ""):
    flag_highlight_dependencies = ref_g is not None and pred_g is not None
    if flag_highlight_dependencies is not None:
        ref_dependencies = set(ref_g.edges())
        pred_dependencies = set(pred_g.edges())

    vinfo = vinfo[1:len(example._raw_tokens)+1, 1:len(example._raw_tokens)+1]

    # print(vinfo)

    tokens = example._raw_tokens

    # plt.figure()
    fig, ax = plt.subplots(figsize=(0.5 * len(tokens)+6, 0.3*len(tokens)+6))
    im = ax.imshow(vinfo)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(tokens)), labels=tokens)
    ax.set_yticks(np.arange(len(tokens)), labels=tokens)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if flag_highlight_dependencies:
                if (i+1, j+1) in ref_dependencies and (i+1, j+1) not in pred_dependencies:
                    label = "{:.1f}_{}".format(vinfo[i, j], 'D')
                    text = ax.text(j, i, label, ha='center', va='center', color='r', fontweight='bold')
                elif (i+1, j+1) not in ref_dependencies and (i+1, j+1) in pred_dependencies:
                    label = "{:.1f}_{}".format(vinfo[i, j], 'D')
                    text = ax.text(j, i, label, ha='center', va='center', color='b', fontweight='bold')
                elif (i+1, j+1) in ref_dependencies and (i+1, j+1) in pred_dependencies:
                    label = "{:.1f}_{}".format(vinfo[i, j], 'D')
                    text = ax.text(j, i, label, ha='center', va='center', color='m', fontweight='bold')
                else:
                    label = "{:.1f}".format(vinfo[i, j])
                    text = ax.text(j, i, label, ha='center', va='center', color='w')
            else:
                label = "{:.1f}".format(vinfo[i, j])
                text = ax.text(j, i, label, ha='center', va='center', color='w')
            # print(label)

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmap_{}_{}.jpg'.format(id, remark)), bbox_inches='tight')
    plt.close()

def printHeatmap(example, save_dir, vinfo, ref_g=None, pred_g = None, id = 1, baseline = None, remark = ""):
    flag_highlight_dependencies = ref_g is not None and pred_g is not None
    if flag_highlight_dependencies is not None:
        ref_dependencies = set(ref_g.edges())
        pred_dependencies = set(pred_g.edges())

    # if baseline is not None:
    #     vinfo = UDSentence.get_vinfo_matrix(example, {'strategy_type':'vinfo-with-baseline', 'baseline': baseline})
    #         # print(vinfo_mtx)
    # else:
    # vinfo = UDSentence.get_vinfo_matrix(example, baseline)

    vinfo = vinfo[1:len(example._raw_tokens)+1, 1:len(example._raw_tokens)+1]

    # print(vinfo)

    tokens = example._raw_tokens

    # plt.figure()
    fig, ax = plt.subplots(figsize=(0.5 * len(tokens)+6, 0.3*len(tokens)+6))
    im = ax.imshow(vinfo)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(tokens)), labels=tokens)
    ax.set_yticks(np.arange(len(tokens)), labels=tokens)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if flag_highlight_dependencies:
                if (i+1, j+1) in ref_dependencies and (i+1, j+1) not in pred_dependencies:
                    label = "{:.1f}".format(vinfo[i, j], 'D')
                    text = ax.text(j, i, label, ha='center', va='center', color='r', fontweight='bold')
                elif (i+1, j+1) not in ref_dependencies and (i+1, j+1) in pred_dependencies:
                    label = "{:.1f}".format(vinfo[i, j], 'D')
                    text = ax.text(j, i, label, ha='center', va='center', color='b', fontweight='bold')
                elif (i+1, j+1) in ref_dependencies and (i+1, j+1) in pred_dependencies:
                    label = "{:.1f}".format(vinfo[i, j], 'D')
                    text = ax.text(j, i, label, ha='center', va='center', color='m', fontweight='bold')
                # elif (j+1, i+1) in dependencies:
                #     label = "{:.1f}_{}".format(vinfo[i, j], 'C')
                #     text = ax.text(j, i, label, ha='center', va='center', color='r', fontweight='bold')
                else:
                    label = "{:.1f}".format(vinfo[i, j])
                    text = ax.text(j, i, label, ha='center', va='center', color='w')
                # text = ax.text(i, j, label, ha='center', va='center', color='w')
            else:
                label = "{:.1f}".format(vinfo[i, j])
                text = ax.text(j, i, label, ha='center', va='center', color='w')
            # print(label)

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmap_{}_{}.jpg'.format(id, remark)), bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()

    # df = pd.DataFrame(vinfo, columns=tokens, index=tokens)
    # plt.figure(figsize=(0.5 * len(tokens)+6, 0.3*len(tokens)+6))
    # heatmap_plot = sns.heatmap(df, annot=True, fmt=".1f")
    # fig = heatmap_plot.get_figure()
    # plt.savefig(os.path.join(save_dir, 'heatmap_{}.jpg'.format(id)), bbox_inches='tight')
    # plt.show() 

class MyScheduler(nn.Module):
    def __init__(self, lbd) -> None:
        super().__init__()
        self.cnt = 0
        self.lbd = lbd
    
    def step(self):
        self.cnt+=1
    
    def get_param(self):
        return self.lbd(self.cnt)
    

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
    
class MarginalConditionalPOSMeanStdBaseline(Baseline):
    def __init__(self, dataset:CoNLLDataset):
        super().__init__()
        self.data: Dict[Tuple[str, str, int], List[np.array]] = {}
        self.collect_dataset_statistics(dataset)
        self.epsilon = 1e-9
    
    def collect_dataset_statistics(self, dataset: CoNLLDataset):
        for example in dataset.examples:
            if example.overlength: continue
            vinfo_mtx: np.array = example.h5dst[:]
            tok_seq_len:int = len(example._upos)
            tokens: List[str] = example._upos
            for i in range(tok_seq_len):
                for j in range(tok_seq_len):
                    if i==j: continue
                    # if np.isnan(vinfo_mtx[i, j]):
                        # print(i, j, example._raw, vinfo_mtx)
                        # return
                    tok_i, tok_j = list(map(lambda x: tokens[x], [i, j]))
                    key = (tok_i, tok_j, i-j)
                    if key not in self.data.keys():
                        self.data[key] = []
                    self.data[key].append(vinfo_mtx[i, j])
        mean_std = {key: (np.mean(item), np.std(item), len(item)) if len(item)>200 else (0, 1, 1) for key, item in self.data.items()}
        self.statistics = mean_std
        # return mean_std

    def forward(self, example: UDSentence, i:int, j:int, vinfo):
        pos_i = example._upos[i] 
        pos_j = example._upos[j]
        key = (pos_i, pos_j, i-j)
        mean, std, _ =  self.statistics.get(key, (0, 1, 1))
        return (vinfo-mean)/np.sqrt(std**2+self.epsilon)

        # return 

class DistMeanStdBaseline(Baseline):
    def __init__(self, dataset:CoNLLDataset):
        super().__init__()
        self.data: Dict[int, List[np.array]] = {}
        self.collect_dataset_statistics(dataset)
        self.epsilon = 1e-4
    
    def collect_dataset_statistics(self, dataset: CoNLLDataset):
        for example in dataset.examples:
            if example.overlength: continue
            vinfo_mtx: np.array = example.h5dst[:]
            tok_seq_len:int = len(example._raw_tokens)
            # tokens: List[str] = example._upos
            for i in range(tok_seq_len):
                for j in range(tok_seq_len):
                    if i==j: continue
                    key = i-j
                    if key not in self.data.keys():
                        self.data[key] = []
                    self.data[key].append(vinfo_mtx[i, j])
        mean_std = {key: (np.mean(item), np.std(item), len(item)) if len(item)>200 else (0, 1, 1) for key, item in self.data.items() }
        self.statistics = mean_std
        # return mean_std

    def forward(self, example: UDSentence, i:int, j:int, vinfo):
        key = i-j
        mean, std, _ =  self.statistics.get(key, (0, 1, 1))
        # print(vinfo , '->', (vinfo-mean)/np.sqrt(std**2+self.epsilon))
        return (vinfo-mean)/(std+self.epsilon)
        # return 

def CollateFnForNullModel(b_inputs, tokenizer, device = 'cuda:0'):
    b_size = len(b_inputs)
    # print(b_inputs)
    b_encodings = tokenizer(list(map(lambda x: x[1]._raw, b_inputs)), padding=True, return_tensors='pt')
    # b_encodings = tokenizer(list(map(lambda x: x[1], b_inputs)), padding=True, return_tensors='pt')
    b_i_indices = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x[0][0], dtype=torch.int64), b_inputs)), batch_first=True)
    b_i_index_mask = torch.where(b_i_indices==0, 0, 1)
    b_j_indices = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x[0][1], dtype=torch.int64), b_inputs)), batch_first=True)
    b_j_index_mask = torch.where(b_j_indices==0, 0, 1)
    
    return {
        "i_choices": b_i_indices.to(device),
        "i_choice_mask": b_i_index_mask.to(device),
        "j_choices": b_j_indices.to(device),
        "j_choice_mask": b_j_index_mask.to(device),
        "encodings": b_encodings.to(device)
    }

def CollateFnForPairwise(b_inputs, tokenizer, device = 'cuda:0'):
    b_size = len(b_inputs)
    b_encodings = tokenizer(list(map(lambda x: x['sent']._raw, b_inputs)), padding=True, return_tensors='pt')
    # b_encodings_x_appended = tokenizer(list(zip(list(map(lambda x: x['sent']._raw, b_inputs)), list(map(lambda x: x['x_token'], b_inputs)))), padding=True, return_tensors='pt')
    # b_encodings = tokenizer(list(map(lambda x: x[1], b_inputs)), padding=True, return_tensors='pt')
    b_i_indices = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x['xy_pair_idx'][0], dtype=torch.int64), b_inputs)), batch_first=True)
    b_i_index_mask = torch.where(b_i_indices==0, 0, 1)
    b_j_indices = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x['xy_pair_idx'][1], dtype=torch.int64), b_inputs)), batch_first=True)
    b_j_index_mask = torch.where(b_j_indices==0, 0, 1)
    
    return {
        "i_choices": b_i_indices.to(device),
        "i_choice_mask": b_i_index_mask.to(device),
        "j_choices": b_j_indices.to(device),
        "j_choice_mask": b_j_index_mask.to(device),
        "encodings": b_encodings.to(device),
        'sent': list(map(lambda x: x['sent']._raw, b_inputs))
    }

def CollateFnForPairwiseWithXAppending(b_inputs, tokenizer, device = 'cuda:0'):
    b_size = len(b_inputs)
    b_encodings = tokenizer(list(zip(list(map(lambda x: x['sent']._raw, b_inputs)), list(map(lambda x: x['x_token'], b_inputs)))), padding=True, return_tensors='pt')
    b_i_indices = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x['xy_pair_idx'][0], dtype=torch.int64), b_inputs)), batch_first=True)
    b_i_index_mask = torch.where(b_i_indices==0, 0, 1)
    b_j_indices = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x['xy_pair_idx'][1], dtype=torch.int64), b_inputs)), batch_first=True)
    b_j_index_mask = torch.where(b_j_indices==0, 0, 1)
    
    return {
        "i_choices": b_i_indices.to(device),
        "i_choice_mask": b_i_index_mask.to(device),
        "j_choices": b_j_indices.to(device),
        "j_choice_mask": b_j_index_mask.to(device),
        "encodings": b_encodings.to(device),
        'sent': list(map(lambda x: x['sent']._raw, b_inputs))
    }

def CollateFnForPairwiseWithDuplicating(b_inputs, tokenizer, device = 'cuda:0'):
    b_size = len(b_inputs)
    b_encodings = tokenizer(list(zip(list(map(lambda x: x['sent']._raw, b_inputs)), list(map(lambda x: x['sent']._raw, b_inputs)))), padding=True, return_tensors='pt')
    b_i_indices = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x['xy_pair_idx'][0], dtype=torch.int64), b_inputs)), batch_first=True)
    b_i_index_mask = torch.where(b_i_indices==0, 0, 1)
    b_j_indices = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x['xy_pair_idx'][1], dtype=torch.int64), b_inputs)), batch_first=True)
    b_j_index_mask = torch.where(b_j_indices==0, 0, 1)
    
    return {
        "i_choices": b_i_indices.to(device),
        "i_choice_mask": b_i_index_mask.to(device),
        "j_choices": b_j_indices.to(device),
        "j_choice_mask": b_j_index_mask.to(device),
        "encodings": b_encodings.to(device),
        'sent': list(map(lambda x: x['sent']._raw, b_inputs))
    }

def CollateFnForWWMLMInference(b_inputs, tokenizer, device = 'cuda:0'):
    b_size = len(b_inputs)
    
    b_x_idx = [input['x_idx'] for input in b_inputs]
    b_y_idx = [input['y_idx'] for input in b_inputs]
    x_tokens = [input['x_tokens'] for input in b_inputs]
    # print(b_x_idx)
    appending_tokens = [x_token for x_token in x_tokens]
    # print(appending_tokens)
    b_encodings = tokenizer(list(zip(list(map(lambda x: x['sent']._raw, b_inputs)), appending_tokens)), padding=True, return_tensors='pt')

    b_x_index_mask = torch.zeros_like(b_encodings['input_ids']).bool()
    for b_id, x_idx in enumerate(b_x_idx):
        for x_id in x_idx:
            b_x_index_mask[b_id, x_id] = True
    
    b_y_index_mask = torch.zeros_like(b_encodings['input_ids']).bool()
    for b_id, y_idx in enumerate(b_y_idx):
        for y_id in y_idx:
            b_y_index_mask[b_id, y_id] = True



    return {
        # "i_choices": b_i_indices.to(device),
        "i_choice_mask": b_x_index_mask.to(device),
        # "j_choices": b_j_indices.to(device),
        "j_choice_mask": b_y_index_mask.to(device),
        "encodings": b_encodings.to(device),
        'sent': list(map(lambda x: x['sent']._raw, b_inputs))
    }

def CollateFnForWWMLMWithOrderedMaskedTokens(b_inputs, tokenizer, device = 'cuda:0'):
    # This collate function is for the Wholw-Word-Masking LM objective
    b_size = len(b_inputs)
    b_raw_tokens: list[list[str]] = [input['sent']._raw_tokens for input in b_inputs]
    b_x_idx = [input['x_idx'] for input in b_inputs]
    b_y_idx = [input['y_idx'] for input in b_inputs]
    appending_tokens = [' '.join([rt for rt, ind in zip(raw_tokens, x_idx) if ind]) for raw_tokens, x_idx in zip(b_raw_tokens, b_x_idx)]

    b_encodings = tokenizer(list(zip(list(map(lambda x: x['sent']._raw, b_inputs)), appending_tokens)), padding=True, return_tensors='pt')

    b_subtoken_alignment = [input['subtoken_alignment'] for input in b_inputs]

    b_x_idx_for_subtokens = torch.zeros(b_encodings['input_ids'].size(), dtype=torch.bool)
    for b_idx, (mask_idx, alignment) in enumerate(zip(b_x_idx, b_subtoken_alignment)):
        alignment = alignment[1:] #drop the padding group in the alignment
        for t, ind in enumerate(mask_idx):
            if ind:
                for x_idx in alignment[t]:
                    b_x_idx_for_subtokens[b_idx, x_idx] = True 
    # b_i_indices_mask = b_x_idx_for_subtokens

    b_y_idx_for_subtokens = torch.zeros(b_encodings['input_ids'].size(), dtype=torch.bool)
    for b_idx, (mask_idx, alignment) in enumerate(zip(b_y_idx, b_subtoken_alignment)):
        alignment = alignment[1:] #drop the padding group in the alignment
        for t, ind in enumerate(mask_idx):
            if ind:
                for y_idx in alignment[t]:
                    b_y_idx_for_subtokens[b_idx, y_idx] = True 


    # b_j_indices_mask = torch.nn.utils.rnn.pad_sequence([inp['y_idx_subtokens'] for inp in b_inputs], batch_first=True)
    # b_j_indices_mask = torch.nn.functional.pad(b_j_indices_mask, (0, b_x_idx_for_subtokens.size(1)-b_j_indices_mask.size(1)))

    # b_j_indices_mask = torch.logical_and(b_j_indices_mask, torch.logical_not(b_i_indices_mask))
    b_y_idx_for_subtokens = torch.logical_and(b_y_idx_for_subtokens, torch.logical_not(b_x_idx_for_subtokens))

    
    return {
        # "i_choices": b_i_indices.to(device),
        "i_choice_mask": b_x_idx_for_subtokens.to(device),
        # "j_choices": b_j_indices.to(device),
        "j_choice_mask": b_y_idx_for_subtokens.to(device),
        "encodings": b_encodings.to(device),
        'sent': list(map(lambda x: x['sent']._raw, b_inputs))
    }
# This function will need some more thinkings 

def collateFnForSupervisedAnalysis(b_inputs_raw: List[DictObj], tokenizer, device='cuda:0', corruption_strategy = {}):

    b_inputs: ListObj = ListObj(b_inputs_raw)
    b_encodings = tokenizer(b_inputs.raw, padding=True, return_tensors = 'pt')
    subtoken_alignment = b_inputs.subtoken_alignment
    deps = b_inputs.deps
    tok_lens: List[int] = b_inputs.token_len
    # dep_rels = b_inputs.dep_rels
    leading_token_idx = torch.nn.utils.rnn.pad_sequence([torch.tensor([0]+[token_piece[0] for token_piece in alignment[1:]]) for bid, alignment in enumerate(subtoken_alignment)], batch_first=True) # -> (b, max_seq+1 (mask tok))
    # print(deps)
    assert max(tok_lens)+1 == leading_token_idx.size(1)
    assert max([len(_) for _ in deps]) == 1
    deps_idx = torch.tensor([_[0][1:2] for _ in deps])
    corruptable_token_mask = torch.where(b_encodings['input_ids']==tokenizer.pad_token_id, False, True)

    nonconfliting_deps = [[pair[:2] for pair in dep_list] for b_id, dep_list in enumerate(deps)]
    nonconfliting_deprels = [[pair[2] for pair in dep_list ] for b_id, dep_list in enumerate(deps)]
    # nonconfliting_deps = deps


    corruption_token_mask = []
    deptoken_mask = torch.zeros_like(corruptable_token_mask).bool()
    for bid, dep_list in enumerate(nonconfliting_deps):
        for hid, did in dep_list:
            for thid in subtoken_alignment[bid][hid]:
                deptoken_mask[bid, thid]=True
            for tdid in subtoken_alignment[bid][did]:
                deptoken_mask[bid, tdid]=True
    context_mask = torch.logical_not(deptoken_mask) * corruptable_token_mask

    for context_strategy in corruption_strategy.get('context_corruption_prob', [0.0]):


        context_corruption_prob = context_strategy#corruption_strategy.get('context_corruption_prob', 0.1)
        context_corruption_token_prob = context_mask.float().clone() * context_corruption_prob
        context_corruption_token_mask = torch.bernoulli(context_corruption_token_prob).bool()


        for strategy in corruption_strategy.get('deptoken_corruption_type', ['full']):
            # deptoken_corruption_type = strategy.get('deptoken_corruption_type', 'full')
            if strategy == 'full':
                deptoken_corruption_token_mask = deptoken_mask.clone()
            elif strategy == 'none':
                deptoken_corruption_token_mask = torch.zeros_like(corruptable_token_mask).bool()
            elif strategy == 'head':
                deptoken_corruption_token_mask = torch.zeros_like(corruptable_token_mask).bool()
                for bid, dep_list in enumerate(nonconfliting_deps):
                    for hid, did in dep_list:
                        for thid in subtoken_alignment[bid][hid]:
                            deptoken_corruption_token_mask[bid, thid]=True
            elif strategy == 'dependent':
                deptoken_corruption_token_mask = torch.zeros_like(corruptable_token_mask).bool()
                for bid, dep_list in enumerate(nonconfliting_deps):
                    for hid, did in dep_list:
                        for tdid in subtoken_alignment[bid][did]:
                            deptoken_corruption_token_mask[bid, tdid]=True
            else:
                raise NotImplementedError

            corruption_token_mask.append(torch.logical_or(deptoken_corruption_token_mask, context_corruption_token_mask))



    valid_loss_token_indices = torch.tensor([bid*leading_token_idx.size(1) + item[0] for bid, _ in enumerate(nonconfliting_deps) for item in _])
    head_to_gather_indices = torch.tensor([bid*leading_token_idx.size(1) + item[1] for bid, _ in enumerate(nonconfliting_deps) for item in _])
    nonconfliting_deps_seq_len = torch.tensor([tok_lens[bid]+1 for bid, _ in enumerate(nonconfliting_deps) for item in _])
    max_nonconfliting_deps_seq_len = torch.max(nonconfliting_deps_seq_len.flatten()).item()
    nonconfliting_deps_mask = torch.where(
        torch.arange(0, max_nonconfliting_deps_seq_len).unsqueeze(0).expand(nonconfliting_deps_seq_len.size(0), -1) < nonconfliting_deps_seq_len.unsqueeze(1).expand(-1, max_nonconfliting_deps_seq_len),
        True, False)

    nonconfliting_deps = torch.tensor([item[1] for _ in nonconfliting_deps for item in _])
    nonconfliting_deprels = torch.tensor([item for _ in nonconfliting_deprels for item in _])
    oh_nonconfliting_deps = nn.functional.one_hot(nonconfliting_deps, num_classes = max_nonconfliting_deps_seq_len)
    assert oh_nonconfliting_deps.size(0) == nonconfliting_deprels.size(0)
    oh_nonconfliting_joint = oh_nonconfliting_deps * nonconfliting_deprels.unsqueeze(-1)

    target_padding_size = (leading_token_idx.size(1) - max_nonconfliting_deps_seq_len)#.item()
    assert oh_nonconfliting_joint.size() == nonconfliting_deps_mask.size()

    return DictObj({
        'b_encodings': b_encodings.to(device),
        'leading_token_idx': leading_token_idx.to(device),
        'corruption_token_mask': [item.to(device) for item in corruption_token_mask],
        'valid_loss_token_indices': valid_loss_token_indices.to(device),
        'dep_to_gather_indices': valid_loss_token_indices.to(device),
        'head_to_gather_indices': head_to_gather_indices.to(device),
        'target': nn.functional.pad(nonconfliting_deps, (0, int(target_padding_size))).to(device),
        'target_mask': nn.functional.pad(nonconfliting_deps_mask, (0, int(target_padding_size))).to(device),
        'token_len': (torch.tensor(tok_lens)+1).to(device),
        'target_rel': nonconfliting_deprels.to(device),
        'target_joint' : nn.functional.pad(oh_nonconfliting_joint, (0, int(target_padding_size))).to(device),
        'deps_idx': deps_idx.to(device)
    })
    pass

def collateFnForSupervisedDependencyParsing(b_inputs_raw: List[DictObj], tokenizer, device='cuda:0', corruption_strategy = {}):
    b_inputs: ListObj = ListObj(b_inputs_raw)
    b_encodings = tokenizer(b_inputs.raw, padding=True, return_tensors = 'pt')
    subtoken_alignment = b_inputs.subtoken_alignment
    deps = b_inputs.deps
    tok_lens: List[int] = b_inputs.token_len
    # dep_rels = b_inputs.dep_rels
    leading_token_idx = torch.nn.utils.rnn.pad_sequence([torch.tensor([0]+[token_piece[0] for token_piece in alignment[1:]]) for bid, alignment in enumerate(subtoken_alignment)], batch_first=True) # -> (b, max_seq+1 (mask tok))
    assert max(tok_lens)+1 == leading_token_idx.size(1)


    # generating the training data according to the corruption strategy
    corruption_type = corruption_strategy.get('type', 'context')
    corruptable_token_mask = torch.where(b_encodings['input_ids']==tokenizer.pad_token_id, False, True)

    



    # print(corruptable_token_mask.float() * 0.1)
    if corruption_type == 'context':
        # 0. generate corruption tokens
        corruption_prob = corruption_strategy.get('prob', 0.0)
        corruption_token_prob = corruptable_token_mask.float() * corruption_prob
        corruption_token_mask = torch.bernoulli(corruption_token_prob).bool()
        # 1. get all lines that involved masked tokens
        nonconfliting_deps = [[pair[:2] for pair in dep_list if not (corruption_token_mask[b_id, pair[0]] or corruption_token_mask[b_id, pair[1]])] for b_id, dep_list in enumerate(deps)]
        nonconfliting_deprels = [[pair[2] for pair in dep_list if not (corruption_token_mask[b_id, pair[0]] or corruption_token_mask[b_id, pair[1]])] for b_id, dep_list in enumerate(deps)]

    elif corruption_type == 'headdep':
        dep_token_percentage = corruption_strategy.get('token_percentage', 0.1)
        deptoken_prob = corruptable_token_mask.float() * dep_token_percentage
        deptoken_mask = torch.bernoulli(deptoken_prob).bool()
        context_mask = torch.logical_not(deptoken_mask) * corruptable_token_mask

        deptoken_corruption_prob = corruption_strategy.get('deptoken_corruption_prob', 0.1)
        deptoken_corruption_token_prob = deptoken_mask.float() * deptoken_corruption_prob
        deptoken_corruption_token_mask = torch.bernoulli(deptoken_corruption_token_prob).bool()

        context_corruption_prob = corruption_strategy.get('context_corruption_prob', 0.1)
        context_corruption_token_prob = context_mask.float() * context_corruption_prob
        context_corruption_token_mask = torch.bernoulli(context_corruption_token_prob).bool()

        corruption_token_mask = torch.logical_or(deptoken_corruption_token_mask, context_corruption_token_mask)

        nonconfliting_deps = [[pair[:2] for pair in dep_list if (deptoken_mask[b_id, pair[0]] and deptoken_mask[b_id, pair[1]])] for b_id, dep_list in enumerate(deps)]
        nonconfliting_deprels = [[pair[2] for pair in dep_list if (deptoken_mask[b_id, pair[0]] and deptoken_mask[b_id, pair[1]])] for b_id, dep_list in enumerate(deps)]

    elif corruption_type == 'depselc':
        dep_keep_percentage = corruption_strategy.get('dep_keepprob', 0.1)
        max_dep_count = max([len(dep_list) for dep_list in deps])
        batch_size = len(deps)
        # deps = [(bid, *pair[:2]) for bid, dep_list in enumerate(deps) for pair in dep_list]
        dep_keep_mask = torch.bernoulli(torch.ones(max_dep_count*batch_size) * dep_keep_percentage).bool()

        nonconfliting_deps = [[pair[:2] for dep_id, pair in enumerate(dep_list) if dep_keep_mask[b_id*max_dep_count+dep_id]] for b_id, dep_list in enumerate(deps)]
        nonconfliting_deprels = [[pair[2] for dep_id, pair in enumerate(dep_list) if dep_keep_mask[b_id*max_dep_count+dep_id]] for b_id, dep_list in enumerate(deps)]


        # deptoken_prob = corruptable_token_mask.float() * dep_token_percentage
        deptoken_mask = torch.zeros_like(corruptable_token_mask).bool()
        for bid, dep_list in enumerate(nonconfliting_deps):
            for hid, did in dep_list:
                for thid in subtoken_alignment[bid][hid]:
                    deptoken_mask[bid, thid]=True
                for tdid in subtoken_alignment[bid][did]:
                    deptoken_mask[bid, tdid]=True
        context_mask = torch.logical_not(deptoken_mask) * corruptable_token_mask

        deptoken_corruption_prob = corruption_strategy.get('deptoken_corruption_prob', 0.1)
        deptoken_corruption_token_prob = deptoken_mask.float() * deptoken_corruption_prob
        deptoken_corruption_token_mask = torch.bernoulli(deptoken_corruption_token_prob).bool()

        context_corruption_prob = corruption_strategy.get('context_corruption_prob', 0.1)
        context_corruption_token_prob = context_mask.float() * context_corruption_prob
        context_corruption_token_mask = torch.bernoulli(context_corruption_token_prob).bool()

        corruption_token_mask = torch.logical_or(deptoken_corruption_token_mask, context_corruption_token_mask)

        nonconfliting_deps = [[pair[:2] for pair in dep_list if (deptoken_mask[b_id, pair[0]] and deptoken_mask[b_id, pair[1]])] for b_id, dep_list in enumerate(deps)]
        nonconfliting_deprels = [[pair[2] for pair in dep_list if (deptoken_mask[b_id, pair[0]] and deptoken_mask[b_id, pair[1]])] for b_id, dep_list in enumerate(deps)]
    else:        
        raise NotImplementedError
        # 2. create indices for the masked lines and the unmasked ones


    # assert len(nonconfliting_deps) > 0

    valid_loss_token_indices = torch.tensor([bid*leading_token_idx.size(1) + item[0] for bid, _ in enumerate(nonconfliting_deps) for item in _])
    head_to_gather_indices = torch.tensor([bid*leading_token_idx.size(1) + item[1] for bid, _ in enumerate(nonconfliting_deps) for item in _])
    nonconfliting_deps_seq_len = torch.tensor([tok_lens[bid]+1 for bid, _ in enumerate(nonconfliting_deps) for item in _])
    if nonconfliting_deps_seq_len.numel() == 0:
        max_nonconfliting_deps_seq_len = 0
    # print(nonconfliting_deps_seq_len.size())
    else:
        max_nonconfliting_deps_seq_len = torch.max(nonconfliting_deps_seq_len.flatten()).item()
    # print(max_nonconfliting_deps_seq_len)
    nonconfliting_deps_mask = torch.where(
        torch.arange(0, max_nonconfliting_deps_seq_len).unsqueeze(0).expand(nonconfliting_deps_seq_len.size(0), -1) < nonconfliting_deps_seq_len.unsqueeze(1).expand(-1, max_nonconfliting_deps_seq_len),
        True, False)

    nonconfliting_deps = torch.tensor([item[1] for _ in nonconfliting_deps for item in _])
    nonconfliting_deprels = torch.tensor([item for _ in nonconfliting_deprels for item in _])
    oh_nonconfliting_deps = nn.functional.one_hot(nonconfliting_deps, num_classes = max_nonconfliting_deps_seq_len)
    assert oh_nonconfliting_deps.size(0) == nonconfliting_deprels.size(0)
    oh_nonconfliting_joint = oh_nonconfliting_deps * nonconfliting_deprels.unsqueeze(-1)

    target_padding_size = (leading_token_idx.size(1) - max_nonconfliting_deps_seq_len)#.item()
    assert oh_nonconfliting_joint.size() == nonconfliting_deps_mask.size()

    return DictObj({
        'b_encodings': b_encodings.to(device),
        'leading_token_idx': leading_token_idx.to(device),
        'corruption_token_mask': corruption_token_mask.to(device),
        'valid_loss_token_indices': valid_loss_token_indices.to(device),
        'dep_to_gather_indices': valid_loss_token_indices.to(device),
        'head_to_gather_indices': head_to_gather_indices.to(device),
        'target': nn.functional.pad(nonconfliting_deps, (0, int(target_padding_size))).to(device),
        'target_mask': nn.functional.pad(nonconfliting_deps_mask, (0, int(target_padding_size))).to(device),
        'token_len': (torch.tensor(tok_lens)+1).to(device),
        'target_rel': nonconfliting_deprels.to(device),
        'target_joint' : nn.functional.pad(oh_nonconfliting_joint, (0, int(target_padding_size))).to(device)
    })





def collateFnForHeatmapPrinting(b_inputs: List[Dict[None,None]], tokenizer, device = 'cuda:0', pad_constant = -np.inf):
    b_encodings = tokenizer(list(map(lambda x: x['sent']._raw, b_inputs)), padding=True, return_tensors='pt')
    b_vinfo_mtx: List[np.Array] = []
    ref_graphs = []
    ref_graphs_nopunct = []
    punct_idx_list = []

    b_buffer: List[int] = [len(item['sent']._raw_tokens) for item in b_inputs]
    max_buffer_size = max(b_buffer)
    print('max_buffer_size: ', max_buffer_size, b_buffer)

    for _, item in enumerate(b_inputs):
        item = item['sent']
        vinfo_mtx = UDSentence.get_vinfo_matrix(item, baseline=None)
        # if _ == 0:
        #     print(vinfo_mtx[: , :, 0])

        # vinfo_mtx = np.tril(vinfo_mtx+100, k=-1)
        
        ref_deps: List[Tuple[int, int, str]] = UDSentence.sample_ij_pairs(item, {'strategy_type':'syn-deps', 'reverse': True}, seq_len=0)
        ref_deps_nopunct: List[Tuple[int, int, str]] = UDSentence.sample_ij_pairs(item, {'strategy_type':'syn-deps', 'reverse': True, 'remove_punct': True}, seq_len=0)

        # print(len(ref_deps))
        # 
        g:nx.DiGraph = nx.DiGraph()
        g.add_edges_from([item[:2] for item in ref_deps])
        ref_graphs.append(g)
        
        nopunct_g = nx.DiGraph()
        nopunct_g.add_edges_from([item[:2] for item in ref_deps_nopunct])
        ref_graphs_nopunct.append(nopunct_g)

        punct_idx = [id for id, _ in enumerate(item._upos) if _=='PUNCT']
        punct_idx_list.append(punct_idx)


        padded_vinfo_mtx = np.pad(vinfo_mtx, ((1, max_buffer_size-len(item._raw_tokens)), (1, max_buffer_size-len(item._raw_tokens)), (0, 0)), constant_values = pad_constant)
        b_vinfo_mtx.append(np.array(padded_vinfo_mtx))

    print([item.shape for item in b_vinfo_mtx])

    return {
        'b_vinfo_mtx': torch.tensor(b_vinfo_mtx, device = device),
        'b_buffer_size': torch.tensor(b_buffer, dtype=torch.int, device = device).unsqueeze(-1)+1,
        'b_encodings': b_encodings.to(device), 
        'raw_tokens': [item['sent']._raw_tokens for item in b_inputs],
        'ref_graphs': ref_graphs,
        'ref_graphs_nopunct': ref_graphs_nopunct,
        'b_inputs': b_inputs,
        'punct_idx': punct_idx_list,
        'b_sents': [item['sent'] for item in b_inputs]
    }
    pass


def collateFnForGPTProbing(b_inputs: List[DictObj], tokenizer, device = 'cuda:0', num_samples = 32) -> Dict:
    # b_inputs : A list of DictObj where raw is the truncated sentence -> reduce computational requirement
    bz = len(b_inputs)
    b_inputs: ListObj = ListObj(b_inputs)
    b_encodings = tokenizer([raw for raw in b_inputs.raw for _ in range(num_samples)], padding=True, return_tensors = 'pt')
    b_encodings_target = tokenizer([raw for raw in b_inputs.raw_target for _ in range(num_samples)], padding=True, return_tensors = 'pt')
    b_encodings_target.input_ids = b_encodings_target.input_ids[:, 1:] # remove the bos token
    b_encodings_target.attention_mask = b_encodings_target.attention_mask[:, 1:] # remove the bos token

    subtoken_alignments = b_inputs.subtoken_alignment

    dependent_subtoken_idx = torch.tensor([subtoken_alignments[bid][dep_id][-1] for bid, dep_id in enumerate(b_inputs.dependent_id) for _ in range(num_samples)], dtype=torch.long).view(-1, 1)
    cpu_dependent_token_idx = b_inputs.dependent_id

    sid = b_inputs.sid

    # for _ in range(bz):
    #     print('raw <bos>{}<eos>'.format(b_inputs.raw_target[_]))
    #     print("input: {} || target: {}".format(
    #         tokenizer.convert_ids_to_tokens(b_encodings.input_ids[_*num_samples]), 
    #         tokenizer.convert_ids_to_tokens(b_encodings_target.input_ids[_*num_samples])))


    return {
        'b_encodings': b_encodings.to(device),
        'b_encodings_target': b_encodings_target.to(device),
        'subtoken_alignments': subtoken_alignments,
        'num_samples': num_samples,
        'bz': bz, 
        'dependent_token_idx': dependent_subtoken_idx.to(device),
        'cpu_dependent_token_idx': cpu_dependent_token_idx,
        'sid': sid, 
    }

def collateFnForGPTSubtokReplacingProbing(b_inputs: List[DictObj], tokenizer, device = 'cuda:0') -> Dict:
    # b_inputs : A list of DictObj where raw is the truncated sentence -> reduce computational requirement
    bz = len(b_inputs)
    b_inputs: ListObj = ListObj(b_inputs)

    # print(b_inputs.num_samples)
    num_samples = b_inputs.num_samples[0]

    b_raw_baseline = [' '.join(raw) for item in b_inputs.raw_head_random for raw in item]
    b_raw_test = [' '.join(raw) for item in b_inputs.raw_head_sameclass for raw in item]
    b_raw_target= [' '.join(raw) for item in b_inputs.raw_target for raw in item]#_ in range(num_samples)]

    # print('baseline', b_raw_baseline)
    # print('test', b_raw_test)
    # print(b_raw_target)

    b_encodings_baseline = tokenizer(b_raw_baseline, padding=True, return_tensors='pt')
    b_encodings_test = tokenizer(b_raw_test, padding=True, return_tensors='pt')
    b_encodings_target = tokenizer(b_raw_target, padding=True, return_tensors='pt')
    b_gather_token_idx_baseline = b_encodings_baseline.attention_mask.sum(-1, keepdim=True)-1
    b_gather_token_idx_test = b_encodings_test.attention_mask.sum(-1, keepdim=True)-1
    b_input_ids_target = b_encodings_target.input_ids.gather(1, b_encodings_target.attention_mask.sum(-1, keepdim=True)-1)


    sid = b_inputs.sid


    return {
        'b_encodings_baseline': b_encodings_baseline.to(device),
        'b_encodings_test': b_encodings_test.to(device),
        'b_input_ids_target': b_input_ids_target.to(device),
        'b_gather_token_idx_baseline': b_gather_token_idx_baseline.to(device),
        'b_gather_token_idx_test': b_gather_token_idx_test.to(device),
        'head_idx': b_inputs.head_id,
        'dependent_idx': b_inputs.dependent_id,
        'num_samples': num_samples,
        'bz': bz, 
        'sid': sid, 
    }

def collateFnForbidirGPTProbing(b_inputs: List[DictObj], tokenizer, device='cuda:0', num_samples = 32) -> Dict:
    # b_inputs : A list of DictObj where raw is the truncated sentence -> reduce computational requirement
    # b_inputs.dependent_id := id of dependent starting from 0, also serves as the id for grabbing the dependent token prediction from GPT

    bz = len(b_inputs)
    b_inputs: ListObj = ListObj(b_inputs)

    raws_forward = [' '.join([tokenizer.bos_token]+raw_tokens[:dep_id]) for raw_tokens, dep_id in zip(b_inputs.raw_tokens, b_inputs.dependent_id)]
    raws_forward_target = [' '.join([tokenizer.bos_token]+raw_tokens[:dep_id+1]) for raw_tokens, dep_id in zip(b_inputs.raw_tokens, b_inputs.dependent_id)]
    b_encodings_forward = tokenizer([raw for raw in raws_forward for _ in range(num_samples)], padding=True, return_tensors = 'pt')
    b_encodings_forward.data['input_ids']+=1 
    b_encodings_forward_target = tokenizer([raw for raw in raws_forward_target for _ in range(num_samples)], padding=True, return_tensors = 'pt')
    b_encodings_forward_target.data['input_ids'] = b_encodings_forward_target.input_ids[:, 1:] +1 # remove the bos token
    b_encodings_forward_target.data['attention_mask'] = b_encodings_forward_target.attention_mask[:, 1:] # remove the bos token

    raws_backward = [' '+' '.join(raw_tokens[dep_id+1:])+tokenizer.eos_token for raw_tokens, dep_id in zip(b_inputs.raw_tokens, b_inputs.dependent_id)]
    raws_backward_target = [' '+ ' '.join(raw_tokens[dep_id:]) +tokenizer.eos_token for raw_tokens, dep_id in zip(b_inputs.raw_tokens, b_inputs.dependent_id)]
    # print(raws_backward)
    # print(raws_backward_target)
    b_encodings_backward = tokenizer([raw for raw in raws_backward for _ in range(num_samples)])
    b_encodings_backward_target = tokenizer([raw for raw in raws_backward_target for _ in range(num_samples)])
    b_encodings_backward.data['input_ids'] = nn.utils.rnn.pad_sequence([torch.tensor(input_ids[::-1]) for input_ids in b_encodings_backward.input_ids], batch_first=True, padding_value=tokenizer.bos_token_id)+1
    b_encodings_backward.data['attention_mask'] = nn.utils.rnn.pad_sequence([torch.tensor(input_ids[::-1]) for input_ids in b_encodings_backward.attention_mask], batch_first=True, padding_value=0)
    # the target requires a right-shift 
    # print(b_encodings_backward_target.input_ids)
    b_encodings_backward_target.data['input_ids'] = nn.utils.rnn.pad_sequence([torch.tensor(input_ids[::-1][1:]) for input_ids in b_encodings_backward_target.input_ids], batch_first=True, padding_value=tokenizer.bos_token_id)+1  # the backward-forward model requires +1 offset in input_ids
    b_encodings_backward_target.data['attention_mask'] = nn.utils.rnn.pad_sequence([torch.tensor(input_ids[::-1][1:]) for input_ids in b_encodings_backward_target.attention_mask], batch_first=True, padding_value=0)
    # print(b_encodings_backward_target.input_ids)
    # print(b_encodings_forward)
    # print(b_encodings_backward)
    # print(b_encodings_backward.__getattr__('input_ids'))

    forward_subtoken_alignments = b_inputs.subtoken_alignment
    max_subtoken_idx = [alignment[-1][-1] for alignment in b_inputs.subtoken_alignment]
    backward_subtoken_alignments = [[[max_piece_id-piece_id for piece_id in token] for token in alignment] for alignment, max_piece_id in zip(b_inputs.subtoken_alignment, max_subtoken_idx) ] # flip the index


    forward_gather_subtoken_idx = torch.tensor([forward_subtoken_alignments[bid][dep_id][-1] for bid, dep_id in enumerate(b_inputs.dependent_id) for _ in range(num_samples)], dtype=torch.long).view(-1, 1)
    backward_gather_subtoken_idx = torch.tensor([backward_subtoken_alignments[bid][dep_id+2][0] for bid, dep_id in enumerate(b_inputs.dependent_id) for _ in range(num_samples)], dtype=torch.long).view(-1, 1)
    cpu_dependent_token_idx = b_inputs.dependent_id
    # print([(len(raw_tokens), dep_id) for dep_id, raw_tokens in zip(b_inputs.dependent_id, b_inputs.raw_tokens)])
    # cpu_dependent_token_idx_backward = [len(raw_tokens)-dep_id-1 for dep_id, raw_tokens in zip(b_inputs.dependent_id, b_inputs.raw_tokens)]

    sid = b_inputs.sid

    # for _ in range(bz):
    #     print('raw <bos>{}<eos>'.format(b_inputs.raw_target[_]))
    #     print("input: {} || target: {}".format(
    #         tokenizer.convert_ids_to_tokens(b_encodings.input_ids[_*num_samples]), 
    #         tokenizer.convert_ids_to_tokens(b_encodings_target.input_ids[_*num_samples])))


    return {
        'b_raw_tokens': b_inputs.raw_tokens,
        'b_encodings_forward': b_encodings_forward.to(device),
        'b_encodings_backward': b_encodings_backward.to(device),
        'b_encodings_forward_target': b_encodings_forward_target.to(device),
        'b_encodings_backward_target': b_encodings_backward_target.to(device),
        'subtoken_alignments_forward': forward_subtoken_alignments,
        'subtoken_alignments_backward': backward_subtoken_alignments,
        'num_samples': num_samples,
        'bz': bz, 
        # 'seqlen_token': [len(_) for _ in b_inputs.raw_tokens],
        'gather_subtoken_idx_forward': forward_gather_subtoken_idx.to(device),
        'gather_subtoken_idx_backward': backward_gather_subtoken_idx.to(device), 
        'dependent_token_idx': cpu_dependent_token_idx,
        # 'store_dependent_token_idx_forward': cpu_dependent_token_idx_forward,
        # 'store_dependent_token_idx_backward': cpu_dependent_token_idx_backward,
        'sid': sid, 
        'mask_left_boundry': [1]* bz*num_samples,
        'mask_right_boundry': torch.tensor([len(_)+1 for _ in b_inputs.raw_tokens]).to(device)
    }


def collateFnForOpenAIQuery(b_inputs: List[DictObj], tokenizer, embeddings, device='cuda:0', num_samples = 8):

    bz = len(b_inputs)
    # b_inputs: ListObj = ListObj(b_inputs)

    output = []

    for item in b_inputs:
        raw_tokens = item.item_ud._raw_tokens
        head_id = item.x_id
        dependent_id = item.y_id
        head_tok = item.x_token.lower()
        dependent_tok = item.y_token.lower()
        if not (embeddings.__contains__(head_tok) and embeddings.__contains__(dependent_tok)): continue
        # print(head_tok)
        head_neighbor = embeddings.nearest_neighbors(head_tok, num_samples)
        dependent_neighbor = embeddings.nearest_neighbors(dependent_tok, num_samples)

        # output.append(DictObj(
        #     {
        #         'original_text': ' '.join(raw_tokens).lower(),
        #         'text': ' '.join(raw_tokens).lower(),
        #         'text_with_correction': ' '.join(raw_tokens).lower(),
        #         'sid': item.sid,
        #         'ud_head_id': item.x_id+1,
        #         'ud_dependent_id': item.y_id+1
        #     }
        # ))

        for head_replacement in head_neighbor:
            for dependent_replacement in dependent_neighbor:
                text = raw_tokens.copy()
                text[head_id] = head_replacement
                text[dependent_id] = dependent_replacement
                text_with_correction = raw_tokens.copy()
                text_with_correction[head_id] = '{} (as opposed to {})'.format(head_replacement, text_with_correction[head_id])
                text_with_correction[dependent_id] = '{} (as opposed to {})'.format(dependent_replacement, text_with_correction[dependent_id])
                output.append(DictObj(
                    {
                        'original_text': ' '.join(raw_tokens).lower(),
                        'text': ' '.join(text).lower(),
                        'text_with_correction': ' '.join(text_with_correction).lower(),
                        'sid': item.sid,
                        'ud_head_id': item.x_id+1,
                        'ud_dependent_id': item.y_id+1
                    }
                ))

    return output


def collateFnForMIViaImportanceSampling(b_inputs: List[DictObj], tokenizer, model_output_logit_size, device='cuda:0', brown_cluster_pack=None, num_chains = 1, upos_masks = {}):
    # currently assuming that both x and y are single token words
    assert len(upos_masks)>0
    vocab_size = len(tokenizer.vocab)

    bz = len(b_inputs)
    b_inputs: ListObj = ListObj(b_inputs)

    # print([' '.join(rts) for rts in b_inputs.raw_tokens])
    # print(b_inputs.x_id)
    # print(b_inputs.y_id)

    b_encodings = tokenizer([' '.join(rts) for rts in b_inputs.raw_tokens], padding=True, return_tensors='pt')
    b_subtoken_alignment = b_inputs.subtoken_alignment


    # canonical x/y_idx 
    # b_canonical_x_idx, b_canonical_y_idx = [], []
    # for subtoken_alignment, x_id, y_id in zip(b_subtoken_alignment, b_inputs.x_id, b_inputs.y_id):
    #     canonical_x_id, canonical_y_id = (x_id, y_id) if x_id<y_id else (y_id, x_id)
    #     b_canonical_x_idx.append(torch.tensor(subtoken_alignment[canonical_x_id+1]))
    #     b_canonical_y_idx.append(torch.tensor(subtoken_alignment[canonical_y_id+1]))
    # # print(b_canonical_x_idx)
    # b_canonical_x_idx = torch.nn.utils.rnn.pad_sequence(b_canonical_x_idx, batch_first=True).long()
    # b_canonical_y_idx = torch.nn.utils.rnn.pad_sequence(b_canonical_y_idx, batch_first=True).long()
    # b_canonical_x_idx_mask = b_canonical_x_idx !=0
    # b_canonical_y_idx_mask = b_canonical_y_idx !=0




    # x/y_idx contains the subtoken index
    b_x_idx = [subtoken_alignment[x_id+1] for subtoken_alignment, x_id in zip(b_subtoken_alignment, b_inputs.x_id)]  # note that the x_id is the token id starting from 0 while the real tokens in the subtoken_alignment starts from 1
    b_y_idx = [subtoken_alignment[y_id+1] for subtoken_alignment, y_id in zip(b_subtoken_alignment, b_inputs.y_id)]  
    b_xy_idx = [ x_idx+y_idx for x_idx, y_idx in zip(b_x_idx, b_y_idx)]

    # packing them into a tensor
    b_x_idx = [torch.tensor(x_idx) for x_idx in b_x_idx]
    b_y_idx = [torch.tensor(y_idx) for y_idx in b_y_idx]
    b_xy_idx = [torch.tensor(xy_idx) for xy_idx in b_xy_idx]
    b_x_idx = torch.nn.utils.rnn.pad_sequence(b_x_idx, batch_first=True).long()
    b_y_idx = torch.nn.utils.rnn.pad_sequence(b_y_idx, batch_first=True).long()
    b_xy_idx = torch.nn.utils.rnn.pad_sequence(b_xy_idx, batch_first=True).long()

    b_x_idx_mask = b_x_idx != 0.
    b_y_idx_mask = b_y_idx != 0.
    b_xy_idx_mask = b_xy_idx != 0.

    
    b_x_mask_opening_piece = [torch.tensor([True]+[False]*(len(x_idx)-1)) for x_idx in b_x_idx]
    b_y_mask_opening_piece = [torch.tensor([True]+[False]*(len(y_idx)-1)) for y_idx in b_y_idx]
    b_xy_mask_opening_piece = [torch.cat([x_mask, y_mask], dim=0) for x_mask, y_mask in zip(b_x_mask_opening_piece, b_y_mask_opening_piece)]
    b_x_mask_opening_piece = torch.nn.utils.rnn.pad_sequence(b_x_mask_opening_piece, batch_first=True).bool()
    b_y_mask_opening_piece = torch.nn.utils.rnn.pad_sequence(b_y_mask_opening_piece, batch_first=True).bool()
    b_xy_mask_opening_piece = torch.nn.utils.rnn.pad_sequence(b_xy_mask_opening_piece, batch_first=True).bool()

    b_x_mask_continuing_piece = [torch.tensor([False]+[True]*(len(x_idx)-1)) for x_idx in b_x_idx]
    b_y_mask_continuing_piece = [torch.tensor([False]+[True]*(len(y_idx)-1)) for y_idx in b_y_idx]
    b_xy_mask_continuing_piece = [torch.cat([x_mask, y_mask], dim=0) for x_mask, y_mask in zip(b_x_mask_continuing_piece, b_y_mask_continuing_piece)]
    b_x_mask_continuing_piece = torch.nn.utils.rnn.pad_sequence(b_x_mask_continuing_piece, batch_first=True).bool()
    b_y_mask_continuing_piece = torch.nn.utils.rnn.pad_sequence(b_y_mask_continuing_piece, batch_first=True).bool()
    b_xy_mask_continuing_piece = torch.nn.utils.rnn.pad_sequence(b_xy_mask_continuing_piece, batch_first=True).bool()

    # b_x_upos_ids = [upos2uposid[upos] for upos in b_inputs.x_upos]
    # b_y_upos_ids = [upos2uposid[upos] for upos in b_inputs.y_upos]

    b_x_upos_mask_ids = torch.nn.utils.rnn.pad_sequence([upos_masks[upos] for upos in b_inputs.x_upos], batch_first=True)
    b_y_upos_mask_ids = torch.nn.utils.rnn.pad_sequence([upos_masks[upos] for upos in b_inputs.y_upos], batch_first=True)        

    # enabling mask
    b_x_upos_mask = torch.zeros(bz, vocab_size).scatter(1, b_x_upos_mask_ids, 1).bool().unsqueeze(1).repeat(1, b_x_idx.size(1), 1)
    b_x_upos_mask[:,:,  0] = False
    b_y_upos_mask = torch.zeros(bz, vocab_size).scatter(1, b_y_upos_mask_ids, 1).bool().unsqueeze(1).repeat(1, b_y_idx.size(1), 1)
    b_y_upos_mask[:,:,  0] = False

    assert torch.all(torch.any(b_x_upos_mask, dim=2))
    assert torch.all(torch.any(b_y_upos_mask, dim=2))

    


    # Masks for encoding $S_{XY}$

    def brown_cluster_id_2_conditional_mask(brown_cluster_id2cluster, cid, vocab):
        """This function generates the conditional mask from the brown cluster id

        Args:
            brown_cluster_id2cluster (_type_): _description_
            cid (_type_): _description_
            vocab (_type_): _description_

        Returns:
            _type_: _description_
        """        
        if cid>=0:
            cluster = brown_cluster_id2cluster[cid]
            conditional_mask_indices = torch.tensor([vocab[word] for word in cluster if word in vocab.keys()])
            # print(conditional_mask_indices)
            conditional_mask = torch.zeros(model_output_logit_size).scatter(0, conditional_mask_indices, value=1)
        else:
            conditional_mask = torch.ones(model_output_logit_size)
        return conditional_mask


    flag_brown_clustering = brown_cluster_pack is not None
    if flag_brown_clustering:
        brown_clusters_word2id, brown_clusters_id2cluster = brown_cluster_pack
        b_x_conditional_mask, b_y_conditional_mask = [],  []
        # b_canonical_x_conditional_mask, b_canonical_y_conditional_mask = [],  []
        tokenizer_vocab = tokenizer.get_vocab()
        for raw_tokens, x_id, y_id in  zip(b_inputs.raw_tokens, b_inputs.x_id, b_inputs.y_id):
            x_tok, y_tok = raw_tokens[x_id], raw_tokens[y_id]
            x_cluster_id, y_cluster_id = int(brown_clusters_word2id.get(x_tok, '-1')), int(brown_clusters_word2id.get(y_tok, '-1'))
            # print('x cluster id', x_cluster_id)
            # print(tokenizer_vocab[x_tok])
            x_conditional_mask = brown_cluster_id_2_conditional_mask(brown_clusters_id2cluster, x_cluster_id, tokenizer_vocab)
            # print(tokenizer_vocab[y_tok])
            y_conditional_mask = brown_cluster_id_2_conditional_mask(brown_clusters_id2cluster, y_cluster_id, tokenizer_vocab)
            b_x_conditional_mask.append(x_conditional_mask)
            b_y_conditional_mask.append(y_conditional_mask)
            # print(x_conditional_mask)
            # if x_id<y_id:
            #     b_canonical_x_conditional_mask.append(x_conditional_mask)
            #     b_canonical_y_conditional_mask.append(y_conditional_mask)
            # else:
            #     b_canonical_x_conditional_mask.append(y_conditional_mask)
            #     b_canonical_y_conditional_mask.append(x_conditional_mask)

        b_x_conditional_mask = torch.stack(b_x_conditional_mask, dim=0).view(bz, 1, model_output_logit_size)        
        b_y_conditional_mask = torch.stack(b_y_conditional_mask, dim=0).view(bz, 1, model_output_logit_size)        
        # b_canonical_x_conditional_mask = torch.stack(b_canonical_x_conditional_mask, dim=0).view(bz, 1, model_output_logit_size)        
        # b_canonical_y_conditional_mask = torch.stack(b_canonical_y_conditional_mask, dim=0).view(bz, 1, model_output_logit_size)    
        # print(b_x_conditional_mask.sum(-1))    

    else:
        b_x_conditional_mask = torch.ones(bz, b_x_idx.size(1), model_output_logit_size)
        b_y_conditional_mask = torch.ones(bz, b_y_idx.size(1), model_output_logit_size)
        # b_canonical_x_conditional_mask = torch.ones(bz, b_x_idx.size(1), model_output_logit_size)
        # b_canonical_y_conditional_mask = torch.ones(bz, b_y_idx.size(1), model_output_logit_size)

    b_x_tokens = list(map(lambda raw, id: raw[id], b_inputs.raw_tokens, b_inputs.x_id))
    b_y_tokens = list(map(lambda raw, id: raw[id], b_inputs.raw_tokens, b_inputs.y_id))
    b_sent = list(map(lambda x: ' '.join(x), b_inputs.raw_tokens))



    return DictObj({
        'b_inputs': b_inputs,
        'b_encodings': b_encodings.to(device),
        'b_x_idx': b_x_idx.to(device),
        'b_x_idx_mask': b_x_idx_mask.to(device),
        'b_y_idx': b_y_idx.to(device),
        'b_y_idx_mask': b_y_idx_mask.to(device),
        'b_xy_idx': b_xy_idx.to(device),
        'b_xy_idx_mask': b_xy_idx_mask.to(device),
        'b_x_conditional_mask': b_x_conditional_mask.to(device),
        'b_y_conditional_mask': b_y_conditional_mask.to(device),
        # 'b_canonical_x_idx': b_canonical_x_idx.to(device),
        # 'b_canonical_y_idx': b_canonical_y_idx.to(device),
        # 'b_canonical_x_idx_mask': b_canonical_x_idx_mask.to(device),
        # 'b_canonical_y_idx_mask': b_canonical_y_idx_mask.to(device),
        # 'b_canonical_x_conditional_mask': b_canonical_x_conditional_mask.to(device),
        # 'b_canonical_y_conditional_mask': b_canonical_y_conditional_mask.to(device),
        'sid': b_inputs.sid,
        'b_x_token_id': b_inputs.x_id,
        'b_y_token_id': b_inputs.y_id,
        'b_x_tokens': b_x_tokens,
        'b_y_tokens': b_y_tokens,
        'b_sent': b_sent,

        'b_x_mask_opening_piece': b_x_mask_opening_piece,
        'b_y_mask_opening_piece': b_y_mask_opening_piece,
        'b_xy_mask_opening_piece': b_xy_mask_opening_piece,

        'b_x_mask_continuing_piece': b_x_mask_continuing_piece,
        'b_y_mask_continuing_piece': b_y_mask_continuing_piece,
        'b_xy_mask_continuing_piece': b_xy_mask_continuing_piece,

        'b_x_upos_mask': b_x_upos_mask.to(device),
        'b_y_upos_mask': b_y_upos_mask.to(device)

    })