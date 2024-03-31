
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Type, Set, Tuple, Any
from random import random, choices
import json
from datetime import datetime
import h5py
import unicodedata
from transformers import PreTrainedTokenizer
import networkx.algorithms.isomorphism as iso
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from .auxobjs import DictObj
import random


em = iso.numerical_edge_match("weight", 1)
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFKD', s)
                  if unicodedata.category(c) != 'Mn')

def normalize_double_quotation(s:str) -> str:
    return s.replace("''", '"').replace("``", '"')

def align_tokenized_terminals_wordpiece(tokenized: List[str], terminals: List[str], tokenizer: PreTrainedTokenizer):
    excessive_tokens = {
        '<sep>',
        '<cls>'
    }

    tokenized_idx, tokenized = zip(*[(id, cleaned_item) for id, item in enumerate(tokenized) if len(cleaned_item:=item.replace('▁', ''))>0 and cleaned_item not in excessive_tokens])
    ptr_tok = 0
    ptr_trm = 0
    groups = [[]]
    while ptr_trm<len(terminals) or ptr_tok<len(tokenized):
        tok = tokenized[ptr_tok]
        ref_tok = terminals[min(ptr_trm, len(terminals)-1)]
        if ref_tok.startswith(tok):
            groups.append([tokenized_idx[ptr_tok]])
            ptr_trm+=1
        elif tok == tokenizer.unk_token:
            groups.append([tokenized_idx[ptr_tok]])
            ptr_trm+=1
        else:
            assert ptr_tok<len(tokenized)
            groups[-1].append(tokenized_idx[ptr_tok])
        ptr_tok+=1
    groups.append([])
    return groups

def align_tokenized_terminals_gpt(tokenized: List[str], terminals: List[str], tokenizer: PreTrainedTokenizer):
    excessive_tokens = {
        '<sep>',
        '<cls>'
    }

    tokenized_idx, tokenized = zip(*[(id, item) for id, item in enumerate(tokenized) ])
    ptr_tok = 0
    ptr_trm = 0
    groups = [[]]
    while ptr_trm<len(terminals) or ptr_tok<len(tokenized):
        tok = tokenized[ptr_tok]
        # print(ptr_tok, tok)
        assert ptr_tok==0 or (ptr_tok!=0 and tok!= tokenizer.unk_token)
        ref_tok = terminals[min(ptr_trm, len(terminals)-1)]
        if tok.startswith('Ġ'):
            groups.append([tokenized_idx[ptr_tok]])
            ptr_trm+=1
        # elif tok == tokenizer.unk_token:
        #     groups.append([tokenized_idx[ptr_tok]])
        #     ptr_trm+=1
        else:
            assert ptr_tok<len(tokenized)
            groups[-1].append(tokenized_idx[ptr_tok])
        ptr_tok+=1
    groups.append([ptr_tok])
    return groups
    
def align_tokenized_terminals_sentencepiece(tokenized: List[str], terminals: List[str], tokenizer: PreTrainedTokenizer):
    excessive_tokens = {
        '<s>',
        '</s>',
        '<sep>',
        '<cls>',
        '[SEP]',
        '[CLS]'
    }

    terminals = terminals.copy()
    terminals.append('@[EOS]@')

    clean_tok = lambda x: x if not x.startswith('##') else x[2:]

    tokenized_idx, tokenized = zip(*[(id, cleaned_item) for id, item in enumerate(tokenized) if len(cleaned_item:=item.replace('▁', ''))>0 and cleaned_item not in excessive_tokens])

    ptr_tok = 0
    ptr_trm = 0
    groups = [[]]
    current_ref_tok = ''
    while ptr_trm<=len(terminals) and ptr_tok<len(tokenized):
        tok = tokenized[ptr_tok].lower()
        # ref_tok = terminals[min(ptr_trm, len(terminals)-1)].lower()
        ref_tok = terminals[ptr_trm].lower()
        # next_ref_tok = terminals[min(ptr_trm+1, len(terminals)-1)]
        assert tok != tokenizer.unk_token
        # print(terminals[ptr_trm], tokenized[ptr_tok], 'current ref tok', current_ref_tok)
        if current_ref_tok.startswith(clean_tok(tok)):
            assert ptr_tok<len(tokenized)
            groups[-1].append(tokenized_idx[ptr_tok])
            assert len(current_ref_tok) >= len(clean_tok(tok))
            current_ref_tok = current_ref_tok[len(clean_tok(tok)):]
        elif ref_tok.startswith(tok):
            groups.append([tokenized_idx[ptr_tok]])
            current_ref_tok = ref_tok[len(clean_tok(tok)):]
            ptr_trm+=1
        # elif tok == tokenizer.unk_token.lower():
            # groups.append([tokenized_idx[ptr_tok]])
            # ptr_trm+=1
        else:
            assert ptr_tok<len(tokenized)
            groups[-1].append(tokenized_idx[ptr_tok])
        ptr_tok+=1
    # groups.append()
    # print(groups)
    # print()
    # groups.append([])
    return groups



map_tokenized_terminals_alignment = {
    'sentencepiece': align_tokenized_terminals_sentencepiece,
    # 'wordpiece': align_tokenized_terminals_wordpiece
}



def generate_window(window_size, lb, ub, flag_compute_missing_vinfo_from_cache = False, h5dst_handler = None):
    output = []
    # print(lb, ub)
    for i in range(lb, ub):
        for j in range(max(lb, i-window_size), min(ub, i+window_size)):
            if i == j: continue
            if flag_compute_missing_vinfo_from_cache and not np.isnan(h5dst_handler[i-1, j-1]): continue
            assert i>=lb and i<ub
            assert j>=lb and j<ub
            output.append((i, j))
    # print(output)
    return output
def generate_neg_window(window_size, lb, ub, flag_compute_missing_vinfo_from_cache = False, h5dst_handler = None):
    output = []
    for i in range(lb, ub):
        for j in range(lb, ub):
            if j >=max(lb, i-window_size+1) and j<min(ub, i+window_size-1): continue
            if flag_compute_missing_vinfo_from_cache and not np.isnan(h5dst_handler[i-1, j-1]): continue
            assert i>=lb and i<ub
            assert j>=lb and j<ub
            output.append((i, j))
    return output
def generate_pair_with_exact_dist(dist, lb, ub, flag_compute_missing_vinfo_from_cache = False, h5dst_handler = None):
    assert not flag_compute_missing_vinfo_from_cache
    output = []
    for i in range(lb, ub):
        if i-dist>=lb and i-dist<ub:
            output.append((i, i-dist))
        if lb<= i+dist and i+dist<ub:
            output.append((i, i+dist))
    return output



@dataclass
class UDSentence:
    # tokens: List[List[str]]
    ID_IDX: int = 0
    WORD_FORM_IDX: int = 1
    UPOS_IDX: int = 3
    HEAD_IDX: int = 6
    REL_IDX: int = 7
    def __init__(self, groups, h5dst_handler = None, h5dst_idx = None, h5dst_partition = None):
        if groups[0].startswith('!#'):
            # print('meta data: ', groups[0][2:].strip())
            self.meta_data = json.loads(groups[0][2:].strip())
            groups = [item for item in groups[1:] if not item.startswith('#')]
        else:
            groups = [item for item in groups if not item.startswith('#')]
        self.tokens: List[str] = [ele for item in groups if '-' not in (ele := item.split('\t'))[0] and '.' not in ele[0]]
        # self._raw_tokens = [normalize_double_quotation(strip_accents(tok[self.WORD_FORM_IDX].strip())) for tok in self.tokens]
        # temporary disable accent stripping as it conflicts with GPT's token-start marker
        self._raw_tokens = [normalize_double_quotation(tok[self.WORD_FORM_IDX].strip()) for tok in self.tokens]
        self._upos = [tok[self.UPOS_IDX] for tok in self.tokens]
        self.dep_head_pairs = [(int(tok[self.ID_IDX]), {k: int(v) for k, v in json.loads(tok[self.HEAD_IDX]).items()}, tok[self.REL_IDX], tok[self.UPOS_IDX]) for tok in self.tokens]# if int(json.loads(tok[self.HEAD_IDX])['ud']) !=0]
        self.dep_head_pairs_dict = {(int(tok[self.ID_IDX]), json.loads(tok[self.HEAD_IDX]).get('ud', -1)): tok[self.REL_IDX] for tok in self.tokens}# if int(json.loads(tok[self.HEAD_IDX])['ud']) !=0]

        self._raw = ' '.join(self._raw_tokens)
        self.syn_rels = set([tok[self.REL_IDX] for tok in self.tokens])
        # self.h5dst_idx = h5dst_idx
        self.h5dst_idx = str(self.meta_data['sid'])
        self.overlength = None
        # if h5dst_idx is not None and h5dst_handler is not None:
        #     if h5dst_idx not in h5dst_handler.keys():
        #         init_arr = np.empty([len(self._raw_tokens)]*2)
        #         init_arr[:] = np.nan
        #         self.h5dst = h5dst_handler.create_dataset(h5dst_idx, data=init_arr)
        #     else:
        #         self.h5dst = h5dst_handler[h5dst_idx]
        # else:
        #     self.h5dst=None

        if h5dst_partition is not None and h5dst_handler is not None:
            self.h5dst = h5dst_handler[self.h5dst_idx][h5dst_partition]
        self.h5dst_hit = None
        # print('h5dst: ', self.h5dst)
        self.g_mst_cache = nx.DiGraph()
        self.mst_cache = nx.DiGraph()

    def sample_supervised_dependencies(self:UDSentence, synrel2id):
        # only shows the original data of dependencies        
        
        # head_indicator_mtx = [torch.nn.functional.one_hot(item[1]['ud'], num_classes = len(self.tokens)+1) for item in self.dep_head_pairs] # -> an indicator mtx of shape (seq, seq+1)
        # dep_rels = [torch.nn.functional.one_hot(synrel2id.get(item[2], 0)) for item in self.dep_head_pairs] # (seq, class_rels)

        # print(self.dep_head_pairs)
        # print([(item[0], item[1]['ud']) for item in self.dep_head_pairs])
        # print(self._raw)
        # print(self._raw_tokens)

        return {
            'deps': [(item[0], item[1]['ud'], int(synrel2id.get(item[2], 0))) for item in self.dep_head_pairs],
            # 'dep_rels': [int(synrel2id.get(item[2], 0)) for item in self.dep_head_pairs],
        }

    @staticmethod
    def save_vinfos(self:UDSentence, vinfos, lr_pairs: List[Tuple[Tuple[int, int], Any]]):
        # raise NotImplementedError
        for pair, weight in zip(lr_pairs, vinfos):
            # print("saving to ", pair)
            self.h5dst[pair[0][0]-1, pair[0][1]-1] = weight
    
    @staticmethod
    def get_vinfo_matrix(self: UDSentence, baseline = None):
        # print(baseline)
        seq_len: int = self.h5dst.shape[0]
        h5dst = self.h5dst[:]
        if baseline is None: return h5dst
        for i in range(seq_len):
            for j in range(seq_len):
                if i==j: continue
                h5dst[i, j] -= baseline.forward(self, i, j, h5dst[i, j])
        return h5dst



    @staticmethod
    def construct_graph_from_cache(self: UDSentence, strategy:Dict[str, str]):
        # how to construct the graph here?
        seq_len: int = self.h5dst.shape[0]
        g: nx.DiGraph = nx.DiGraph()
        g.add_nodes_from([(id, {'tok':tok}) for id, tok in enumerate(self._raw_tokens)])

        # strategy_type = strategy['strategy_type']
        h5dst = UDSentence.get_vinfo_matrix(self, baseline=strategy.get('baseline', None))
        edges = self.sample_ij_pairs(self, strategy, seq_len, 0, seq_len)
        edges = [(item[0], item[1], {'weight': h5dst[item[0], item[1]]})for item in edges]
        assert not any(map(lambda x: np.isnan(x), [item[2]['weight'] for item in edges]))
        g.add_edges_from(edges)
    
        if nx.is_isomorphic(self.g_mst_cache, g, edge_match=em):
            self.g_mst_cache = g
            g = self.mst_cache
        else:
            self.g_mst_cache = g
            g: nx.DiGraph = nx.algorithms.tree.branchings.Edmonds(g).find_optimum(preserve_attrs=True, kind = min_max)
            self.mst_cache=g

        return g

    @staticmethod 
    def sample_ij_pairs(self, strategy:Dict[str, Union[str, int]], seq_len: int, lb = -1, ub = -1):
        # !! 1 -  seq_len-1 
        # assert strategy in self.ALLOWED_SAMPLING_STRATEGY
        strategy_type = strategy['strategy_type']
        # print(strategy)
        if strategy_type in ['rand', 'mst']:
            assert lb!=-1 and ub!=-1
            flag_compute_missing_vinfo_from_cache = strategy.get('compute_missing_vinfo', False)
            # ret = []
            if 'window_size' in strategy.keys():
                max_ij_gap = strategy.get('window_size', 8)
                output = generate_window(max_ij_gap, lb, ub, flag_compute_missing_vinfo_from_cache, self.h5dst if flag_compute_missing_vinfo_from_cache else None)
                # ret.extend(output)
                # print(output)
                return output
            if 'outside_of_window_size' in strategy.keys():
                max_ij_gap = int(strategy['outside_of_window_size'])
                output = generate_neg_window(max_ij_gap, lb, ub)
                # ret.extend(output)
                return output
            if 'exact_distance' in strategy.keys():
                dist = int(strategy['exact_distance'])
                output = generate_pair_with_exact_dist(dist, lb, ub)
                return output
        elif strategy_type == 'non-syndeps':
            ud_section = strategy.get('ud-section', 'ud')
            # print('dep_head_pair', self.dep_head_pairs)
            gold_deps = [(item[0], item[1][ud_section]) for item in self.dep_head_pairs] #if item[1]['ud'] != 0]
            assert lb!=-1 and ub!=-1
            if 'window_size' in strategy.keys():
                max_ij_gap = strategy.get('window_size', 8)
                output = generate_window(max_ij_gap, lb, ub, False, self.h5dst if False else None)
                # print("random:", output)
                # print("gold", gold_deps)
                output = [arc for arc in output if arc not in gold_deps]
                # print('non-syndeps', output)
                return output
        elif strategy_type == 'syndeps':
            ud_section = strategy.get('ud-section', 'ud')
            # print('dep_head_pair', self.dep_head_pairs)
            ret = [(item[0], item[1][ud_section], item[2], item[3]) for item in self.dep_head_pairs] #if item[1]['ud'] != 0]
            # print('ret', ret)
            if 'relation_list_filter' in strategy.keys():
                target_relations = strategy['relation_list_filter']
                ret = [item for item in ret if item[2] in target_relations]
            if strategy.get('remove_punct', False):
                ret = [item for item in ret if item[3]!='PUNCT']
            if 'relation_length_filter_gt' in strategy.keys():
                target_length = strategy['relation_length_filter_gt']
                ret = [item for item in ret if abs(item[0]-item[1])>=target_length]
            if 'relation_length_filter' in strategy.keys():
                target_length = strategy['relation_length_filter']
                ret = [item for item in ret if abs(item[0]-item[1])==target_length]
            # if 'reverse' in strategy.keys() and strategy['reverse']:
                # ret = [(item[1], item[0], item[2]) for item in ret]
            return [item[:2] for item in ret]
        elif strategy_type == 'mst':
            raise NotImplementedError
            ret = []
            flag_compute_missing_vinfo_from_cache = strategy.get('compute_missing_vinfo', False)
            if 'window_size' in strategy.keys():
                max_ij_gap = strategy['window_size']
                for i in range(1, seq_len-1):
                    for j in range(max(1, i-max_ij_gap), min(seq_len-1, i+max_ij_gap)):
                        if i == j: continue
                        if flag_compute_missing_vinfo_from_cache and not np.isnan(self.h5dst[i-1, j-1]): continue
                        # self.data.append((i, j, item_id))
                        assert i>=1 and i<seq_len-1
                        assert j>=1 and j<seq_len-1
                        ret.append((i, j))
                return ret
        elif strategy_type == 'one-way-branching':
            right_branching = strategy.get('right_branching', True)
            ret = [(n, n+1) if right_branching else (n+1, n) for n in range(lb, ub)]
        else:
            raise ValueError
        
        return ret

    @staticmethod
    def sample_directed_graph(self:UDSentence, required_samples: int = 10):
        raise NotImplementedError
        # from random import random
        # print(list(range(len(self._raw_tokens))))
        roots = choices(list(range(len(self._raw_tokens))), k = required_samples)
        return [nx.dfs_tree(self.LERW_sampler.sample(nx.complete_graph(len(self._raw_tokens))).to_directed(), roots[n]) for n in range(required_samples)] 
    
    @staticmethod
    def sample_vinfo_graph(self: UDSentence, strategy):
        raise NotImplementedError
        seq_len: int = len(self.tokens)
        strategy_type = strategy['strategy_type']
        if strategy_type == 'rand':
            ret = []
            if 'window_size' in strategy.keys():
                max_ij_gap = strategy['window_size']
                for i in range(1, seq_len+1):
                    for j in range(max(1, i-max_ij_gap), min(seq_len+1, i+max_ij_gap)):
                        if i == j: continue
                        # self.data.append((i, j, item_id))
                        ret.append((i, j))
                return ret
        else:
            raise NotImplementedError
    


class CoNLLDataset(object):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, SentenceClass: UDSentence, vinfo_hdf5_path:Optional[str] = None, vinfo_cache_ro:bool=True, cache_partition: str = 'adjusted_rank', max_token_count = 256):


        # WARNING: THE SENTENCE LENGTH IS NOW LIMITED TO 30 PIECES
        with open(file_path, encoding="utf-8-sig") as f:
            lines = f.read().splitlines()

        self.examples = []
        gcontainer = []
        self.sentence_class: Type[UDSentence] = SentenceClass

        # assert vinfo_hdf5_path is not None
        if vinfo_hdf5_path is None:
            self.vinfo_hdf5_cache:Optional[Any] = None
        else:
            os.makedirs(os.path.dirname(vinfo_hdf5_path), exist_ok=True)
            self.vinfo_hdf5_cache: Optional[Any] = h5py.File(vinfo_hdf5_path, 'r') 

        example_cnt: int = 0
        for line in lines:
            if line != '':
                # print()
                gcontainer.append(line)
            else:
                if len(gcontainer)<max_token_count+1: #otherwise just drop the tokens, including the root token
                    self.examples.append(SentenceClass(gcontainer, self.vinfo_hdf5_cache, h5dst_partition=cache_partition))
                    example_cnt+=1

                # self.examples[-1]
                gcontainer = []
        
        self.syn_rels = sorted(list(set().union(*[item.syn_rels for item in self.examples])))
        self.synrel2id = {'UNK':0, **{item: id+1 for id, item in enumerate(self.syn_rels)}}
        self.id2synrel = [rel for rel, id in self.synrel2id.items()] # might be risky
        # self.sem_rels = set().union(*[item.sem_rels for item in self.examples])
        print("getting batch encodings")
        self.tokenizer: PreTrainedTokenizer = tokenizer



    def sample_sentences(self):
        self.data = []
        for item_ud in self.examples:
            self.data.append(
                DictObj({
                'item_ud': item_ud,
                'raw_tokens': item_ud.raw_tokens
                })
            )

    @staticmethod
    def sample_openai_ij_pairs(self: UDSentence, strategy:Dict[str, Any] = {'strategy_type': 'syn-deps'}, num_samples: int  = 32):
        self.data = []
        for item_id in range(len(self.examples)):
            item_ud = self.examples[item_id]
            item_encoding = self.tokenizer(item_ud._raw)
            seq_len = len(item_encoding['input_ids'])
            
            # print(seq_len)
            if item_ud.overlength is None:
                if seq_len>96:
                    # self.sent_separator_idx.append(separator_cnt)
                    item_ud.overlength = True
                    continue
                else:
                    item_ud.overlength = False
            if item_ud.overlength: continue


            tokenized = self.tokenizer.convert_ids_to_tokens(item_encoding['input_ids'])
            subtoken_alignment = align_tokenized_terminals_sentencepiece(tokenized, item_ud._raw_tokens, self.tokenizer)
            edges = UDSentence.sample_ij_pairs(item_ud, strategy, len(item_ud._raw.split(' '))+2, lb = 1, ub = len(item_ud._raw_tokens)+1)
            # print(edges)
            self.data.extend([
                DictObj({
                    'item_ud': item_ud,
                    'raw_tokens': item_ud._raw_tokens,
                    'x_id': item[1]-1,
                    'y_id': item[0]-1,
                    'x_token': item_ud._raw_tokens[item[1]-1],
                    'y_token': item_ud._raw_tokens[item[0]-1],
                    'sid': item_ud.meta_data['sid'],
                    'subtoken_alignment': subtoken_alignment,
                    'x_upos': item_ud._upos[item[1]-1],
                    'y_upos': item_ud._upos[item[0]-1] 
                })
                for item in edges if len(subtoken_alignment[item[0]])==1 and len(subtoken_alignment[item[1]])==1 and item[0]>=item[1]
            ])

    def save_vinfos(self: CoNLLDataset, vinfos):
        for n in range(len(self.sent_separator_idx)-1):
            l_id = self.sent_separator_idx[n]
            r_id = self.sent_separator_idx[n+1]
            item_ud:UDSentence = self.examples[n]
            item_ud.save_vinfos(item_ud, vinfos[l_id:r_id], self.data_word[l_id: r_id])

    def generate_empty_gpt_holder(self, num_samples = 32):
        holder = {}
        for item_ud in self.examples:
            sid = item_ud.meta_data['sid']
            token_seqlen = len(item_ud._raw_tokens)
            assert sid not in holder.keys()
            holder[sid] = {
                'vinfo': np.full([token_seqlen, token_seqlen, num_samples], -10.),
                'adjusted_rank': np.full([token_seqlen, token_seqlen, num_samples], np.inf),
                'absolute_rank': np.full([token_seqlen, token_seqlen, num_samples], np.inf)
                }
        return holder

    def sample_GPT_probing_data(self, strategy = {}):
        self.data = []
        for item_ud in self.examples:
            item_ud: UDSentence 
            raw = ' '.join([self.tokenizer.bos_token] + item_ud._raw_tokens)
            item_encoding = self.tokenizer(raw)
            tokenized_subtoken = item_encoding['input_ids']
            if len(tokenized_subtoken)>160 or len(item_ud._raw_tokens)<2:# or 'http' in item_ud._raw:
                continue
            # tokenized_token = item_ud._upos

            tokenized = self.tokenizer.convert_ids_to_tokens(item_encoding['input_ids'])
            subtoken_alignment = align_tokenized_terminals_gpt(tokenized, item_ud._raw_tokens, self.tokenizer)

            # gold_dependencies = item_ud.sample_supervised_dependencies(synrel2id)
            # gold_dependency_set = {dep[:2]: dep[2] for dep in gold_dependencies}

            self.data.extend([DictObj({
                'raw': ' '.join([self.tokenizer.bos_token] + item_ud._raw_tokens[:dep_id]),
                'raw_target': ' '.join([self.tokenizer.bos_token] + item_ud._raw_tokens[:dep_id+1]),
                'subtoken_alignment': subtoken_alignment,
                'sid': item_ud.meta_data['sid'],
                'dependent_id': dep_id
            }) for dep_id in range(1, len(item_ud._raw_tokens)) if len(subtoken_alignment[dep_id+1])==1])
        self.data.sort(key = lambda x: len(x.raw.split(' ')), reverse=False)


    def sample_GPT_probing_wordclass_data(self, strategy = {}, upos_word_map = {}, num_samples = 1):
        assert len(upos_word_map)>0
        self.data = []
        shuffle_pointer = {k:0 for k in upos_word_map.keys()}
        shuffle_boundary = {k:len(v) for k, v in upos_word_map.items()}
        for item_ud in tqdm(self.examples):
            item_ud: UDSentence 
            raw = ' '.join([self.tokenizer.bos_token] + item_ud._raw_tokens)
            item_encoding = self.tokenizer(raw)
            tokenized_subtoken = item_encoding['input_ids']
            if len(tokenized_subtoken)>160 or len(item_ud._raw_tokens)<2:# or 'http' in item_ud._raw:
                continue
            # tokenized_token = item_ud._upos

            tokenized = self.tokenizer.convert_ids_to_tokens(item_encoding['input_ids'])
            subtoken_alignment = align_tokenized_terminals_gpt(tokenized, item_ud._raw_tokens, self.tokenizer)

            # gold_dependencies = item_ud.sample_supervised_dependencies(synrel2id)
            # gold_dependency_set = {dep[:2]: dep[2] for dep in gold_dependencies}
            for dep_id in range(1, len(item_ud._raw_tokens)):
                if len(subtoken_alignment[dep_id+1])!=1:
                    continue
                for head_id in range(dep_id):

                    head_sameclass_samples = random.choices(upos_word_map[item_ud._upos[head_id]], k=num_samples)
                    raw_head_sameclass = [[self.tokenizer.bos_token] + item_ud._raw_tokens[:head_id] + [head_sameclass_samples[_]]+item_ud._raw_tokens[head_id+1:dep_id] for _ in range(num_samples)]

                    head_random_samples = random.choices(upos_word_map['clean_all'], k=num_samples)#[shuffle_pointer['clean_all']:shuffle_pointer['clean_all']+num_samples]
                    raw_head_random = [[self.tokenizer.bos_token] + item_ud._raw_tokens[:head_id] + [head_random_samples[_]]+item_ud._raw_tokens[head_id+1:dep_id] for _ in range(num_samples)]
                    
                    target_sameclass_samples = random.choices(upos_word_map[item_ud._upos[dep_id]], k=num_samples)
                    raw_target_sameclass = [[self.tokenizer.bos_token] + item_ud._raw_tokens[:dep_id] + [target_sameclass_samples[_]] for _ in range(num_samples)]

                    self.data.extend([
                        DictObj({
                        'raw_head_random': raw_head_random,#[self.tokenizer.bos_token] + item_ud._raw_tokens[:dep_id],                'raw_sameclass'
                        'raw_head_sameclass': raw_head_sameclass,
                        'raw_target': raw_target_sameclass,#[self.tokenizer.bos_token] + item_ud._raw_tokens[:dep_id+1],
                        'subtoken_alignment': subtoken_alignment,
                        'sid': item_ud.meta_data['sid'],
                        'dependent_id': dep_id,
                        'head_id': head_id,
                        'num_samples': num_samples})
                    ])
        self.data.sort(key = lambda x: len(x.raw_head_sameclass), reverse=False)

    def sample_bidirGPT_probing_data(self, strategy = {}):
        self.data = []
        for item_ud in self.examples:
            item_ud: UDSentence 
            raw = ' '.join([self.tokenizer.bos_token] + item_ud._raw_tokens)
            item_encoding = self.tokenizer(raw)
            tokenized_subtoken = item_encoding['input_ids']
            if len(tokenized_subtoken)>160 or len(item_ud._raw_tokens)<2:# or 'http' in item_ud._raw:
                continue
            # tokenized_token = item_ud._upos

            tokenized = self.tokenizer.convert_ids_to_tokens(item_encoding['input_ids'])
            subtoken_alignment = align_tokenized_terminals_gpt(tokenized, item_ud._raw_tokens, self.tokenizer)

            self.data.extend([DictObj({
                'raw_tokens': item_ud._raw_tokens,
                'subtoken_alignment': subtoken_alignment, #including a bos token, right shifted by 1
                'sid': item_ud.meta_data['sid'],
                'dependent_id': dep_id
            }) for dep_id in range(0, len(item_ud._raw_tokens)) if len(subtoken_alignment[dep_id+1])==1])
        self.data.sort(key = lambda x: len(x.raw_tokens), reverse=True)

    def sample_supervised_dependency_data(self, strategy = {}, synrel2id = {}):
        print('samping supervised dependency data with strategy: ', strategy)
        assert len(synrel2id)>0
        self.data = []
        flag_sample_subtokens = True #strategy.get('sample_subtokens', True)
        strategy_type = strategy.get('type', 'train')
        for item_ud in self.examples:
            item_ud: UDSentence 
            item_encoding = self.tokenizer(item_ud._raw)
            tokenized_subtoken = item_encoding['input_ids']
            if len(tokenized_subtoken)>96 or len(item_ud._raw_tokens)<2:# or 'http' in item_ud._raw:
                continue
            # tokenized_token = item_ud._upos

            bert_tokenized = self.tokenizer.convert_ids_to_tokens(item_encoding['input_ids'])
            subtoken_alignment = align_tokenized_terminals_sentencepiece(bert_tokenized, item_ud._raw_tokens, self.tokenizer)
            gold_dependencies = item_ud.sample_supervised_dependencies(synrel2id)
            gold_dependency_set = {dep[:2]: dep[2] for dep in gold_dependencies}
            random_dependencies = generate_window(10000, 1, len(item_ud._raw_tokens))
            random_dependencies = [(*udep, gold_dependency_set[udep]) if udep in gold_dependency_set.keys() else (*udep, 0)  for udep in random_dependencies ]

            if strategy_type == 'train': 
                sample_ratio = strategy.get('sample_ratio', 1)

                self.data.extend([DictObj({
                    **gold_dependencies,
                    'subtoken_alignment': subtoken_alignment,
                    'sent': item_ud,
                    'token_len': len(item_ud._raw_tokens),
                    'raw_tokens': item_ud._raw_tokens,
                    'raw': item_ud._raw
                }) for _ in range(sample_ratio)])
            elif strategy_type == 'eval-gold':
                self.data.extend([DictObj({
                    'deps': [dep],
                    'subtoken_alignment': subtoken_alignment,
                    'sent': item_ud,
                    'token_len': len(item_ud._raw_tokens),
                    'raw_tokens': item_ud._raw_tokens,
                    'raw': item_ud._raw
                }) for dep in gold_dependencies['deps']])
            elif strategy_type == 'eval-random':
                self.data.extend([DictObj({
                    'deps': [dep],
                    'subtoken_alignment': subtoken_alignment,
                    'sent': item_ud,
                    'token_len': len(item_ud._raw_tokens),
                    'raw_tokens': item_ud._raw_tokens,
                    'raw': item_ud._raw
                }) for dep in random_dependencies])
            else: 
                raise NotImplementedError


    def sample_mlm_data(self, strategy: Dict[str, str]):
        print('sample mlm data')
        #assumes the same training strategy as BERT
        self.data = []
        flag_sample_subtokens = strategy.get('sample_subtokens', True)
        for item_id in tqdm(range(len(self.examples)), disable=True):
            
            item_ud = self.examples[item_id]
            item_encoding = self.tokenizer(item_ud._raw)
            # if flag_sample_subtokens:
            # print(item_ud._raw_tokens)
            tokenized_subtoken = item_encoding['input_ids']
            if len(tokenized_subtoken)>96:
                continue
            tokenized_token = item_ud._upos

            bert_tokenized = self.tokenizer.convert_ids_to_tokens(item_encoding['input_ids'])
            subtoken_alignment = align_tokenized_terminals_sentencepiece(bert_tokenized, item_ud._raw.split(' '), self.tokenizer)


            

            sample_ratio: int = int(strategy.get('sample_ratio', 3))
            x_prob = float(strategy.get('x_prob', 0.1))
            y_prob = float(strategy.get('y_prob', 0.15))
            x_idx = torch.bernoulli((torch.ones(len(tokenized_token))* x_prob).unsqueeze(0).repeat(sample_ratio, 1)).bool()
            x_idx_subtokens = torch.bernoulli((torch.ones(len(tokenized_subtoken))* x_prob).unsqueeze(0).repeat(sample_ratio, 1)).bool()
            y_idx = torch.bernoulli((torch.ones(len(tokenized_token))* y_prob).unsqueeze(0).repeat(sample_ratio, 1)).bool()
            y_idx_subtokens = torch.bernoulli((torch.ones(len(tokenized_subtoken))* y_prob).unsqueeze(0).repeat(sample_ratio, 1)).bool()
            # y_idx = torch.logical_and(y_idx, torch.logical_not(x_idx)) # to ensure that y does not overlap with x
            # y_idx_subtokens = torch.logical_and(y_idx_subtokens, torch.logical_not(x_idx_subtokens)) #

            self.data.extend([{
                'sent': item_ud,
                'x_idx': x.squeeze(0),
                'y_idx': y.squeeze(0),
                'x_idx_subtokens': xs.squeeze(0),
                'y_idx_subtokens': ys.squeeze(0),
                'subtoken_alignment': subtoken_alignment,
                }
            for x, y, xs, ys in zip(x_idx.split(1, dim=0), y_idx.split(1, dim=0), x_idx_subtokens.split(1, dim=0), y_idx_subtokens.split(1, dim=0))])

    @staticmethod
    def sample_vinfo_ij_pair(self: UDSentence, strategy):
        self.data = []
        self.data_word = []
        self.sent_separator_idx = [0]
        separator_cnt = 0
        flag_sample_subtokens = strategy.get('sample_subtokens', True)
        print(len(self.examples))
        for item_id in range(len(self.examples)):
            # print("processing item_id", item_id)
            item_ud = self.examples[item_id]
            # print(item_ud)
            item_encoding = self.tokenizer(item_ud._raw)
            seq_len = len(item_encoding['input_ids'])
            
            # print(seq_len)
            if item_ud.overlength is None:
                if seq_len>96:
                    self.sent_separator_idx.append(separator_cnt)
                    item_ud.overlength = True
                    continue
                else:
                    item_ud.overlength = False
            if item_ud.overlength: continue

            
            if flag_sample_subtokens:
                tokenized = self.tokenizer.convert_ids_to_tokens(item_encoding['input_ids'])
                subtoken_alignment = align_tokenized_terminals_sentencepiece(tokenized, item_ud._raw.split(' '), self.tokenizer)
                edges = [_ for _ in UDSentence.sample_ij_pairs(item_ud, strategy, seq_len = len(item_ud._raw.split(' '))+2, lb = 1, ub = len(item_ud._raw_tokens)+1) if 0 not in _]
                edges = [item for item in edges if len(subtoken_alignment[item[0]]) <= 5 and len(subtoken_alignment[item[1]])<=5]
                # print(edges)
                self.data.extend([{
                        'x_idx': subtoken_alignment[item[0]],
                        'y_idx': subtoken_alignment[item[1]],
                        'sent': item_ud,
                        'x_tokens': item_ud._raw_tokens[item[0]-1],
                        'y_tokens': item_ud._raw_tokens[item[1]-1],
                    } for item in edges])
                self.data_word.extend([((item[0], item[1]), item_ud) for item in edges])
                separator_cnt += len(edges)
                self.sent_separator_idx.append(separator_cnt)
            else:
                edges = UDSentence.sample_ij_pairs(item_ud, strategy, len(item_ud._raw.split(' '))+2, lb = 1, ub = len(item_ud._raw_tokens)+1)
                self.data.extend([
                    {
                        'sent': item_ud,
                        'xy_pair_idx': (item[0], item[1]),
                        'x_token': item_ud._raw_tokens[item[0]-1],
                        'y_token': item_ud._raw_tokens[item[1]-1],
                    }
                    for item in edges
                ])
                separator_cnt += len(edges)
                self.sent_separator_idx.append(separator_cnt)

    def __getitem__(self, i):
        # print(self.data[i])
        # i, j, item_id = self.data[i]
        # return ((i, j), self.examples[item_id]._raw)
        return self.data[i]


    def __len__(self):
        return len(self.data)

    @staticmethod
    def prep_for_graph_decoding(self:CoNLLDataset, strategy):
        raise NotImplementedError

        self.data = []
        self.data_word = []
        separator_cnt = 0
        self.sent_separator_idx = [0]

        for item_id in range(len(self.examples)):
            item_ud = self.examples[item_id]
            edges = [_ for _ in self.sentence_class.sample(item_ud, strategy, len(item_ud._raw.split(' '))+2) if 0 not in _]
            
            # print(edges)

            item_encoding = self.tokenizer(item_ud._raw)
            seq_len = len(item_encoding['input_ids'])
            # ud_seq_len = len(item_ud._raw.split(' '))

            if item_ud.overlength is None:
                if seq_len>96:
                    self.sent_separator_idx.append(separator_cnt)
                    item_ud.overlength = True
                    continue
                else:
                    item_ud.overlength = False
            if item_ud.overlength: continue

            tokenized = self.tokenizer.convert_ids_to_tokens(item_encoding['input_ids'])
            
            # if str(item_id) in self.vinfo_hdf5_cache.keys():
                # self.vinfo_hdf5_cache.create_dataset(str(item_id), (ud_seq_len, ud_seq_len, ), dtype='f')
            # subtoken_alignment = map_tokenized_terminals_alignment[model_args.tokenizer_type](tokenized, item_ud._raw.split(' '), self.tokenizer)
            subtoken_alignment = align_tokenized_terminals_sentencepiece(tokenized, item_ud._raw.split(' '), self.tokenizer)
            self.data.extend([{
                'xy_pair_idx': (subtoken_alignment[item[0]], subtoken_alignment[item[1]]),
                'sent': self.examples[item_id],
                'y_neighbor_idx': [item_subtok for _ in range(max(0, item[1]-strategy['neighbor_mask']), min(len(item_ud._raw_tokens)+1, item[1]+strategy['neighbor_mask']+1)) for item_subtok in subtoken_alignment[_] if item_subtok not in  subtoken_alignment[item[0]]] if 'neighbor_mask' in strategy.keys() else None,
                'x_token': item_ud._raw_tokens[item[0]-1],
                # 'subtoken_alignment': subtoken_alignment
                } for item in edges])
            self.data_word.extend([((item[0], item[1]), self.examples[item_id]) for item in edges])
            separator_cnt += len(edges)
            self.sent_separator_idx.append(separator_cnt)

    #decreapted
    @staticmethod
    def sample_sentence(self: CoNLLDataset):
        #shoudl use the sample_ij_pairs function
        # raise NotImplementedError
        self.data = []
        self.sid2data = {}
        for id, item_ud in enumerate(self.examples):
            # if item.overlength is None:
            #     item_encoding = self.tokenizer(item._raw)
            #     seq_len = len(item_encoding['input_ids'])
            #     if seq_len>96:
            #         item.overlength = True
            #         continue
            # if item.overlength: continue
            # if len(item._raw_tokens) < 2: continue
            obj = DictObj({
                'item_ud': item_ud,
                'raw_tokens': item_ud._raw_tokens,
                'sid': item_ud.meta_data['sid']
                })
            self.sid2data[str(item_ud.meta_data['sid'])] = obj
            self.data.append(obj)
                
    def sample_CMI_dependencies(self, strategy: Dict[str, str]):
        self.data = []
        for item_ud in self.examples:
            raw_tokens = item_ud._raw_tokens
            edges = UDSentence.sample_ij_pairs(item_ud, strategy, len(item_ud._raw.split(' ')), lb = 0, ub = len(item_ud._raw_tokens))
            self.data.extend([DictObj({
                'dep': (item[0], item[1]),
                'sent': item_ud,
                'raw_tokens': raw_tokens,
                'sid': item_ud.meta_data['sid']
                # 'y_neighbor_idx': [item_subtok for _ in range(max(0, item[1]-strategy['neighbor_mask']), min(len(item_ud._raw_tokens)+1, item[1]+strategy['neighbor_mask']+1)) for item_subtok in subtoken_alignment[_] if item_subtok not in  subtoken_alignment[item[0]]] if 'neighbor_mask' in strategy.keys() else None,
                # 'x_token': item_ud._raw_tokens[item[0]-1],
                # 'subtoken_alignment': subtoken_alignment
                }) for item in edges if item[0]<=item[1]])



        pass


    @staticmethod
    def sample_vinfo_pairs_by_sentence(self, strategy: Dict[str, str]):
        raise NotImplementedError
        self.data = []
        flag_sample_subtokens = strategy.get('sample_subtokens', True)
        for item_id in range(len(self.examples)):
            
            item_ud = self.examples[item_id]
            item_encoding = self.tokenizer(item_ud._raw)
            seq_len = len(item_encoding['input_ids'])
            if seq_len>96:
                continue
            # print(item_ud._raw_tokens)
            if flag_sample_subtokens:
                tokenized = self.tokenizer.convert_ids_to_tokens(item_encoding['input_ids'])
                subtoken_alignment = align_tokenized_terminals_sentencepiece(tokenized, item_ud._raw.split(' '), self.tokenizer)
                edges = [_ for _ in UDSentence.sample(item_ud, strategy, len(item_ud._raw.split(' '))+2) if 0 not in _]
                self.data.extend([{
                    'xy_pair_idx': (subtoken_alignment[item[0]], subtoken_alignment[item[1]]),
                    'sent': self.examples[item_id],
                    'y_neighbor_idx': [item_subtok for _ in range(max(0, item[1]-strategy['neighbor_mask']), min(len(item_ud._raw_tokens)+1, item[1]+strategy['neighbor_mask']+1)) for item_subtok in subtoken_alignment[_] if item_subtok not in  subtoken_alignment[item[0]]] if 'neighbor_mask' in strategy.keys() else None,
                    'x_token': item_ud._raw_tokens[item[0]-1],
                    # 'subtoken_alignment': subtoken_alignment
                    } for item in edges if len(subtoken_alignment[item[0]]) <= 5 and len(subtoken_alignment[item[1]])<=5])
            else:
                edges = UDSentence.sample(item_ud, strategy, len(item_ud._raw.split(' '))+2)
                self.data.extend([
                    {
                        'sent': item_ud,
                        'xy_pair_idx': (item[0], item[1])
                    }
                    for item in edges
                ])




            # ij_choices = self.sentence_class.sample(item_ud, strategy, len(item_ud._raw.split(' '))+2)
            edges = [_ for _ in UDSentence.sample(item_ud, strategy, len(item_ud._raw.split(' '))+2) if 0 not in _]
            # print(edges)
            # for item in edges:
                # ignore words that need more than 5 sub-tokens
                # if len(subtoken_alignment[item[0]]) > 5 or len(subtoken_alignment[item[1]])>5: continue
            self.data.extend([{
                'xy_pair_idx': (subtoken_alignment[item[0]], subtoken_alignment[item[1]]),
                'sent': self.examples[item_id],
                'y_neighbor_idx': [item_subtok for _ in range(max(0, item[1]-strategy['neighbor_mask']), min(len(item_ud._raw_tokens)+1, item[1]+strategy['neighbor_mask']+1)) for item_subtok in subtoken_alignment[_] if item_subtok not in  subtoken_alignment[item[0]]] if 'neighbor_mask' in strategy.keys() else None,
                'x_token': item_ud._raw_tokens[item[0]-1],
                # 'subtoken_alignment': subtoken_alignment
                } for item in edges if len(subtoken_alignment[item[0]]) <= 5 and len(subtoken_alignment[item[1]])<=5])
                # self.data.append(((subtoken_alignment[choice[1]], subtoken_alignment[choice[0]]), self.examples[item_id]))
