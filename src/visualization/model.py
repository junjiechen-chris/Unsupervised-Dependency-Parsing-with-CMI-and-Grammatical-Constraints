from torch import nn
from typing import Optional, List, Dict, Union, Type, Set, Tuple, Any
from tqdm import tqdm
from easydict import EasyDict as edict
import networkx as nx
import numpy as np
from collections import Counter
import scipy

import unicodedata
import pickle

class Eisner(object):
    """
    Dependency decoder class
    """

    def __init__(self):
        self.verbose = False

    def parse_proj(self, scores):
        """
        Parse using Eisner's algorithm.
        """

        # ----------
        # Solution to Exercise 4.3.6
        nr, nc = np.shape(scores)
        if nr != nc:
            raise ValueError("scores must be a squared matrix with nw+1 rows")
            return []

        N = nr - 1  # Number of words (excluding root).

        # Initialize CKY table.
        complete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
        incomplete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
        complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)  # s, t, direction (right=1).
        incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)  # s, t, direction (right=1).

        incomplete[0, :, 0] -= np.inf

        # Loop from smaller items to larger items.
        for k in range(1, N+1):
            for s in range(N-k+1):
                t = s + k

                # First, create incomplete items.
                # left tree
                incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s]
                incomplete[s, t, 0] = np.max(incomplete_vals0)
                incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
                # right tree
                incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t]
                incomplete[s, t, 1] = np.max(incomplete_vals1)
                incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

                # Second, create complete items.
                # left tree
                complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
                complete[s, t, 0] = np.max(complete_vals0)
                complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
                # right tree
                complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
                complete[s, t, 1] = np.max(complete_vals1)
                complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

        value = complete[0][N][1]
        heads = -np.ones(N + 1, dtype=int)
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

        value_proj = 0.0
        for m in range(1, N+1):
            h = heads[m]
            value_proj += scores[h, m]

        return heads, value_proj

        # End of solution to Exercise 4.3.6
        # ----------

    def backtrack_eisner(self, incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
        """
        Backtracking step in Eisner's algorithm.
        - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
        an end position, and a direction flag (0 means left, 1 means right). This array contains
        the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
        - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
        an end position, and a direction flag (0 means left, 1 means right). This array contains
        the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
        - s is the current start of the span
        - t is the current end of the span
        - direction is 0 (left attachment) or 1 (right attachment)
        - complete is 1 if the current span is complete, and 0 otherwise
        - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
        head of each word.
        """
        if s == t:
            return
        if complete:
            r = complete_backtrack[s][t][direction]
            if direction == 0:
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
                return
            else:
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
                return
        else:
            r = incomplete_backtrack[s][t][direction]
            if s == 1 or t == 1:
                pass
            if direction == 0:
                heads[s] = t
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
                return
            else:
                heads[t] = s
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
                return



class EdmondAgentWithPOSConstraint(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def decode(self, measure, uposmask, upos_fn_mask):
        flag_need_redo = True
        while flag_need_redo:
            # print(measure)
            counter = Counter()
            g = nx.from_numpy_array(measure, create_using=nx.Graph).to_undirected()
            pred = set(nx.maximum_spanning_tree(g).edges())
            counter.update([item[0] for item in pred])
            counter.update([item[1] for item in pred])
            flag_need_redo = False
            for i, cnt in counter.items():
                if upos_fn_mask[i] and cnt > 1:
                    measure[i, :] *= 0.5
                    measure[:, i] *= 0.5
                    flag_need_redo = True
        return pred
        

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # Here they are dividing a wrong dimension!!!

class EsinerAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def batch_decode(self, b_vinfo_mtx, b_buffer_size, device='cuda:0', b_punct_list=None):
        b_vinfo_mtx = b_vinfo_mtx + b_vinfo_mtx.T
        b_buffer_size = b_buffer_size
        outputs = []
        if b_punct_list is None:
            b_punct_list = [[] for _ in range(len(b_buffer_size))]
        for vinfo, buffer_size, punct_list  in zip(b_vinfo_mtx, b_buffer_size, b_punct_list):
            buffer_size = buffer_size.item()
            # print(vinfo)
            # print(buffer_size)
            outputs.append(self.decode(vinfo[:buffer_size, :buffer_size], buffer_size, punct_list))    
        return outputs

    def fill_chart(self, vinfo_mtx, buffer_size):
        score_chart: np.Array = -9999. * np.ones((buffer_size, buffer_size, 2, 2, ), dtype=float)
        backtrace_chart: np.Array = np.zeros((buffer_size, buffer_size, 2, 2,), dtype=int)
        dependency_storage: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}
        depednency_count: np.Array = np.zeros((buffer_size, buffer_size, 2, 2,), dtype=int)
        
        for i in range(0, buffer_size):
            score_chart[i, i] = 0
        # print(score_chart.reshape(buffer_size, buffer_size, 4))

        for k in range(1, buffer_size):
            for i in range(0, buffer_size):
                # for j in range(i+1, buffer_size):
                j = i+k
                if j>=buffer_size: continue
                for q in range(i, j+1):
                # if j>=buffer_size: br?eak
                    if q < j:
                        pseudo_score = np.clip(score_chart[i, q, 1, 1] + score_chart[q+1, j, 0, 1] + vinfo_mtx[j, i]+5, a_min=-9999., a_max=None)
                        if score_chart[i, j, 0, 0] < pseudo_score and pseudo_score > -9e3:
                            score_chart[i, j, 0, 0] = pseudo_score
                            backtrace_chart[i, j, 0, 0] = q
                            dependency_storage[(i, j, 0, 0)] = (j, i)
                            depednency_count[i, j, 0, 0] = depednency_count[i, q, 1, 1] + depednency_count[q+1, j, 0, 1] + 1

                        pseudo_score = np.clip(score_chart[i, q, 1, 1] + score_chart[q+1, j, 0, 1] + vinfo_mtx[i, j]+5, a_min=-9999., a_max=None)
                        if score_chart[i, j, 1, 0] < pseudo_score and pseudo_score > -9e3:
                            score_chart[i, j, 1, 0] = pseudo_score
                            backtrace_chart[i, j, 1, 0] = q
                            dependency_storage[(i, j, 1, 0)] = (i, j)
                            depednency_count[i, j, 1, 0] = depednency_count[i, q, 1, 1] + depednency_count[q+1, j, 0, 1] + 1

                    pseudo_score = np.clip(score_chart[i, q, 0, 1] + score_chart[q, j, 0, 0], a_min=-9999., a_max=None)
                    if score_chart[i, j, 0, 1] < pseudo_score and pseudo_score > -9e3:
                        score_chart[i, j, 0, 1] = pseudo_score
                        backtrace_chart[i, j, 0, 1] = q
                        depednency_count[i, j, 0, 0] = depednency_count[i, q, 0, 1] + depednency_count[q, j, 0, 0]
                    

                    pseudo_score = np.clip(score_chart[i, q, 1, 0] + score_chart[q, j, 1, 1], a_min=-9999., a_max=None)
                    # if i==0 and j == buffer_size-1:
                        # print(q, pseudo_score)
                    if score_chart[i, j, 1, 1] < pseudo_score and pseudo_score > -9e3:
                        score_chart[i, j, 1, 1] = pseudo_score
                        backtrace_chart[i, j, 1, 1] = q
                        depednency_count[i, j, 1, 1] = depednency_count[i, q, 1, 0] + depednency_count[q, j, 1, 1]
        return score_chart, backtrace_chart, dependency_storage, depednency_count

    
    def eisner_span_retracer(self, ref_g, buffer_size):
        queue = [(0, buffer_size-1, 1, 1)]
        selected_spans = [(0, buffer_size-1, 1, 1)]
        dependencies = set(ref_g.nodes)
        ptn_map = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
        backtrace_patterns = [((1, 1), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0)), ((1, 0), (1, 1))]
        while len(queue) > 0:
            span = queue.pop(0)
            assert len(span) ==4
            span_type = span[2:]
            ptn_type_idx = ptn_map[span_type]
            if ptn_map[span_type] == 0:
                pass
            elif ptn_map[span_type] == 1:
                pass
            elif ptn_map[span_type] == 2:
                i, j = span[:2] # i is the current head
                q = max([item[1] for item in dependencies if item[0] == j])
                queue+=[(bdy, ptn) for bdy, ptn in zip([(i, q), (q, j)], backtrace_patterns[ptn_type_idx])]
                pass
            elif ptn_map[span_type] == 3:
                i, j = span[:2] # i is the current head
                q = max([item[1] for item in dependencies if item[0] == i])
                queue+=[(bdy, ptn) for bdy, ptn in zip([(i, q), (q, j)], backtrace_patterns[ptn_map[span_type]])]
                pass
            else:
                raise NotImplementedError

        pass



    def decode(self, vinfo_mtx, uposmask, root):
        
        vinfo_mtx += vinfo_mtx.T
        vinfo_mtx = vinfo_mtx[uposmask, :][:, uposmask]
        # vinfo_mtx *= 10
        
        # print(uposmask)
        # print(vinfo_mtx)
        
        seqlen = vinfo_mtx.shape[0]
        vinfo_mtx = np.concatenate([np.zeros((seqlen, 1)), vinfo_mtx], axis=1)
        vinfo_mtx = np.concatenate([np.zeros((1, seqlen+1)), vinfo_mtx], axis=0)
        vinfo_mtx[0, 0] = 0
        vinfo_mtx[0, root+1] = 999
        vinfo_mtx[root+1] = 0
        vinfo_mtx[root+1, 0] = 999
        
        # vinfo_mtx = softmax(vinfo_mtx) 
        # print(vinfo_mtx)
        
        decoder = Eisner()
        best_arcs, root_pred = decoder.parse_proj(vinfo_mtx)
        arcs = [sorted([id, head]) for id, head in enumerate(best_arcs) if head != 0 and id!=0 ]
        # input()
        return edict({
            'edges': [tuple(sorted([i[0]-1, i[1]-1])) for i in  arcs if i[0]!=0 and i[1]!=0],
            'raw_edges': arcs,
            # 'nx': g,
            # 'vinfo': vinfo/(buffer_size-1)
        })
        
        buffer_size = vinfo_mtx.shape[0]
        # vinfo_mtx = scipy.special.softmax(vinfo_mtx, axis=-1)
        
        # print(vinfo_mtx)
        # input()

        score_chart, backtrace_chart, dependency_storage, _ = self.fill_chart(vinfo_mtx, buffer_size)

        backtrace_patterns = [((1, 1), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0)), ((1, 0), (1, 1))]
        ptn_map = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}

        best_parse = []
        span_queue = [(0, buffer_size-1, 1, 1)]
        loop_cnt = 0
        vinfo = 0.
        while len(span_queue)>0:
            # print(span_queue)
            span = span_queue.pop(0)
            # print(score_chart[span])
            assert score_chart[span] > -9e-3
            # print(span)
            current_ptn = span[2:]
            i, j = span[:2]
            if i == j: continue
            split_point = backtrace_chart[span]
            ptn_idx = ptn_map[span[2:]]
            assert split_point >= i and split_point<=j
            # print(split_point, ptn_idx)
            ptn_a, ptn_b = backtrace_patterns[ptn_idx]
            if current_ptn in {(0, 0), (1, 0)}:
                # print(span)
                best_parse.append(dependency_storage[span])
                vinfo += vinfo_mtx[span[:2]]
                span_queue+=[(i, split_point, *ptn_a), (split_point+1, j, *ptn_b)]
            else:
                # if split_point == 0: continue
                # best_parse.append((split_point, split_point+1))
                span_queue+=[(i, split_point, *ptn_a), (split_point, j, *ptn_b)]
            loop_cnt += 1
            if loop_cnt > 200:
                break

        # g: nx.DiGraph = nx.DiGraph()
        # g.add_edges_from([item for item in best_parse])
        
        # print(best_parse)        


        return edict({
            'edges': [tuple(sorted([i[0]-1, i[1]-1])) for i in  best_parse if i[0]!=0 and i[1]!=0],
            'raw_edges': best_parse,
            # 'nx': g,
            # 'vinfo': vinfo/(buffer_size-1)
        })

def find_root(parse):
    # root node's head also == 0, so have to be removed
    for token in parse[1:]:
        if token.head == 0:
            return token.id
    return False
def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def match_tokenized_to_untokenized(subwords, sentence):
    token_subwords = np.zeros(len(sentence))
    subwords = [x.lower() for x in subwords]
    subwords = [_run_strip_accents(x) for x in subwords]
    sentence = [x.lower() for x in sentence]
    # sentence = [x.lower() for x in sentence]
    sentence = [_run_strip_accents(x) for x in sentence]
    token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
    for i, subword in enumerate(subwords):
        if subword in ["[CLS]".lower(), "[SEP]".lower()]: continue

        while current_token_normalized is None:
            current_token_normalized = sentence[current_token].lower()

        if subword.startswith("[UNK]".lower()):
            unk_length = int(subword[6:])
            subwords[i] = subword[:5]
            subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
        else:
            subwords_str += subword[2:] if subword.startswith("##") else subword
        if not current_token_normalized.startswith(subwords_str):
            return False

        token_ids[i] = current_token
        token_subwords[current_token] += 1
        if current_token_normalized == subwords_str:
            subwords_str = ""
            current_token += 1
            current_token_normalized = None

    assert current_token_normalized is None
    while current_token < len(sentence):
        assert not sentence[current_token]
        current_token += 1
    assert current_token == len(sentence)

    return token_ids

def normalize_double_quotation(s:str) -> str:
    return s.replace("''", '"').replace("``", '"').replace('“', '"').replace('”', '"').replace('\'\'', '"').replace('’', '\'').replace('—', '-').replace('–', '-').replace('…', '...').replace('`', '\'')


def decoding(matrix, flag_use_scipy_softmax=False, flag_use_softmax = True, flag_add_root_heuristic=True):
    trees = []
    deprels = []
    with open(matrix, 'rb') as f:
        results = pickle.load(f)
    new_results = []
    decoder = Eisner()
    root_found = 0

    for (line, tokenized_text, matrix_as_list) in tqdm(results):
        orginal_line = line
        sentence = [normalize_double_quotation(x.form) for x in line][1:]
        deprels.append([x.deprel for x in line])
        root = find_root(line)

        print(tokenized_text, sentence)

        mapping = match_tokenized_to_untokenized(tokenized_text, sentence)
        # print('mapping', mapping)
        # print(sentence, orginal_line)

        init_matrix = matrix_as_list

        # merge subwords in one row
        merge_column_matrix = []
        for i, line in enumerate(init_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    new_row.append(buf[0])
                    buf = []
            merge_column_matrix.append(new_row)

        # merge subwords in multi rows
        # transpose the matrix so we can work with row instead of multiple rows
        merge_column_matrix = np.array(merge_column_matrix).transpose()
        merge_column_matrix = merge_column_matrix.tolist()
        final_matrix = []
        for i, line in enumerate(merge_column_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    new_row.append((sum(buf) / len(buf)))
                    buf = []
            final_matrix.append(new_row)

        # transpose to the original matrix
        final_matrix = np.array(final_matrix).transpose()

        if final_matrix.shape[0] == 0:
            print('find empty matrix:',sentence)
            continue
        assert final_matrix.shape[0] == final_matrix.shape[1]

        if flag_add_root_heuristic:
            final_matrix[root] = 0
            final_matrix[root, 0] = 1
            final_matrix[0, 0] = 0

        new_results.append((orginal_line, tokenized_text, final_matrix))

        if flag_use_softmax:
            if flag_use_scipy_softmax:
                final_matrix = scipy.special.softmax(final_matrix, axis=1)
            else:
                final_matrix = softmax(final_matrix)

        final_matrix = final_matrix.transpose()



        best_heads, _ = decoder.parse_proj(final_matrix)
        for i, head in enumerate(best_heads):
            if head == 0 and i == root:
                root_found += 1
        trees.append([(i, head) for i, head in enumerate(best_heads)])
    return trees, new_results, deprels

