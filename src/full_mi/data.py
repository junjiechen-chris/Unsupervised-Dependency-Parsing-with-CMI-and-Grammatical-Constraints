
from src.common.data import align_tokenized_terminals_sentencepiece, UDSentenceBase, UDDatasetBase, TOKENIZER_OFFSET_DICT
from src.common.utilities import vmap_list, ListObj
from easydict import EasyDict as edict
import torch
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from typing import Optional, List, Dict, Union, Type, Set, Tuple, Any
from itertools import repeat

rng = np.random.default_rng()
class UDSentenceForCLMSampler(UDSentenceBase):    
    def sample_single_token_pairs(self, tokenizer, offset_str = 'BERT_TOKENIZER_IDX_OFFSET', left_context = 200, right_context = 200, max_seqlen = 64, flag_strip_accent = True, flag_mask_query_toks = True):
        assert flag_mask_query_toks, 'The flag_mask_query_toks must be True'
        #sample pairs that are of single token
        offset = TOKENIZER_OFFSET_DICT[offset_str]
        data = []
        # mask_allowed_tokens = [len(group) == 1 for group in subtoken_alignment]
        # print(subtoken_alignment)
        
        assert len(self.tokens) == len(self._raw_tokens), 'The length of tokens and raw tokens should be the same'
        
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens)):
                if i < j:
                    tmp_tokens = deepcopy(self._raw_tokens)
                    if flag_mask_query_toks:
                        tmp_tokens[i] = tokenizer.mask_token 
                        tmp_tokens[j] = tokenizer.mask_token
                    # print(tmp_tokens)
                    l_boundary = max(0, i-left_context)
                    r_boundary = min(len(self._raw_tokens), j+right_context+1)
                    adjusted_word_idx = (i - max(0, i-left_context), j - max(0, i-left_context))
                    tmp_tokens = tmp_tokens[l_boundary:r_boundary]
                    # print(tmp_tokens)


                    tokenized = tokenizer.tokenize(' '.join(tmp_tokens))
                    # print(len(tokenized), len(tokenized) > max_seqlen)
                    if len(tokenized)> max_seqlen:
                        continue
                    # print(tokenized,[token == tokenizer.unk_token for token in tokenized] )
                    # print(tmp_tokens, tokenized)
                    assert not any([token == tokenizer.unk_token for token in tokenized]), f'none of the token should be unk, {tmp_tokens}, {tokenized}'
                    subtoken_alignment = align_tokenized_terminals_sentencepiece(tokenized, tmp_tokens, tokenizer, flag_strip_accents=flag_strip_accent)
                    subtoken_alignment = [[i+offset for i in group] for group in subtoken_alignment]
                    # print(subtoken_alignment)
                    
                    assert len(subtoken_alignment[adjusted_word_idx[0]]) == 1 and len(subtoken_alignment[adjusted_word_idx[1]]) == 1, 'The query token can only have a length 1'
                    assert i < len(self.tokens) and j< len(self.tokens), 'The index of the query token should be less than the length of the sentence'
                    # assert self._raw_tokens 
                    
                    
                    data.append(edict({
                        'word_idx':(i, j),
                        # 'adjusted_word_idx': adjusted_word_idx,
                        'side_token_idx':subtoken_alignment[adjusted_word_idx[0]],
                        'query_token_idx':subtoken_alignment[adjusted_word_idx[1]],
                        'flag_is_dependency': self._is_dependency(i, j),
                        'sid': self.meta_data['sid'],
                        'upos': (self._upos[i], self._upos[j]),
                        'raw_tokens': tmp_tokens,
                        }))
        # print(data)
        # input()
        # self.data = data
        return data

    pass        


def merge_samples_with_sentence(raw_tokens, samples, idx, left_context = 1000, right_context = 1000):
    # here we assume the wordform of the sentence
    assert len(idx) == 2, 'idx should be a tuple of length 2, but got {}'.format(idx)
    # print(samples)
    raws = []
    i, j = idx
    seqlen = len(raw_tokens)
    for sample in samples:
        assert len(sample) == 2, 'samples should only be 2-length'
        tmp_raw = deepcopy(raw_tokens[max(0, idx[0]-left_context):min(seqlen, idx[1]+right_context)])
        lshift = max(0, idx[0]-left_context)
        tmp_raw[i-lshift] = sample[0].decode('utf-8')
        tmp_raw[j-lshift] = sample[1].decode('utf-8')
        raws.append(' '.join(tmp_raw))
    return raws
    

class UDDatasetForCLMSampler(UDDatasetBase):
    def __init__(self, data, tokenizer, f_handler = None):
        super().__init__(data, SentenceClass=UDSentenceForCLMSampler, f_handler=f_handler)
        self.tokenizer = tokenizer
        self.data = None
        
    
    def sample(self, pair_filter = None, left_context = 200, right_context = 200, max_seqlen = 64,flag_strip_accent = True, flag_mask_query_toks = True):
        assert flag_mask_query_toks, 'The flag_mask_query_toks must be True'
        data = []
        assert pair_filter is not None, 'Please provide a pair filter'
        for sid, sentence in tqdm(enumerate(self.examples)):
            sentence: UDSentenceForCLMSampler
            if len([ i for i in self.examples[sid]._upos if i != 'PUNCT' ]) <= 1:
                continue
            # sample single-token pairs according to the given tokenizer
            # the single-token constraint needs only be met in the proposal model, but not the scoring model
            pairs = sentence.sample_single_token_pairs(self.tokenizer, left_context = left_context, right_context = right_context, max_seqlen=max_seqlen, flag_strip_accent=flag_strip_accent, flag_mask_query_toks=flag_mask_query_toks)
            extended_pairs = [edict({
                **pair,
            }) for pair in pairs if pair_filter(sid, pair.word_idx, pair.upos)]
            # print("pre-filtered", [p.word_idx for p in pairs])
            # print("post-filtered", [pair.word_idx for pair in pairs if pair_filter(sid, pair.word_idx, pair.upos)])
            data.extend(extended_pairs)
        self.data = data
        
    def bucketing(self, collate_fn, tokenizer, limit_tokens = 2048, max_token_gap_tolerance = 0):
        # This function groups data by the number of raw tokens 
        assert self.data is not None, 'Please sample the data first'
        # sorted_data = sorted(self.data, key=lambda x: len(x.raw_tokens), reverse=True)
        sorted_data = sorted(self.data, key=lambda x: len(tokenizer.tokenize(' '.join(x.raw_tokens))), reverse=True)
        chunks = [[]]
        min_max_length = [999999, 0]
        tok_cnt = 0
        # print("data len", len(self.data))
        # print(self.data[:10])
        for d in sorted_data:
            # d_len = len(d.raw_tokens)
            d_len = len(tokenizer.tokenize(' '.join(d.raw_tokens))) 
            if tok_cnt + d_len > limit_tokens or min_max_length[1] - min_max_length[0] > max_token_gap_tolerance:
                # chunks[-1] = collate_fn(chunks[-1])
                # print(chunks[-1])
                assert len(chunks[-1]) > 0, 'The chunk should not be empty'
                chunks.append([])
                tok_cnt = 0
                min_max_length = [999999, 0]
            chunks[-1].append(d)
            tok_cnt += d_len
            min_max_length = [min(min_max_length[0], d_len), max(min_max_length[1], d_len)]
        self.chunked_data = chunks
        self.collate_fn = collate_fn

        print("number of chunks", len(chunks))
        
    def __iter__(self):
        # This function returns a generator yielding chunked data
        assert self.chunked_data is not None, 'Please bucket the data first'
        for item in self.chunked_data:
            yield self.collate_fn(item)
        # yield [self.collate_fn(i) for i in self.chunked_data]
        
        
            
        
        
        
    def _get_chunked_lens(self):
        return len(self.chunked_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

####################################


class UDSentenceForNEMI(UDDatasetBase):
    pass

class GenericDatasetContainer:#(Dataset)
    def __init__(self, data) -> None:
        self.data = data
        pass
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class UDSentenceForNEMI(UDSentenceBase):

    def sample_single_token_pairs(self, tokenizer, offset_str = 'BERT_TOKENIZER_IDX_OFFSET', flag_return_bidir_dependencies = True):
        #sample pairs that are of single token
        offset = TOKENIZER_OFFSET_DICT[offset_str]
        data = []
        tokenized = tokenizer.tokenize(self._raw)
        subtoken_alignment = align_tokenized_terminals_sentencepiece(tokenized, self._raw_tokens, tokenizer)
        subtoken_alignment = [[i+offset for i in group] for group in subtoken_alignment]
        mask_allowed_tokens = [len(group) == 1 for group in subtoken_alignment]
        # print(subtoken_alignment)
        
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens)):
                if i<j:
                    data.append(edict({
                        'word_idx':(i, j),
                        'side_token_idx':subtoken_alignment[i],
                        'query_token_idx':subtoken_alignment[j],
                        'flag_is_dependency': self._is_dependency(i, j),
                        'sid': self.meta_data['sid'],
                        }))
        # self.data = data
        return data
    
        
class UDDatasetForNEMI(UDDatasetBase):
    def __init__(self, data, tokenizer, flag_training = True):
        super().__init__(data, SentenceClass=UDSentenceForNEMI)
        self.tokenizer = tokenizer
        self.flag_training = flag_training
        
    
    def sample_sampled_pairs(self, h5f, filter_func = lambda *args : True):
        def remove_samples_with_unk_codes(samples):
            # print(samples)
            data = []
            for s in samples:
                try:
                    s = s.astype(str)
                    # print(s.shape)
                    data.append(s)
                except Exception:
                    pass
            return np.asarray(data)
        
        def query_has_samples(sid, word_idx):
            # arr = remove_samples_with_unk_codes(h5f['samples'][str(sid)][word_idx[0], word_idx[1]])
            arr = h5f['samples'][str(sid)][word_idx[0], word_idx[1]]
            flag = arr.shape[0]>0
            flag_empty = all([b[0].decode('utf-8') == '' and b[1].decode('utf-8') == '' for b in arr])
            return flag and not flag_empty
            for b in arr:
                # print(b)
                if (b[0].decode('utf-8') == '' and b[1].decode('utf-8') == ''):
                    flag = False
                    break
                    # tmp_flag = False
            # print('---------')
            return flag
                    

            arr = arr.astype(str)
            # if arr.shape[0]< 64:
                # print(arr.shape)
                # print(arr)
            # print(arr, arr.shape)
            return arr.shape[0] > 0 and not np.all(np.char.str_len(arr) == 0)
        def get_samples(sid, word_idx, shuffle = False):
            arr = h5f['samples'][str(sid)][word_idx[0], word_idx[1]]
            # print(arr)

            # if shuffle:
            #     np.random.shuffle(arr[:, 0])
            # arr = [[i.decode('utf-8') for i in b] for b in arr]
            return arr
        if self.flag_training:
            data = [[], []]
        else:
            data = []
        for did, sentence in tqdm(enumerate(self.examples)):
            assert self.examples[did] == sentence, 'The sentence is not the same'
            sentence: UDSentenceForNEMI
            if len([ i for i in sentence._upos if i != 'PUNCT' ]) <= 1:
                continue
            sid = sentence.meta_data['sid']
            if not str(sid) in h5f['samples'].keys(): continue # temporarily bypass sentences that are not sampled for some reasons
            pairs = sentence.sample_single_token_pairs(self.tokenizer, flag_return_bidir_dependencies=False)
            extended_pairs = [edict({
                'samples': get_samples(sid, pair.word_idx),
                **pair,
            }) for pair in pairs  if query_has_samples(sid, pair.word_idx)]
            
            print('ref1', [pair.word_idx for pair in pairs])
            print('ref2', [pair.word_idx for pair in pairs  if query_has_samples(sid, pair.word_idx)])
            # print(extended_pairs[0])
            if self.flag_training:
                raise NotImplementedError('the XYZ mode is disabled')
                data[0].extend([edict({'sid': sid, 'raw': raw_p, 'raw_original': ' '.join(sentence._raw_tokens)}) for pair in extended_pairs for raw_p in pair.raws_positive])
                data[1].extend([edict({'sid': sid, 'raw': raw_n, 'raw_original': ' '.join(sentence._raw_tokens)}) for pair in extended_pairs for raw_n in pair.raws_negative])
            else:
                data.extend([
                    edict({
                        'sid': sid,
                        'raw_tokens': sentence._raw_tokens,
                        **pair,
                    }) for pair in extended_pairs if filter_func(sentence, pair.word_idx)
                ])
        self.data = data
        
        
def collate_fn_for_nemi_fixz(batch, tokenizer, device):
    #yield a generator of negative samples
    assert len(batch) == 1, 'The batch size must be 1'
    def shuffle_samples_1(samples):
        samples = samples.copy()
        samples = rng.permuted(samples, axis=0)
        return samples
    
    def byte_to_str(samples):
        return [[i.decode('utf-8') for i in b] for b in samples]
    
    raw_tokens = [d.raw_tokens for d in batch][0]
    samples = [d.samples for d in batch][0]
    # samples = samples[::4, :]
    shuffled_samples = shuffle_samples_1(samples)
    word_idx = [d.word_idx for d in batch][0]
    raws_p = (rp:=merge_samples_with_sentence(raw_tokens, samples, word_idx), tokenizer(rp, padding=True, return_tensors='pt'))
    raws_n  =  (rn:=sum([merge_samples_with_sentence(raw_tokens, shuffle_samples_1(samples), word_idx) for i in range(4)], []), tokenizer(rn, padding=True, return_tensors='pt'))
    iter = (raws_p, raws_n)
    
    flag_is_dependency = [d.flag_is_dependency for d in batch][0]    

    if flag_is_dependency:
        print(samples[:4], shuffled_samples[:4])
        print(raws_p[0][::4])
        print(raws_n[0][::4])
        print(raw_tokens)
        print('#######', flush=True)
    
    
    sid = [d.sid for d in batch][0]
    

    return edict({
        'sid': sid,
        'iter': iter, #((raws_p, BE_p), (raws_n, BE_n))
        'flag_is_dependency': flag_is_dependency,
        'samples': samples,
        'word_idx': word_idx,
        }), None

def collate_fn_for_nemi_fixz_negative_generable(batch, tokenizer, device):
    #yield a generator of negative samples
    assert len(batch) == 1, 'The batch size must be 1'
    def shuffle_samples(samples):
        samples = samples.copy()
        # samples = rng.permuted(samples, axis=0)
        np.random.shuffle(samples[:, 0])
        return samples
    
    raw_tokens = [d.raw_tokens for d in batch][0]
    samples = [d.samples for d in batch][0]
    word_idx = [d.word_idx for d in batch][0]
    raws_p = lambda: (r:=merge_samples_with_sentence(raw_tokens, samples, word_idx), tokenizer(r, padding=True, return_tensors='pt'))
    raws_n  = lambda: (r:=merge_samples_with_sentence(raw_tokens, shuffle_samples(samples), word_idx), tokenizer(r, padding=True, return_tensors='pt'))
    iter = (raws_p, raws_n)
    
    flag_is_dependency = [d.flag_is_dependency for d in batch][0]    
    
    sid = [d.sid for d in batch]
    
    return edict({
        'sid': sid,
        'iter': iter, #((raws_p, BE_p), (raws_n, BE_n))
        'flag_is_dependency': flag_is_dependency,
        'samples': samples,
        }), None

def collate_fn_for_nemi_training(batch, tokenizer, device, flag_pair_with_original = False):
    raws = [d.raw for d in batch]
    raws_original = [d.raw_original for d in batch]
    if not flag_pair_with_original:
        inputs = tokenizer(raws, padding=True, return_tensors='pt')
    else:
        raws_for_inputs = [[original, sample] for original, sample in zip(raws_original, raws)]
        inputs = tokenizer(raws_for_inputs, padding=True, return_tensors='pt')
    
    sid = [d.sid for d in batch]
    
    return edict({
        'inputs': inputs.to(device),
        'sid': sid,
        'raws': raws,
        # 'raws_original': raws_original,
        }), None
        
def collate_fn_for_clm_sampler(batch, dataset, tokenizer, device, mask_vocab_ = None):
    # print('DEBUG: collate: batch:', batch)
    
    vocab_size = len(tokenizer.get_vocab())
    
    upos = [d.upos for d in batch] 
    upos = list(map(list, zip(*upos)))
    
    
    # pre-mask the token of interest to avoid multi-token words
    # raw_tokens = [dataset.examples[d.sid]._raw_tokens for d in batch]
    raws = [' '.join(d.raw_tokens) for d in batch]
    word_idx = [d.word_idx for d in batch] # word idx to be sampled
    # print(raws)
    # print(word_idx)
    
    
    inputs = tokenizer(raws, padding=True, return_tensors='pt')

    flag_is_dependency = [d.flag_is_dependency for d in batch]
    sid = [d.sid for d in batch]
    
    token_idx = [torch.tensor([d.side_token_idx for d in batch]), torch.tensor([d.query_token_idx for d in batch])] # token idx in the proposal model to be sampled
    vocab_mask = [torch.zeros((len(batch), vocab_size), dtype=torch.bool) for _ in range(2)]

    mask_vocab_(vocab_mask, upos)
    

        # pass
    # print(vocab_mask[0][0].sum())
    token_info = [(_token_idx.to(device), _vocab_mask.to(device)) for _token_idx, _vocab_mask in zip(token_idx, vocab_mask)]
    
    return edict({
        'inputs': inputs.to(device),
        'word_idx': word_idx,
        'raws': raws, 
        'token_info': token_info,

        'flag_is_dependency': flag_is_dependency,       
        
        'sid': sid,
        }), None
        
        