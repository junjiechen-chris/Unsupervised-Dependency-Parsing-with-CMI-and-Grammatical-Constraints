from src.common.data import UDDatasetBase, UDSentenceBase, normalize_double_quotation
from typing import Optional, List, Dict, Union, Type, Set, Tuple, Any
from tqdm import tqdm
from easydict import EasyDict as edict
import json
import numpy as np

class UDSentenceForVisualization(UDSentenceBase):
    def __init__(self, groups):
        # print(groups, groups[0].startswith('!#'))
        if groups[0].startswith('!#'):
            self.meta_data = edict(json.loads(groups[0][2:].strip()))
            groups = [item for item in groups[1:] if not item.startswith('#')]
        else:
            groups = [item for item in groups if not item.startswith('#')]
        self.tokens: List[str] = [ele for item in groups if '-' not in (ele := item.split('\t'))[0] and '.' not in ele[0]]
        
        self._raw_tokens = [normalize_double_quotation(ele[self.WORD_FORM_IDX].strip()) for item in groups if '-' not in (ele := item.split('\t'))[0] and '.' not in ele[0]]
        self._upos = [tok[self.UPOS_IDX] for tok in self.tokens]
        self.dep_head_pairs = [({k: int(v)-1 for k, v in json.loads(tok[self.HEAD_IDX]).items()}['ud'], int(tok[self.ID_IDX])-1, ) for tok in self.tokens]# if int(json.loads(tok[self.HEAD_IDX])['ud']) !=0]
        self.dep_head_pairs_dict = {tuple(sorted((int(json.loads(tok[self.HEAD_IDX]).get('ud', -1))-1, int(tok[self.ID_IDX])-1))): tok[self.REL_IDX] for tok in self.tokens if int(json.loads(tok[self.HEAD_IDX]).get('ud', -1))!=0}# if int(json.loads(tok[self.HEAD_IDX])['ud']) !=0]
        
        self.root = -1
        for tok in self.tokens:
            if int(json.loads(tok[self.HEAD_IDX]).get('ud', -1))-1 == -1:
                self.root = int(tok[self.ID_IDX])-1
            
        for id, tok in enumerate(self.tokens):
            if int(json.loads(tok[self.HEAD_IDX]).get('ud', -1))==0:
                assert id == self.root
                self.root = id

        self._raw = ' '.join(self._raw_tokens)
        self.syn_rels = set([tok[self.REL_IDX] for tok in self.tokens])
    pass


class UDDatasetForVisualization(UDDatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, SentenceClass=UDSentenceForVisualization)
    
    def sample_pairs(self, filters = [], flag_append = False, 
                     ADMISSIBLE_POS = set(['PRON', 'AUX', 'DET', 'NOUN', 'ADP', 'PROPN', 'VERB', 'NUM', 'ADJ', 'CCONJ', 'ADV', 'PART', 'INTJ', 'SYM', 'SCONJ', 'X']),
                     FUNC_POS = set(["ADP", "AUX", "CONJ", "DET", "PART", "SCONJ", "PRT"]),
                     return_directed_dependency = False

                     ):
        data = []
        for did, sentence in tqdm(enumerate(self.examples)):
            assert self.examples[did] == sentence, 'The sentence is not the same'
            sentence: UDSentenceForVisualization
            if len([ i for i in sentence._upos if i != 'PUNCT' ]) <= 1:
                continue
            pairs = sentence.sample_token_pairs()
            for p in pairs: p.update({'did': did})
            # print(pairs)
            for f in filters:
                pairs = [edict({**p, 'uposmask': np.asarray([p in ADMISSIBLE_POS for p in sentence._upos]), 'upos_fn_mask': np.asarray([p in FUNC_POS for p in sentence._upos])}) for p in pairs if f(sentence, p)]
            data.extend(pairs) if not flag_append else data.append(edict({'did': did, 'sentence': sentence, 'deps':pairs, 'uposmask': np.asarray([p in ADMISSIBLE_POS for p in sentence._upos]), 'upos_fn_mask': np.asarray([p in FUNC_POS for p in sentence._upos]), 'root': sentence.root}))
            # input('stop')
        self.data = data
        
    def sample_sentence(self, filters = [], ADMISSIBLE_POS=[], return_directed_dependency = False):
        return self.sample_pairs(filters, flag_append = True, ADMISSIBLE_POS=ADMISSIBLE_POS, return_directed_dependency=return_directed_dependency)
