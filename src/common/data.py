
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Type, Set, Tuple, Any
import json
import unicodedata
from easydict import EasyDict as edict

def normalize_double_quotation(s:str) -> str:
    return s.replace("''", '"').replace("``", '"').replace('“', '"').replace('”', '"').replace('\'\'', '"').replace('’', '\'').replace('—', '-').replace('–', '-').replace('…', '...').replace('`', '\'')

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFKD', s)
                  if unicodedata.category(c) != 'Mn')


TOKENIZER_OFFSET_DICT = edict({
    'BERT_TOKENIZER_IDX_OFFSET': 1
})

def identity(x):
    return x


def align_tokenized_terminals_sentencepiece(tokenized: List[str], terminals: List[str], tokenizer, flag_strip_accents = True):
    excessive_tokens = {
        '<s>',
        '</s>',
        '<sep>',
        '<cls>',
        '[SEP]',
        '[CLS]'
    }
    
    if not flag_strip_accents:
        terminal_processor = identity
    else:
        terminal_processor = strip_accents

    terminals = [terminal_processor(i) for i in terminals.copy()]
    terminals.append('@[EOS]@')
    
    clean_tok = lambda x: x if not x.startswith('##') else x[2:]



    # print(tokenized)
    tokenized_idx, tokenized = zip(*[(id, cleaned_item) for id, item in enumerate(tokenized) if len(cleaned_item:=terminal_processor(item.strip().replace('▁', '').replace('</w>','').replace('Ġ', '')))>0 and cleaned_item not in excessive_tokens])
    # print(tokenized)

    ptr_tok = 0
    ptr_trm = 0
    groups = []
    current_ref_tok = ''
    while ptr_trm<=len(terminals) and ptr_tok<len(tokenized):
        tok = tokenized[ptr_tok].lower()
        ref_tok = terminals[ptr_trm].lower()
        # print(tok, ref_tok, ref_tok.startswith(tok), current_ref_tok)
        assert tok != tokenizer.unk_token
        if current_ref_tok.startswith(clean_tok(tok)):
            assert ptr_tok<len(tokenized)
            groups[-1].append(tokenized_idx[ptr_tok])
            assert len(current_ref_tok) >= len(clean_tok(tok))
            current_ref_tok = current_ref_tok[len(clean_tok(tok)):]
        elif ref_tok.startswith(tok):
            groups.append([tokenized_idx[ptr_tok]])
            current_ref_tok = ref_tok[len(clean_tok(tok)):]
            ptr_trm+=1
        else:
            assert ptr_tok<len(tokenized)
            groups[-1].append(tokenized_idx[ptr_tok])
        ptr_tok+=1
    return groups



class UDSentenceBase:
    ID_IDX: int = 0
    WORD_FORM_IDX: int = 1
    UPOS_IDX: int = 3
    HEAD_IDX: int = 6
    REL_IDX: int = 7
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
        self.dep_head_pairs = [(int(tok[self.ID_IDX])-1, {k: int(v)-1 for k, v in json.loads(tok[self.HEAD_IDX]).items()}, tok[self.REL_IDX], tok[self.UPOS_IDX]) for tok in self.tokens]# if int(json.loads(tok[self.HEAD_IDX])['ud']) !=0]
        self.dep_head_pairs_dict = {(int(json.loads(tok[self.HEAD_IDX]).get('ud', -1))-1, int(tok[self.ID_IDX])-1): tok[self.REL_IDX] for tok in self.tokens if int(json.loads(tok[self.HEAD_IDX]).get('ud', -1))!=0}# if int(json.loads(tok[self.HEAD_IDX])['ud']) !=0]
            

        self._raw = ' '.join(self._raw_tokens)
        self.syn_rels = set([tok[self.REL_IDX] for tok in self.tokens])
        
    def _is_dependency(self, i, j):
        # print(self.dep_head_pairs_dict)
        # print(i, j, (i, j) in self.dep_head_pairs_dict.keys())
        # input()
        return (i, j) in self.dep_head_pairs_dict.keys() or (j, i) in self.dep_head_pairs_dict.keys()

    def __len__(self):
        return len(self._raw_tokens)
    
    def _sid(self):
        return self.meta_data['sid']

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
                if not flag_return_bidir_dependencies and i > j:
                    continue
                if i != j and mask_allowed_tokens[i] and mask_allowed_tokens[j]:
                    data.append(edict({
                        'word_idx':(i, j),
                        'side_token_idx':subtoken_alignment[i],
                        'query_token_idx':subtoken_alignment[j],
                        'flag_is_dependency': self._is_dependency(i, j),
                        'sid': self.meta_data['sid'],
                        }))
        # self.data = data
        return data
    
    def sample_token_pairs(self):
        #sample pairs that are of single token
        data = []
        
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens)):
                if i > j:
                    continue
                if i != j:
                    data.append(edict({
                        'word_idx':(i, j),
                        'flag_is_dependency': self._is_dependency(i, j),
                        'dependency_label': self.dep_head_pairs_dict.get((i, j), 'None'),
                        'upos': (self._upos[i], self._upos[j]),
                        'sid': self.meta_data['sid'],
                        }))
        # self.data = data
        return data




class UDDatasetBase:
    def __init__(self, file_path: str, max_token_count = 256, SentenceClass = UDSentenceBase, f_handler = None):

        if f_handler is None:
            # WARNING: THE SENTENCE LENGTH IS NOW LIMITED TO 30 PIECES
            with open(file_path, encoding="utf-8-sig") as f:
                lines = f.read().splitlines()
        else:
            lines = f_handler.read().splitlines()
            f_handler.close()
        # print(lines)

        self.examples = []
        gcontainer = []
        for line in lines:
            # print(line.decode())
            if isinstance(line, bytes):
                line = line.decode()
            # print(line)
            if line != '':
                gcontainer.append(line)
            else:
                if len(gcontainer)<max_token_count+1 and len(gcontainer)>0: #otherwise just drop the tokens, including the root token
                    self.examples.append(SentenceClass(gcontainer))
                gcontainer = []    
        if len(gcontainer)<max_token_count+1 and len(gcontainer)>0: #otherwise just drop the tokens, including the root token
            self.examples.append(SentenceClass(gcontainer))
                
    def __len__(self):
        raise NotImplementedError("UDDatasetBase should be use as an abstract class")
    
    def __getitem__(self):
        raise NotImplementedError("UDDatasetBase should be use as an abstract class")
    
    def _get_word_lens(self):
        return [(i._sid(), len(i)) for i in self.examples]

        
        

@dataclass
class ModelArguments:
    model_str: str = field(default='bert-base-uncased')

@dataclass
class DataArguments:
    train_data_file: Optional[str] = field(default=None)
    dev_data_file: Optional[str] = field(default=None)

@dataclass
class TrainingArguments:
    learning_rate: float = field(default = 1e-5)
    weight_decay: float = field(default = 1e-2)
    seed: int = field(default=1)
    epochs: int = field(default=5)
    warmup_epochs: int = field(default=1)
    warmup_startup_factor: float = field(default=0.1)
    amp: bool = field(default=False)
    gradient_clip:float = field(default=1.0)
    batch_size: int = field(default=32)
    num_samples_for_centroids:int = field(default=10)
    