import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Type, Set
import unicodedata

print(sys.argv)
finput, foutput, seq_max_len, flag_count_mode = sys.argv[1:]
seq_max_len = int(seq_max_len)

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFKD', s)
                  if unicodedata.category(c) != 'Mn')

def normalize_double_quotation(s:str) -> str:
    return s.replace("''", '"').replace("``", '"')

@dataclass
class UDSentence:
    tokens: List[List[Union[str, int]]]
    ALLOWED_SAMPLING_STRATEGY: List[str]
    ID_IDX: int = 0
    WORD_FORM_IDX: int = 1
    UPOS_IDX: int = 3
    HEAD_IDX: int = 6
    REL_IDX: int = 7
    def __init__(self, groups):
        self.groups = groups
        groups = [item for item in groups if not item.startswith('#') and not item.startswith('!')]
        self.tokens = [ele for item in groups if '-' not in (ele := item.split('\t'))[0] and '.' not in ele[0]]
        # self.dep_head_pairs = [(int(tok[self.ID_IDX]), int(tok[self.HEAD_IDX]), tok[self.REL_IDX]) for tok in self.tokens if int(tok[self.HEAD_IDX])!=0]
        _raw = [normalize_double_quotation(strip_accents(tok[self.WORD_FORM_IDX].strip())) for tok in self.tokens]
        _upos = [tok[self.UPOS_IDX] for tok in self.tokens]
        if flag_count_mode == 'word':
            self.seq_len = len([i for i in _upos if i != 'PUNCT'])
        elif flag_count_mode == 'token':
            self.seq_len = len(_upos)
        else:
            raise ValueError(f'Unknown flag_count_mode: {flag_count_mode}')
        # self._raw = ' '.join(_raw)


groups = []
g_container = []
with open(finput, 'r') as f:
    lines = f.read().splitlines()

for line in lines:
    if line != '':
        g_container.append(line)
    else:
        groups.append(UDSentence(g_container))
        g_container = []

with open(foutput, 'w') as f:
    for sent in groups:
        if sent.seq_len <= seq_max_len:# and sent.seq_len > 1:
            for item in sent.groups:
                f.write(item)
                f.write('\n')
            f.write('\n')

