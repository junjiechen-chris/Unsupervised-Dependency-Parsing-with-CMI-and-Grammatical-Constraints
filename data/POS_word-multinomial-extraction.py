import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Type, Set, Tuple, Any
import json

f_input = sys.argv[1]


HEAD_IDX=6
ID_IDX: int = 0
WORD_FORM_IDX: int = 1
UPOS_IDX: int = 3
HEAD_IDX: int = 6
REL_IDX: int = 7

f_partitions = {}
with open(f_input, encoding="utf-8-sig") as f:
    lines = f.read().splitlines()

def normalize_double_quotation(s:str) -> str:
    return s.replace("''", '"').replace("``", '"')


upos_word_map = {'all': [], 'clean_all':[]}
clean_exclude_upos = ['X', 'SYM', 'PUNCT']


example_cnt: int = 0
gcontainer = []
pos_set = set()
for line in lines:
    if line != '':
        # print()
        gcontainer.append(line)
    else:
        # if len(gcontainer)<max_token_count+1: #otherwise just drop the tokens, including the root token
        groups = [item for item in gcontainer[1:] if not item.startswith('#')]
        tokens: List[str] = [ele for item in groups if '-' not in (ele := item.split('\t'))[0] and '.' not in ele[0]]
        # self._raw_tokens = [normalize_double_quotation(strip_accents(tok[self.WORD_FORM_IDX].strip())) for tok in self.tokens]
        # temporary disable accent stripping as it conflicts with GPT's token-start marker
        print([tok[WORD_FORM_IDX].strip() for tok in tokens])
        print(tokens)
        raw_tokens = [normalize_double_quotation(tok[WORD_FORM_IDX].strip()) for tok in tokens]
        upos = [tok[UPOS_IDX] for tok in tokens]
        pos_set = pos_set.union(set(upos))
        for pos, rt in zip(upos, raw_tokens):
            if pos not in upos_word_map.keys():
                upos_word_map[pos] = []
            upos_word_map[pos].append(rt)
            upos_word_map['all'].append(rt)
            if pos not in clean_exclude_upos:
                upos_word_map['clean_all'].append(rt)
        gcontainer = []
print(pos_set)
with open("{}.upos_word".format(f_input), 'w') as f:
    f.write(json.dumps(upos_word_map))


        

