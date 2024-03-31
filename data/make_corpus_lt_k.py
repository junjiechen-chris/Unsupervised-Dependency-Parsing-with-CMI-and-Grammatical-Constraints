import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Type, Set, Tuple, Any
import json
import os

f_input = sys.argv[1]
lb, ub = list(map(lambda x: int(x), sys.argv[2: 4]))




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
container = []
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
        raw_tokens = [normalize_double_quotation(tok[WORD_FORM_IDX].strip()) for tok in tokens]
        sentence_size = len(raw_tokens)
        if sentence_size <= ub and sentence_size > lb:
            container.append(gcontainer)
        gcontainer = []
# print(pos_set)
fn, ext = os.path.splitext(f_input)
with open("{}-lb{}-ub{}{}".format(fn, lb, ub, ext), 'w') as f:
    for item in container:
        for line in item:
            # print()
            f.write(line)
            f.write('\n')
        f.write('\n')


        

