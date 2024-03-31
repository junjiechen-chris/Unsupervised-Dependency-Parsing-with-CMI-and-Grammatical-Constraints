import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Type, Set, Tuple, Any
import json
import os
import tarfile
from io import StringIO, BytesIO
import os.path
import numpy as np

f_input = sys.argv[1]
# split_word_limit = int(sys.argv[2])
split_file_limit = int(sys.argv[2])




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
fn, ext = os.path.splitext(f_input)

tar = tarfile.open("{}.tar".format(f_input), 'w')

example_cnt: int = 0
container = [[] for _ in range(split_file_limit)]
gcontainer = []
pos_set = set()
cnt_toks = np.array([0 for _ in range(split_file_limit)])
# cnt_file = 0

def add_to_tar(container, cnt_file):
    print('loading to {}.worker{}'.format(f_input, cnt_file))
    tarinfo = tarfile.TarInfo('{}.worker{}'.format(f_input, cnt_file))
    tarload = StringIO()
    for item in container:
        for line in item:
            # print()
            tarload.write(line)
            tarload.write('\n')
        tarload.write('\n')
    # tarload.write('\n')
    tarload.seek(0)
    tarinfo.size = len(tarload.getvalue())
    tar.addfile(tarinfo=tarinfo, fileobj=BytesIO(tarload.read().encode('utf8')))
    # container = []

for line in lines:
    if line != '':
        # print()
        gcontainer.append(line)
    else:
        meta_data = json.loads(gcontainer[0][2:])
        groups = [item for item in gcontainer[1:] if not item.startswith('#')]
        tokens: List[str] = [ele for item in groups if '-' not in (ele := item.split('\t'))[0] and '.' not in ele[0]]
        raw_tokens = [normalize_double_quotation(tok[WORD_FORM_IDX].strip()) for tok in tokens]
        sentence_size = len(raw_tokens)
        print(sentence_size)
        
        fillin_bucket = np.argmin(cnt_toks)
        container[fillin_bucket].append(gcontainer)
        cnt_toks[fillin_bucket] += sentence_size ** 2 #* min(30, sentence_size)
        gcontainer = []
# print(pos_set)

for cnt_file, container in enumerate(container):
    add_to_tar(container, cnt_file)




        

