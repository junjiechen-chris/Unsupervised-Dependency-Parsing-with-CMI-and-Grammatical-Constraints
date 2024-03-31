from easydict import EasyDict as edict
from copy import deepcopy
import torch
import os
import json
from tqdm import tqdm

class ListObj:
    def __init__(self, list) -> None:
        self.list = list
        pass

    def __getattribute__(self, attr):
        # print(self.list[0].__dict__)
        return list(map(lambda x: x.__getattribute__(attr), self.list))

    def __deepcopy__(self):
        return edict(deepcopy(self.list))

def vmap_list(fn, oprand_list):
    return [fn(oprand) for oprand in oprand_list]

def get_fullword_mask(tokenizer):
    fragment_piece = []
    for w, id in tokenizer.vocab.items():
        if not w.startswith('##'):
            fragment_piece.append(id)

    fragment_piece = torch.tensor(fragment_piece)
    
    def vocab_mask_(vocab_mask, upos):
        for _mask in vocab_mask:
            for _mask_sample in _mask:
                _mask_sample.scatter_(0, fragment_piece, True)
    
    return vocab_mask_

# %%
# data_args.vinfo_hdf5_dir
def get_upos_mask(tokenizer, working_dir):
    with open(os.path.join(working_dir,'corpus/ud_en_upos2word.json')) as f:
        upos2word = json.loads(f.read())
    # with open(os.path.join(working_dir,'corpus/ud_eng_word2upos.json')) as f:
    #     word2upos = json.loads(f.read())
    upos_set = upos2word.keys()
    upos_mask = {k:[] for k in upos_set}
    for upos in tqdm(upos_set, disable=True):
        wordset = upos2word[upos]
        for w, id in tokenizer.get_vocab().items():
            # if w.startswith('##'):
                # upos_mask[upos].append(id)
            if w.lower() in wordset:
                upos_mask[upos].append(id)
    upos_mask = {k: torch.tensor(v) for k, v in upos_mask.items()}
    
    def vocab_mask_(vocab_mask, upos):
        for _upos, _mask in zip(upos, vocab_mask):
            for _upos_sample, _mask_sample in zip(_upos, _mask):
                _mask_sample.scatter_(0, upos_mask[_upos_sample], True)
    
    return vocab_mask_
