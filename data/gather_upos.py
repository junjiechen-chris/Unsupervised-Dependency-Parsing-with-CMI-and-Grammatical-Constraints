import sys
from collections import Counter, defaultdict
import json
import unicodedata
from tqdm import tqdm

# conllu_list = sys.argv[1:]
section = sys.argv[1]
# print(conllu_list)
# exit()


def strip_accents(s):
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if unicodedata.category(c) != "Mn"
    )


word2upos = defaultdict(list)
upos2word = defaultdict(list)
# for f_conllu in conllu_list:
print("working on {}".format(section))
with open(f"{section}-full.conllu") as f:
    lines = f.read().splitlines()
    for l in tqdm(lines):
        if not l.startswith("#") and not len(l) == 0:
            pack = l.split("\t")
            # print(pack)
            if "-" not in pack[0]:
                word = strip_accents(pack[1].lower())
                upos = pack[3]
                if upos not in word2upos[word]:
                    word2upos[word].append(upos)
                if word not in upos2word[upos]:
                    upos2word[upos].append(word)
with open(f"ud_{section}_word2upos.json", "w") as f:
    f.write(json.dumps(word2upos))
with open(f"ud_{section}_upos2word.json", "w") as f:
    f.write(json.dumps(upos2word))

print(upos2word)
