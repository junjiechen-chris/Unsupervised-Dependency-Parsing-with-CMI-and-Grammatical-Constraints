import sys
import json

argv = sys.argv[1:]
section_label = 'ud'
main_conllu = argv[0]
output_conllu = argv[1]
HEAD_IDX=6


with open(main_conllu) as f:
    lines = f.read().splitlines()
groups = []
g_contianer = []
cnt = 0
for line in lines:
    if line.startswith('#'):
        continue
    if line!='':
        token = line.split('\t')
        token[HEAD_IDX] = json.dumps({section_label: token[HEAD_IDX]})
        g_contianer.append(token)
    else:
        if len(groups)==0 or len(groups[-1])>0:
            # print(g_contianer)
            groups.append(g_contianer)
            g_contianer = []

print(groups)
with open(output_conllu, 'w') as f:
    for group in groups:
        f.write('!# ')
        f.write(json.dumps({'sid': cnt}))
        f.write('\n')
        for token in group:
            f.write('\t'.join(token)+'\n')
        f.write('\n')
        cnt += 1
        





