# from ast import main
import sys
import json

argv = sys.argv[1:]
main_conllu = argv[0]
output_conllu = argv[1]
aux_conllus = argv[2:]
HEAD_IDX=6

print(aux_conllus)
assert all([item.startswith('--') for id, item in enumerate(aux_conllus) if id%2 == 0])
partition_conllus = [item for id, item in enumerate(aux_conllus) if id%2 == 1]
partition_names =  [item[2:] for id, item in enumerate(aux_conllus) if id%2 == 0]
print(partition_names)


# f_partitions = {name: for name, item in zip(partition_names, partition_conllus)}
f_partitions = {}

for pname, pfile in zip(['ud'] + partition_names, [main_conllu] + partition_conllus):
    with open(pfile) as f:
        lines = f.read().splitlines()
    groups = []
    g_contianer = []
    for line in lines:
        if line.startswith('#'):
            continue
        if line!='':
            g_contianer.append(line.split('\t'))
        else:
            if len(groups)==0 or len(groups[-1])>0:
                # print(g_contianer)
                groups.append(g_contianer)
                g_contianer = []
    f_partitions[pname] = groups

print(f_partitions['ud'][0])

# f_partitions = {k: [line.split['\t'] for line in v.read().splitlines()]  for k, v in f_partitions.items()}

for group_id in range(len(f_partitions['ud'])):
    gold_group = f_partitions['ud'][group_id]
    aux_groups = {pname: f_partitions[pname][group_id] for pname in partition_names}
    # print(gold_group, len(gold_group))
    for line_id in range(len(gold_group)):
        # print(line_id)
        for pname in partition_names:
            # print(aux_groups[pname][line_id][1], gold_group[line_id][1], pname)
            print(aux_groups, gold_group)
            assert aux_groups[pname][line_id][1] == gold_group[line_id][1]
        print(line_id, HEAD_IDX, gold_group[line_id])
        line_head_idx_set = json.dumps({'ud': gold_group[line_id][HEAD_IDX], **{pname: aux_groups[pname][line_id][HEAD_IDX] for pname in partition_names}})
        gold_group[line_id][HEAD_IDX] = line_head_idx_set
        # for pname in partition_names:
            
with open(output_conllu, 'w') as f:
    for gid, group in enumerate(f_partitions['ud']):
        # print(group)
        raw_text = [item[1] for item in group]
        # print(raw_text)
        # break
        f.write('!# {\"sid\" : %s, \"raw_text\": \"%s\"}' % (gid, ' '.join(raw_text).replace('"', '\\\"')))
        f.write('\n')
        for line in group:
            f.write('\t'.join(line))
            f.write('\n')
        f.write('\n')
            




# print(f_partitions)



