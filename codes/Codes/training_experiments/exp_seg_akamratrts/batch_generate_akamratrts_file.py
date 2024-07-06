import os

source = '269'
targets = ['270','271','272','273','274','275','277','278','279']
for target in targets:
    with open(f'exp_seg_akamratrts{source}.py', 'r') as reader:
        with open(f'exp_seg_akamratrts{target}.py', 'w') as writer:
            for line in reader:
                line = line.replace(source, target)
                print(line, end='', file=writer)
#
for target in [source]+targets:
    print(f'python exp_seg_akamratrts/exp_seg_akamratrts{target}.py')

for target in [source]+targets:
    print(f'!python exp_seg_akamratrts{target}.py')
