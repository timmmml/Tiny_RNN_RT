import os

source = '358'
targets = ['359','360','361','367','368','380','382','383','388']
for target in targets:
    with open(f'exp_seg_akamratprl{source}.py', 'r') as reader:
        with open(f'exp_seg_akamratprl{target}.py', 'w') as writer:
            for line in reader:
                line = line.replace(source, target)
                print(line, end='', file=writer)
#
for target in [source]+targets:
    print(f'python exp_seg_akamratprl/exp_seg_akamratprl{target}.py')

for target in [source]+targets:
    print(f'!python exp_seg_akamratprl{target}.py')
