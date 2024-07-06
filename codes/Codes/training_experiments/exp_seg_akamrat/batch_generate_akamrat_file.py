import os

source = '49'
targets = ['50', '51', '52', '53', '54', '100', '95', '96', '97', '98', '99', '263', '264', '266', '267', '268']
for target in targets:
    with open(f'exp_seg_akamrat{source}.py', 'r') as reader:
        with open(f'exp_seg_akamrat{target}.py', 'w') as writer:
            for line in reader:
                line = line.replace(source, target)
                print(line, end='', file=writer)

for yaml_type in ['cpu.','']:
    if os.path.exists(f'train.exp.seg.{yaml_type}akamrat{source}.yaml'):
        for target in targets:
            with open(f'train.exp.seg.{yaml_type}akamrat{source}.yaml', 'r') as reader:
                with open(f'train.exp.seg.{yaml_type}akamrat{target}.yaml', 'w') as writer:
                    for line in reader:
                        line = line.replace(source, target)
                        print(line, end='', file=writer)
#
for target in [source]+targets:
    print(f'python exp_seg_akamrat/exp_seg_akamrat{target}.py')

for target in [source]+targets:
    print(f'!python exp_seg_akamrat{target}.py')

for target in [source]+targets:
    print(f'kubectl apply -f train.exp.seg.cpu.akamrat{target}.yaml')
print('')
for target in [source] + targets:
    print(f'kubectl delete -f train.exp.seg.cpu.akamrat{target}.yaml')