import os
import goto_root_dir
from pathlib import Path
filter = ''
# filter = 'finetune'
#filter = 'trainprob'
# filter = 'dim-20.l'
targets =[
# 'exp_seg_akamrat49',
#  'exp_seg_akamrat50',
#  'exp_seg_akamrat51',
#  'exp_seg_akamrat52',
#  'exp_seg_akamrat53',
#  'exp_seg_akamrat54',
#  'exp_seg_akamrat100',
#  'exp_seg_akamrat95',
#  'exp_seg_akamrat96',
#  'exp_seg_akamrat97',
#  'exp_seg_akamrat98',
#  'exp_seg_akamrat99',
#  'exp_seg_akamrat263',
#  'exp_seg_akamrat264',
#  'exp_seg_akamrat266',
#  'exp_seg_akamrat267',
#  'exp_seg_akamrat268',
'exp_seg_akamrat49_distill',
# # 'exp_nonseg_akamrat49',
# # 'exp_nonseg_akamrat50',
# # 'exp_nonseg_akamrat51',
# # 'exp_nonseg_akamrat52',
# # 'exp_nonseg_akamrat53',
# # 'exp_nonseg_akamrat54',
# # 'exp_seg300_akamrat49',
# # 'exp_seg300_akamrat267',

# 'exp_seg_CPB',
# 'exp_monkeyV',
# 'exp_monkeyW',
# 'exp_monkeyV_dataprop',
# 'exp_monkeyW_dataprop',
#
    # 'exp_sim_millerrat55_nblocks200',
# 'exp_sim_millerrat55',
# 'exp_sim_metarl',
# 'exp_seg_millerrat55',
# 'exp_seg_millerrat64',
# 'exp_seg_millerrat70',
# 'exp_seg_millerrat71',
# 'exp_seg_millerrat55_dataprop',
# 'exp_seg_millerrat64_dataprop',
# 'exp_seg_millerrat70_dataprop',

# 'exp_seg_millerrat88',
# 'exp_seg_akamrat49_dataprop',
# 'exp_seg_millerrat88',
# 'exp_seg_millerrat71',
# 'exp_dezfouli100',
# 'exp_dezfouli98',
# 'exp_dezfouli94',
# 'exp_dezfouli93',
# 'exp_dezfouli92',
# 'exp_dezfouli90',
# 'exp_dezfouli84',
# 'exp_dezfouli78',
# 'exp_dezfouli77',
# 'exp_dezfouli76',
# 'exp_dezfouli75',
# 'exp_dezfouli71',
# 'exp_dezfouli67',
# 'exp_dezfouliAll',
# 'exp_monkeyW',

# 'exp_seg_akamratrts279',
# 'exp_seg_akamratrts278',
# 'exp_seg_akamratrts277',
# 'exp_seg_akamratrts275',
# 'exp_seg_akamratrts274',
# 'exp_seg_akamratrts273',
# 'exp_seg_akamratrts272',
# 'exp_seg_akamratrts271',
# 'exp_seg_akamratrts270',
# 'exp_seg_akamratrts269',
#
# 'exp_seg_akamratprl388',
# 'exp_seg_akamratprl383',
# 'exp_seg_akamratprl382',
# 'exp_seg_akamratprl380',
# 'exp_seg_akamratprl368',
# 'exp_seg_akamratprl367',
# 'exp_seg_akamratprl361',
# 'exp_seg_akamratprl360',
# 'exp_seg_akamratprl359',
# 'exp_seg_akamratprl358',
# 'exp_seg_akamratAll',
# 'exp_Lai',
# 'exp_iblIBL-T1','exp_iblIBL-T4',
# 'exp_iblNYU-01', 'exp_iblNYU-02', 'exp_iblNYU-04', 'exp_iblCSHL_002',
# 'exp_iblCSHL_003', 'exp_iblCSHL_005', 'exp_iblCSHL_007', 'exp_iblCSHL_008',
# 'exp_iblCSHL_010', 'exp_iblCSHL_014', 'exp_iblCSHL_015', 'exp_iblKS003',
# 'exp_iblZM_1369', 'exp_iblZM_1371', 'exp_iblZM_1372', 'exp_iblZM_1743', 'exp_iblZM_1745',
# 'exp_iblibl_witten_04', 'exp_iblibl_witten_05', 'exp_iblibl_witten_06', 'exp_iblibl_witten_07',
# 'exp_iblibl_witten_12', 'exp_iblibl_witten_13', 'exp_iblibl_witten_16'

]
tar_yaml_path = Path('files/kube')
for target in targets:
    with open(tar_yaml_path / f'tar.models.yaml', 'r') as reader:
        if len(filter):
            target_dots = (target+'_'+filter).replace('_','.').lower()
        else:
            target_dots = target.replace('_','.').lower()
        with open(tar_yaml_path / f'tar.models.{target_dots}.yaml', 'w') as writer:
            for line in reader:
                line = line.replace('REPLACEME1',target_dots)
                if len(filter):
                    line = line.replace('REPLACEME2', target+ '_' + filter)
                    line = line.replace('REPLACEME3', target + '/*' + filter+'*')
                else:
                    line = line.replace('REPLACEME2', target)
                    line = line.replace('REPLACEME3', target)
                print(line, end='', file=writer)
        print(f'kubectl apply -f', f'tar.models.{target_dots}.yaml')

print('')
