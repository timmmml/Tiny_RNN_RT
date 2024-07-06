import os
from training_experiments.training import *


def generate_Nautilus_yaml(config_path, resource_dict, n_jobs=1):
    """Generate yaml file for a config_path.
    See behavior_cv_training_job_combination."""
    memory, cpu, gpu = resource_dict['memory'], resource_dict['cpu'], resource_dict['gpu']

    config_path = str(config_path).replace('\\','/')  # for running on linux
    name = config_path
    config_path = config_path.replace('(','\(').replace(')','\)')
    name = name.lower()
    replace_dict = {
        '/': '.', '_': '.', '-': '.', # Nautilus only allow '.' in the name
        'files.': '', '.allfold.config.pkl': '', 'saved.model.': '', '.cognitive.dynamics.': '', 'd:': '',
        'exp.': '', 'rnn.type.': '', 'dim.': '', 'cog.type.':'', # remove redundant information
        'monkey': 'mk.', 'true': 't', 'false': 'f', 'weight.':'wt.', # shorten the name
        'hidden.': 'hd.', 'output.': 'op.', 'input.': 'ip.',
        'readout.': 'ro.', 'polynomial.order.': 'po.',
        'akamrat': 'akr', 'millerrat': 'mlr',
        'trainval.percent.': 'tvpt.', 'dataprop.': 'dp.', 'inner.splits.': 'ins.',
        '(': '', ')': '', # remove brackets
        'agent.name': 'ag', 'seed': 'sd',
        'rank': 'rk',
        'include.': '', 'embedding': 'ebd',
        'finetune': 'ft',
        'trainprob.t': 'tpt',
        '.distill': '', 'student': 'st', 'teacher': 'tc','none': 'no',
        'trainval.size': 'tvs',
    }
    if n_jobs != 1:
        n_job_cmd = f'-n {n_jobs}'
    else:
        n_job_cmd = ''
    for k, v in replace_dict.items():
        name = name.replace(k, v)
    # print('job name=',name, 'len=', len(name))
    assert len(name)<=62, f'Name {name} has length of {len(name)}. The maximum length is 62.'
    log_path = '/volume/logs/' + name + '.out'
    yaml = [
    'apiVersion: batch/v1',
    'kind: Job',
    'metadata:',
    f'  name: {name}',
    'spec:',
    '  template:',
    '    spec:',
    '      containers:',
    '      - name: demo',
    '        image: gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp',
    '        command: ["/bin/bash"]',
    '        args:',
    '          - -c',
    '          - >-',
    '              cd /volume/cognitive_dynamics &&',
    '              pip install pandas==1.5.3 &&',
    f'              python training_experiments/training_job_from_config.py -t {config_path} {n_job_cmd} 1>{log_path} 2>{log_path}',
    '        volumeMounts:',
    '        - mountPath: /volume',
    '          name: mattarlab-volume',
    '        resources:',
    '          limits:',
    f'            memory: {int(memory*2)}Gi',
    f'            cpu: "{cpu}"',
    f'            nvidia.com/gpu: "{gpu}"',
    '          requests:',
    f'            memory: {memory}Gi',
    f'            cpu: "{cpu}"',
    f'            nvidia.com/gpu: "{gpu}"',
    '      volumes:',
    '        - name: mattarlab-volume',
    '          persistentVolumeClaim:',
    '            claimName: mattarlab-volume',
    '      restartPolicy: Never',
    '  backoffLimit: 1',
    ]
    return name, yaml

def write_Nautilus_yaml(config_paths, exp_folder, resource_dict, n_jobs=1):
    """Generate yaml files for all config_paths.
    See behavior_cv_training_job_combination.
    """
    os.makedirs('files/kube', exist_ok=True)
    apply_cmds = []
    delete_cmds = []
    with open(f'files/kube/apply_{exp_folder}.txt', 'a+') as apply_f:
        with open(f'files/kube/delete_{exp_folder}.txt', 'a+') as delete_f:
            for config_path in config_paths:
                name, yaml = generate_Nautilus_yaml(config_path, resource_dict, n_jobs=n_jobs)
                with open(f'files/kube/{name}.yaml', 'w') as f:
                    for y in yaml:
                        print(y, file=f)
                apply_cmd = f'kubectl apply -f {name}.yaml'
                delete_cmd = f'kubectl delete -f {name}.yaml'
                print(apply_cmd, file=apply_f)
                print(delete_cmd, file=delete_f)
                apply_cmds.append(apply_cmd)
                delete_cmds.append(delete_cmd)
    return apply_cmds, delete_cmds


def behavior_cv_training_job_combination(base_config, config_ranges, resource_dict, n_jobs=1):
    """Generate all files for training jobs.

    Each job has a config file (in files/saved_models/exp_name), a yaml file (in files/kube/), and a apply/delete command (in files/kube/).
    We should run these commands manually to submit the jobs to the cluster.

    Args:
        base_config: the base config file.
        config_ranges: a dictionary of config ranges.
        resource_dict: a dictionary of resource requirements.
            e.g. {'memory': 5, 'cpu': 16, 'gpu': 0}
            memory is in Gi, cpu is in core, gpu is in number.
    """
    goto_root_dir.run()
    configs = config_control.vary_config(base_config, config_ranges, mode='combinatorial')
    config_paths = []
    for c in configs:
        config_path = Path('./files/saved_model') / c['model_path'] / 'allfold_config.pkl'
        os.makedirs(config_path.parent, exist_ok=True)
        joblib.dump(c, config_path)
        config_paths.append(config_path)

    apply_cmds, delete_cmds = write_Nautilus_yaml(config_paths, base_config['exp_folder'], resource_dict, n_jobs=n_jobs)
    for cmd in apply_cmds:
        print(cmd)
    return apply_cmds, delete_cmds