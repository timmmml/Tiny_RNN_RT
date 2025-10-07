import pprint
from pathlib import Path

pp = pprint.PrettyPrinter(indent=4, depth=4)


def sim_config_from_inputs(
    print_config=False,
    agent_type=None,
    cog_type=None,
    device=None,
    exp_folder=None,
    seed=None,
    num_blocks=None,
    num_trials=None,
):
    if not agent_type:
        agent_type = input("Enter agent type, default RTSCog: ")
    agent_type = agent_type if agent_type else "RTSCog"
    if not cog_type:
        cog_type = input("Enter cog type, default MB0: ")
    cog_type = cog_type if cog_type else "MB0"
    if not device:
        device = input("Enter device, default cpu: ")
    device = device if device else "cpu"
    if exp_folder is None:
        exp_folder = f'exp_{input("Enter experiment folder name: ")}'
    if seed is None:
        seed = input("Enter seed, default 0: ")
    try:
        seed = int(seed)
    except:
        seed = 0
    if not num_blocks:
        num_blocks = input("Enter number of blocks, default 100: ")
    try:
        num_blocks = int(num_blocks) if num_blocks else 100
    except:
        num_blocks = 100
    if not num_trials:
        num_trials = input("Enter number of trials, default 100: ")
    try:
        num_trials = int(num_trials) if num_trials else 100
    except:
        num_trials = 100

    task = f"Akam_{agent_type[0:3]}"
    exp_folder = exp_folder if exp_folder else f"exp_simulated_{task}"
    task_config = {
        "Akam_PRL": {
            "com_prob": 1,
            "rew_gen": "blocks",
            "block_length": 50,
            "rew_probs": [0.2, 0.8],
        },
        "Akam_RTS": {
            "com_prob": 0.8,
            "rew_gen": "blocks",
            "block_length": 50,
            "rew_probs": [0.2, 0.8],
        },
        "Akam_NTS": {
            "com_prob": 0.8,
            "rew_gen": "trans_rev",
            "block_length": 50,
            "rew_probs": [0.2, 0.8],
        },
    }

    config = {
        ### dataset info
        "dataset": "MillerRat",
        "behav_format": "cog_session",
        ### model info
        "agent_type": agent_type,
        "cog_type": cog_type,
        "device": device,
        "outer_fold": 0,
        "inner_fold": 0,
        "seed": seed,
        "exp_folder": exp_folder,
        "task": task,
        "com_prob": task_config[task]["com_prob"],  # common transition probability
        "rew_gen": task_config[task]["rew_gen"],
        "block_length": task_config[task]["block_length"],
        "rew_probs": task_config[task]["rew_probs"],
        "n_blocks": num_blocks,
        "n_trials": num_trials,
        "sim_seed": 0,
        "sim_exp_name": f"simulated_{task}",
        "additional_name": "",
        "model_path": Path(exp_folder)
        / Path(f"cog_type-{cog_type}")
        / f"outerfold0_innerfold0_seed{seed}",
    }
    if print_config:
        pp.pprint(config)
    return config
