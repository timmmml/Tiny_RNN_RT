"""
Run all models on Akam's rat 49.
Especially the blocks are segmented.
Vary the data proportion.
"""
import sys
from sklearn.model_selection import train_test_split
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

if __name__ == '__main__' or '.' in __name__:
    # when animal_name is 'all', max_segment_length is 150, then [:143] belongs to Akam rat 49. [143:1741] belongs to others
    # split training and testing data for range(0, 143)
    # this animal and other animals
    this_idx = list(range(0, 143))
    other_idx = list(range(143, 1741))
    other_trainval_idx = other_idx
    other_train_idx, other_val_idx = train_test_split(other_trainval_idx, test_size=0.2, random_state=0)
    this_trainval_idx, this_test_idx = train_test_split(this_idx, test_size=0.5, random_state=0) #len: 71 72
    this_trainval_subset_size_list = [2,5,10,20,30,40,50,60,70]
    for this_trainval_subset_size in this_trainval_subset_size_list:
        np.random.seed(0)
        this_trainval_subset_idx = this_trainval_idx[:this_trainval_subset_size]
        this_val_subset_size = int(this_trainval_subset_size * 0.1)
        if this_val_subset_size == 0:
            this_val_subset_size = 1
        this_train_subset_idx, this_val_subset_idx = train_test_split(this_trainval_subset_idx, test_size=this_val_subset_size, random_state=0)
        train_idx = this_train_subset_idx + other_train_idx
        val_idx = this_val_subset_idx + other_val_idx
        test_idx = this_test_idx
        # print('this_trainval_subset_size', this_trainval_subset_size)
        # print('this_train_subset_idx', len(this_train_subset_idx))
        # print('this_val_subset_idx', len(this_val_subset_idx))
        # print('train_idx', len(train_idx))
        # print('val_idx', len(val_idx))
        # print('test_idx', len(test_idx))

        base_config = {
            ### dataset info
            'dataset': 'AkamRat',
            'behav_format': 'tensor',
            'behav_data_spec': ['animal_name', 'max_segment_length','include_embedding'],
            'animal_name': 'all',
            'max_segment_length': 150,
            ### model info
            'agent_type': 'RNN',
            'rnn_type': 'GRU', # which rnn layer to use

            'include_embedding': True,  # if False, then the embedding layer is ignored
            'num_embeddings': 17,
            'embedding_dim': 4,

            'input_dim': 3,
            'hidden_dim': 2, # dimension of this rnn layer
            'output_dim': 2, # dimension of action
            'device': 'cuda',
            'output_h0': True, # whether initial hidden state included in loss
            'trainable_h0': False, # the agent's initial hidden state trainable or not
            'readout_FC': True, # whether the readout layer is full connected or not
            'one_hot': False, # whether the data input is one-hot or not
            ### training info for one model
            'lr':0.005,
            'l1_weight': 1e-5,
            'weight_decay': 0,
            'penalized_weight': 'rec',
            'max_epoch_num': 2000,
            'early_stop_counter': 200,
            ### training info for many models on dataset
            'split_training': True,
            'train_index': train_idx,
            'val_index': val_idx,
            'test_index': test_idx,
            'seed_num': 3,
            ### additional training info
            'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
            'training_diagnose': ['plot_loss'], # can be a list of diagnose function strings, e.g. ['plot_loss']
            ### current training exp path
            'exp_folder': get_training_exp_folder_name(__file__),
        }

        config_ranges = {  # keys are used to generate model names
            'rnn_type': ['GRU'],
            'hidden_dim': [#10,
                20,#50,
                # 100,
                # 200, #400,
            ],
            'l1_weight': [1e-5],
            'include_embedding': [True],
            'embedding_dim': [10,9,8,7,6, 5, 4, 3, 2, 1],
            'trainval_size': [this_trainval_subset_size],
            'distill': ['teacher'], # teacher will use all allowed data on all animals
        }

        resource_dict = {'memory': 14, 'cpu': 1, 'gpu': 1}
        # teacher
        # behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)
        # behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)

        base_config.update({
            'include_embedding': False,
        })
        config_ranges.update({  # keys are used to generate model names
            'include_embedding': [False
                                  ],
        })
        config_ranges.pop('embedding_dim')
        # teacher without embedding
        # behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

        ### for this network, only train on Akam rat 49
        base_config.update({
            'train_index': this_train_subset_idx,
            'val_index': this_val_subset_idx,
            'test_index': this_test_idx,
            'seed_num': 6,
        })
        config_ranges.update({
            'hidden_dim': [4],
            'l1_weight': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'include_embedding': [False],
            'distill': ['none'],
        })
        assert 'embedding_dim' not in config_ranges
        # original
        # behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

        # student
        dt = Dataset('AkamRat',
                     behav_data_spec={'animal_name': 'all', 'max_segment_length': 150, 'include_embedding': True, 'augment': True,})
        base_config.update({
            'behav_data_spec': ['animal_name', 'max_segment_length', 'include_embedding','augment'],
            'train_index': dt.get_after_augmented_block_number(this_train_subset_idx),
            'val_index': dt.get_after_augmented_block_number(this_val_subset_idx),
            'test_index': this_test_idx,
            'augment': True, # whether to augment the data
            'seed_num': 1, #6,
            'teacher_model_path': 'exp_seg_akamrat49_distill/rnn_type-GRU.hidden_dim-20.l1_weight-1e-05.include_embedding-True.embedding_dim-8.trainval_size-XXX.distill-teacher',
        })
        config_ranges.update({
            'hidden_dim': [4],
            'l1_weight': [1e-5,# 1e-4, 1e-3,
                          #1e-2, 1e-1
                          ],
            'include_embedding': [False],
            'distill': ['student'],
        })
        behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)

        # cog models
        base_config = {
            ### dataset info
            'dataset': 'AkamRat',
            'behav_format': 'cog_session',
            'behav_data_spec': ['animal_name', 'max_segment_length'],
            'animal_name': 'all',
            'max_segment_length': 150,
            # 'both' for all blocks
            ### model info
            'agent_type': 'NTSCog',
            'cog_type': 'MB0',
            'device': 'cpu',
            ### training info for one model
            ### training info for many models on dataset
            'split_training': True,
            'train_index': this_train_subset_idx,
            'val_index': this_val_subset_idx,
            'test_index': this_test_idx,
            'seed_num': 16,
            ### additional training info
            'save_model_pass': 'minimal',
            # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
            'training_diagnose': None,  # can be a list of diagnose function strings
            ### current training exp path
            'exp_folder': get_training_exp_folder_name(__file__),
        }

        config_ranges = {  # keys are also used to generate model names
            'cog_type': [
                # 'BAS',
                # 'MF_bs_rb_ck',
                'MF_dec_bs_rb_ck',
                # 'MB_bs_rb_ck',
                # 'MB_dec_bs_rb_ck',
                # 'MF_MB_bs_rb_ck', 'MF_MB_dec_bs_rb_ck',
                # 'MF_MB_vdec_bs_rb_ck',
                # 'MF_MB_bs_rb_ec', 'MF_MB_vdec_bs_rb_ec', 'MF_MB_dec_bs_rb_ec',
                # 'MF_MB_dec_bs_rb_mc', 'MF_MB_dec_bs_rb_ec_mc', 'MFmoMF_MB_dec_bs_rb_ec_mc',
                # 'MFmoMF_dec_bs_rb_ec_mc', 'MB_dec_bs_rb_ec_mc',
                # 'Q(1)', 'MFs',
            ],
            'trainval_size': [this_trainval_subset_size],
            'distill': ['none'],
        }

        # resource_dict = {'memory': 10, 'cpu': 16, 'gpu': 0}
        # behavior_cv_training_job_combination(base_config, config_ranges, resource_dict, n_jobs=-1)

        # behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)