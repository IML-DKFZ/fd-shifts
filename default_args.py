import argparse
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

def Args():


    parser = argparse.ArgumentParser()


    parser.add_argument('--mode', type=str,  default='train_test',
                        help='one out of: train / test / train_test / analysis / create_exp')
    parser.add_argument('--folds', nargs='+', type=int, default=[0],
                        help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='tb_test',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--slurm_job_id', type=str, default=None, help='job scheduler info')
    parser.add_argument('--use_stored_settings', default=False, action='store_true',
                        help='load configs from existing exp_dir instead of source dir. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--dataset_source', type=str, default='loaders',
                        help='specifies, from which source experiment to load configs and data_loader.')
    parser.add_argument('--exec_dir', type=str, default=None,
                        help='None, if script is launched from this directory, else needs to be specified.')
    parser.add_argument('--apply_test_config_mods', action='store_true', default=False,
                        help='apply config mods. at test time. requires respective script in exp_dir')
    parser.add_argument('--apply_test_attr_pickle', action='store_true', default=True,
                        help='apply attribute pickle at test time')
    parser.add_argument('--check_pretrained_model', action='store_true', default=False,
                        help='load checkpoint if exists with, has to be true for slurm submissions')

    parser.add_argument('--pin_memory', action='store_true', default=False,
                        help='???')

    parser.add_argument('--confidence_mode', type=str, default='mpc')
    parser.add_argument('--data_dir', type=str, default='svhn')
    parser.add_argument('--save_path', type=str, default='/checkpoint/pjaeger/')
    parser.add_argument('--model_script', type=str, default='small_conv')
    parser.add_argument('--seed', type=int, default=1, help='if None, training will be non-deterministic')
    parser.add_argument('--img_size', nargs='*', default=[32,32,3])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', nargs='*', default=[5*1e-4]*30 + [1e-4] * 40 + [5*1e-5] * 30)
    parser.add_argument('--do_val', action='store_true', default=True, help='run_validation')
    parser.add_argument('--skip_monitoring_figures', action='store_true', default=False)
    parser.add_argument('--test_monitor_interval', type=int, default=5)
    parser.add_argument('--plot_test_results_in_monitoring', action='store_true', default=True)
    parser.add_argument('--val_monitor_interval', type=int, default=5)
    parser.add_argument('--val_size', type=int, default=0.1)
 #   parser.add_argument('--enlarge_train_split_factor', type=int, default=40) # 40 # gives roughly 300 steps per epcoh as opposed to 1500 in clean study

    parser.add_argument('--augmentations_normalize', nargs='*', default=[[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])

    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--checkpoint_name', type=str, default='last_checkpoint.pth')
    parser.add_argument('--tensorboard_name', type=str, default=str(time.time()))
    parser.add_argument('--num_workers', type=int, default=12)

    parser.add_argument('--relu', type=str, default='relu')
    parser.add_argument('--norm', type=str, default='batch_norm')
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--write_results_to_pickle', action='store_true', default=True)
    parser.add_argument('--results_root_path', type=str, default='/private/home/pjaeger/fs',
                            help='path to trained eval classifier model')
    parser.add_argument('--info_df_name', type=str, default='uniform_df_complex.csv')
    parser.add_argument('--query_monitors', nargs='*', default=['accuracy', 'failauc', 'failap_suc', 'failap_err', "mce", "ece", "aurcs", "default_plot"])

    parser.add_argument('--dataset_mean', type=float, default=123.376)
    parser.add_argument('--dataset_std', type=float, default=19.389)


    args = parser.parse_args()
    print(args)

    args.train_stages = [-1] * args.num_epochs
    # if training stages > 1:
    # if args.model_group == 'vae':
    #     args.train_stages = []
    #     dargs = vars(args)
    #     args.loss_weights_dict = {}
    #     args.freeze_module_dict = {}
    #     str_p = [''] + ['_{}'.format(ix) for ix in range(2, args.num_train_stages + 2)]
    #     for pix in range(args.num_train_stages):
    #
    #         args.train_stages.extend([pix] * (args.num_epochs // args.num_train_stages))
    #
    #         p = str_p[pix]
    #         loss_weights = {}
    #         loss_weights['reco_x_xy'] = dargs['weight_reco_x_xy{}'.format(p)]
    #         loss_weights['reco_x_x'] = dargs['weight_reco_x_x{}'.format(p)]
    #         loss_weights['reco_y_xy'] = dargs['weight_reco_y_xy{}'.format(p)]  # lambda  #important weight for trade off corr vs cov
    #         loss_weights['reco_y_y'] = dargs['weight_reco_y_y{}'.format(p)]  # gamma #important weight for trade off corr vs cov
    #         loss_weights['reco_y_x'] = dargs['weight_reco_y_x{}'.format(p)]  # gamma #important weight for trade off corr vs cov
    #         loss_weights['kl_x'] = dargs['weight_kl_x{}'.format(p)]  # beta_x
    #         loss_weights['kl_y'] = dargs['weight_kl_y{}'.format(p)]  # beta_y
    #         loss_weights['kl_xy'] = dargs['weight_kl_xy{}'.format(p)]  # beta_xy
    #         loss_weights['gan_G_post_x'] = dargs['weight_gan_G_post_x{}'.format(p)]
    #         loss_weights['gan_G_post_xy'] = dargs['weight_gan_G_post_xy{}'.format(p)]
    #         loss_weights['gan_G_prior'] = dargs['weight_gan_G_prior{}'.format(p)]
    #         loss_weights['gan_D_post_x'] = dargs['weight_gan_D_post_x{}'.format(p)]
    #         loss_weights['gan_D_post_xy'] = dargs['weight_gan_D_post_xy{}'.format(p)]
    #         loss_weights['gan_D_prior'] = dargs['weight_gan_D_prior{}'.format(p)]
    #         loss_weights['reco_x_y'] = dargs['weight_reco_x_y{}'.format(p)]
    #         loss_weights['adversarial_disc'] = dargs['weight_adversarial_disc{}'.format(p)]
    #         loss_weights['adversarial_zx'] = dargs['weight_adversarial_zx{}'.format(p)]
    #         loss_weights['reco_y_y_spec'] = dargs['weight_reco_y_y_spec{}'.format(p)]
    #
    #         args.loss_weights_dict[pix] = loss_weights
    #         args.freeze_module_dict[pix] = [('_').join(x.split('_')[:-1]) for x in args.freeze_module_list if int(x.split('_')[-1]) == pix]
    #
    #     args.train_stages.extend([args.train_stages[-1]] * (args.num_epochs - len(args.train_stages)))
    #     assert len(args.train_stages) == args.num_epochs, [len(args.train_stages), args.num_epochs]
    #     args.fill_missing_value = 0 if 'khot' in args.attribute_encode else -1
    #     args.learning_rate = [float(ix) for ix in args.learning_rate]


    args.dataloader_args = {'num_workers': args.num_workers}
    args.augmentations = {}
    if args.augmentations_normalize is not None:
        args.augmentations["normalize"] = args.augmentations_normalize
    args.exp_dir = os.path.join(os.environ["EXPERIMENT_ROOT_DIR"],args.exp_dir)
    args.learning_rate = [float(ix) for ix in args.learning_rate]
    args.data_split_queries = ["train", "val", "test"]


    return args