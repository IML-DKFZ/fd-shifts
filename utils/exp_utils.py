import numpy as np
import logging
import subprocess
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn
seaborn.set_style('white')
import sys
import importlib.util
import pickle
import torch
import time
import pandas as pd
import seaborn as sn
import torch.nn as nn
import random

sys.path.append('../')




def prep_exp(in_args, use_stored_settings=True, is_training=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :param is_training: boolean flag. distinguishes train vs. inference mode.
    :return:
    """

    if is_training:

        # the first process of an experiment creates the directories and copies the config to exp_path.
        if not os.path.exists(in_args.exp_dir):
            os.makedirs(in_args.exp_dir)


        # run training with source code info and copy snapshot of model to exp_dir for later testing (overwrite scripts if exp_dir already exists.)
        args = in_args
        args.model_path = os.path.join(in_args.exec_dir, 'models', '{}.py'.format(args.model_script))
        args.experiment_name = args.exp_dir.split("/")[-1]
        args.exp_group = args.exp_dir.split("/")[-2]
        args.created_fold_id_pickle = False
        args.dataset_name = args.dataset_source.split("/")[-1]

    else:
        # for testing copy the snapshot model scripts from exp_dir back to the source_dir as tmp_model / tmp_backbone.
        with open(os.path.join(in_args.exp_dir, 'configs.pickle'), 'rb') as handle:
            args = pickle.load(handle)
            print('check stored graphical model:', args.model)
        tmp_model_path = os.path.join(in_args.exec_dir, 'tmp_models', 'tmp_model_{}.py'.format(args.exp_dir.split('/')[-1]))
        if not os.path.exists(os.path.join(in_args.exec_dir, 'tmp_models')):
            os.mkdir(os.path.join(in_args.exec_dir, 'tmp_models'))
        subprocess.call('cp {} {}'.format(os.path.join(args.exp_dir, 'model.py'), tmp_model_path), shell=True)
        # args.model_path = tmp_model_path
        args.model_path = os.path.join(args.exec_dir, 'models/build_models.py')

    if args.seed is not None:
        set_deterministic(args)

    return args


def apply_test_config_mods(args):
    mods = import_module('mods', os.path.join(('/').join(args.exp_dir.split('/')[:-1]), 'test_args_mods.py'))
    m = mods.get_mods()
    for k, v in m.items():
        setattr(args, k, v)
        print('setting test mod:', k, v)

    return args



def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_deterministic(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_new_train_stage(args, net, p, logger):

    args.loss_weights = args.loss_weights_dict[p]
    logger.info('new train phase: loss weight dict: {}'.format(args.loss_weights))
    logger.info('new train phase: freeze module list: {}'.format(args.freeze_module_dict))

    for name, param in net.named_parameters():
        if name.split('.')[0] in args.freeze_module_dict[p]:
            param.requires_grad = False
            logger.info('new train phase: freezeing {}'.format(name))
        else:
            param.requires_grad = True

    return net

def init_optimizers(args, net):

    optimizers = {}
    # if args.model_script == 'eval_classifier':
    #     return {'G': torch.optim.Adam(net.parameters(), lr=args.learning_rate[0])}
    #
    # else:
    #     enc_list = [net.encoder_x,
    #                net.encoder_y,
    #                net.encoder_xy,
    #                net.decoder_y,
    #               net.decoder_x]
    #
    #     if args.model == 'double_y':
    #         enc_list += [net.encoder_y_spec]
    #         enc_list += [net.decoder_y_spec]
    #
    #     enc_list = nn.ModuleList(enc_list)
    #
    #     optimizers['G'] = torch.optim.Adam(enc_list.parameters(), lr=args.learning_rate[0])
    #     optimizers['D'] = torch.optim.Adam(net.discriminator_zx.parameters(), lr=args.learning_rate[0])
    optimizers['G'] = torch.optim.Adam(net.parameters(), lr=args.learning_rate[0])

    return optimizers


def write_epoch(args, writer, write_scalars_lists, write_figures_lists):


    if write_scalars_lists is not None:

        loss_dict = {}
        for k in list(write_scalars_lists['train'][0].keys()):
            scalar_dict = {}
            for split in (args.data_split_queries.keys()):
                if len(write_scalars_lists[split]) > 0:
                    scalar_dict[split] = np.mean([i[k] for i in write_scalars_lists[split]])
            writer.add_scalars('scalars/{}'.format(k), scalar_dict, args.current_epoch)
            loss_dict.update({'l_{}_{}'.format(k,sp):v for sp, v in scalar_dict.items()})

    for split in (args.data_split_queries.keys()):

        if len(write_figures_lists[split]) > 0:

            results_dict = write_figures_lists[split][0]


            if 'write_time' in list(results_dict.keys()):
                writer.add_scalar('scalars/epoch_time', results_dict['write_time'], args.current_epoch)

            if 'write_images' in list(results_dict.keys()):
                out_plot_dir = os.path.join(args.results_root_path, args.exp_group, 'plots', args.experiment_name)
                if not os.path.exists(out_plot_dir):
                    os.makedirs(out_plot_dir)
                figure_dict = results_dict['write_images']
                f_reco = get_reco_plot(args, figure_dict)
                f_samples = get_sample_plot(args, figure_dict)
                writer.add_figure('{}_reco_plot'.format(split), f_reco, args.current_epoch)
                writer.add_figure('{}_samples_plot'.format(split), f_samples, args.current_epoch)
                f_reco.savefig(os.path.join(out_plot_dir, '{}_{}_{}_reco.png').format(args.experiment_name, split, args.current_epoch))

            if 'write_hists' in list(results_dict.keys()):
                for k in list(results_dict['write_hists'].keys()):
                    writer.add_histogram(split + '_' + k, results_dict['write_hists'][k].flatten(), args.current_epoch)

            if 'write_text' in list(results_dict.keys()):
                for k in list(results_dict['write_text'].keys()):
                    for ix in range(results_dict['write_images'][k].size()[0]):
                        writer.add_text('{}/{}'.format(split + '_' + k, ix), str(map_binary_attributes_to_text(args, results_dict['write_text'][k][ix])), args.current_epoch)

    writer.add_scalar('scalars/halt', 0, 0)

    if write_scalars_lists is not None:
        print(loss_dict)
        return loss_dict



def get_reco_plot(args, results_dict):


    titles = ['input \n', 'reco x \n', 'reco y \n', 'reco xy \n']
    x_columns = ['input_x', 'reco_x_x', 'reco_x_y', 'reco_x_xy']
    y_columns = ['input_y', 'reco_y_x', 'reco_y_y', 'reco_y_xy']

    f = plt.figure(figsize=(13, 30))
    gs = gridspec.GridSpec(args.n_samples_per_val, len(x_columns))
    gs.update(wspace=0.1, hspace=0.1)
    for i in range(args.n_samples_per_val):
        for j in range(len(x_columns)):
            ax = plt.subplot(gs[i, j])
            ax.axis('off')
            if i == 0:
                plt.title(titles[j], fontsize=20)

            img = results_dict[x_columns[j]][i].cpu().permute(1, 2, 0)
            if img.min() < 0:
                img = (img + 1) /2
            ax.imshow(img)

            y = map_binary_attributes_to_text(args, results_dict[y_columns[j]][i])
            for yix in range(np.min((len(y), 15))):
                ax.text(3, args.img_size + 6 + 3.2 * yix  * args.img_size/64, y[yix])

    return f


def get_sample_plot(args, results_dict):

    titles = ['sample_{}'.format(ix) for ix in range(args.n_samples_per_val)]
    f = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(1, args.n_samples_per_val)
    gs.update(wspace=0.1, hspace=0.1)
    for i in range(1):
        for j in range(args.n_samples_per_val):
            ax = plt.subplot(gs[i, j])
            ax.axis('off')
            if i == 0:
                plt.title(titles[j], fontsize=20)

            img = results_dict['sample_x'][j].cpu().permute(1, 2, 0)
            if img.min() < 0:
                img = (img + 1) /2
            ax.imshow(img)

            y = map_binary_attributes_to_text(args, results_dict['sample_y'][j])
            for yix in range(np.min((len(y), 15))):
                ax.text(3, args.img_size + 6 + 3.2 * yix * args.img_size/64, y[yix])

    return f


def write_test(args, writer, results_dict, plot_results=True, agg_results=True, epoch=None, loss_dict=None):

            if 'plot_dict' in list(results_dict.keys()) and plot_results:

                test_concepts_plot = get_test_images_plot(args, results_dict['plot_dict']['concepts'])
           #     writer.add_figure('test_concepts', test_concepts_plot, 0)
           #     writer.add_figure('test_results', get_test_results_plot(args, results_dict['plot_dict']), 0)
           #     writer.add_figure('correctness_confusion_matrix', get_confusion_matrix_plot(args, results_dict['plot_dict']), 0)

                if len(results_dict['plot_dict']['dist_concepts']) > 0:
                    cov_dists_plot = get_coverage_dists_plot(args, results_dict['plot_dict'])
          #          writer.add_figure('coverage_distributions', cov_dists_plot, 0)

                if 'latent_plot' in list(results_dict['plot_dict'].keys()):
                    latent_plot = get_latent_space_plot(results_dict['plot_dict']['latent_plot'])

            study_name = args.attribute_info_pickle_name.split('/')[-1].split('.')[0]

            if args.write_results_to_pickle:
                out_dir = os.path.join(args.results_root_path, args.exp_group, 'results_per_job')
                out_plot_dir = os.path.join(args.results_root_path, args.exp_group, 'plots', args.experiment_name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                if not os.path.exists(out_plot_dir):
                    os.makedirs(out_plot_dir)

                if 'plot_dict' in list(results_dict.keys()) and plot_results:
                    test_concepts_plot.savefig(os.path.join(out_plot_dir, '{}_test_concepts_{}.png').format(args.experiment_name, epoch))
                    if len(results_dict['plot_dict']['dist_concepts']) > 0:
                        cov_dists_plot.savefig(os.path.join(out_plot_dir, '{}_cov_dists_{}.png').format(args.experiment_name, epoch))
                    if 'latent_plot' in list(results_dict['plot_dict'].keys()):
                        latent_plot.savefig(os.path.join(out_plot_dir, '{}_latent_{}.png').format(args.experiment_name, epoch))


                epoch = epoch if epoch is not None else args.num_epochs

                out_dict = {}
                out_dict['exp_name'] = args.experiment_name
                out_dict['time'] = time.ctime(int(time.time()))
                out_dict['model'] = args.model
                out_dict['missing_attr'] = args.missing_attr_mode
                out_dict['attr_encode'] = args.attribute_encode
                out_dict['attr_decode'] = args.attribute_decode
                out_dict['gan_loss'] = args.gan_loss
                out_dict['n_features'] = args.x_path_width
                out_dict['bn_stats_mode'] = args.bn_stats_mode
                out_dict['study_name'] = study_name
                out_dict['epoch'] = epoch
                out_dict['perc_reco'] = args.perceptual_reco
                out_dict['x_dropx'] = args.x_dropx
                out_dict['x_dropxy'] = args.x_dropxy
                out_dict['seed'] = args.seed
                out_dict.update({k:v for k,v in results_dict['write_out'].items()if 'list' not in k})
                if loss_dict is not None:
                    out_dict.update(loss_dict)
                out_df = pd.DataFrame(out_dict, index=[0])
                out_df.to_csv(os.path.join(out_dir, '{}_{}_results.csv'.format(args.experiment_name, epoch, args.seed)))

                if agg_results:
                    print('aggregating results...')
                    aggregate_results(args)

                plt.close()


def aggregate_results(args):

    exp_group_res_dir = os.path.join(args.results_root_path, args.exp_group, 'results_per_job')
    flist = [os.path.join(exp_group_res_dir, i) for i in os.listdir(exp_group_res_dir)]
    dfs = []
    for f in flist:
        dfs.append(pd.read_csv(f))
    out_df = pd.concat(dfs, ignore_index=True, sort=False).round(3)
    out_df.to_csv(os.path.join(args.results_root_path, args.exp_group, 'agg_results.csv'))


def get_test_results_plot(args, plot_dict):

    f = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.0, hspace=0.0)
    ax = plt.subplot(gs[0, 0])
    plt.title('test results')
    plot_results_dict = {k:v for k, v in plot_dict.items() if ('correctness_' in k or 'coverage_' in k or 'compositionality_' in k) and len(v) > 0}
    plt.boxplot(list(plot_results_dict.values()), labels=list(plot_results_dict.keys()), meanline=True, showmeans=True)
    ax.tick_params(axis='x', labelrotation=60)
    plt.tight_layout()
    plt.close()
    return f


def get_test_images_plot(args, plot_list):

    if len(plot_list) > 15:
        random.shuffle(plot_list)
        plot_list = plot_list[:15]
    f = plt.figure(figsize=(2 * args.show_n_test_samples_per_concept, 3 * len(plot_list)))
    gs = gridspec.GridSpec(len(plot_list), args.show_n_test_samples_per_concept)
    gs.update(wspace=0.1, hspace=0.1)
    for ix in range(len(plot_list)):
        for j in range(args.show_n_test_samples_per_concept):
            ax = plt.subplot(gs[ix, j])
            ax.axis('off')
            if j == 0:
                plt.title(plot_list[ix]['type'] + str(map_binary_attributes_to_text(args, plot_list[ix][args.attribute_decode])), fontsize=15)
            img = np.transpose(plot_list[ix]['imgs'][j], axes=(1, 2, 0))
            if img.min() < 0:
                img = (img + 1) /2
            ax.imshow(img)

            y = map_binary_attributes_to_text(args, plot_list[ix]['y_out'][j])
            for yix in range(np.min((len(y), 15))):
                ax.text(3, args.img_size + 6 + 4 * yix * args.img_size / 64, y[yix])

    plt.close()
    return f


def get_confusion_matrix_plot(args, plot_dict):

    f = plt.figure(figsize=(14, 10 * len(list(args.attribute_dict.keys()))))
    gs = gridspec.GridSpec(len(list(args.attribute_dict.keys())), 1)
    gs.update(wspace=0.1, hspace=0.5)
    for kix, k in enumerate(list(args.attribute_dict.keys())):

        try:
            ax = plt.subplot(gs[kix, 0])
            plt.title(k)
            df = pd.DataFrame()
            df['pred'] = plot_dict['confusion_matrix'][k]['pred']
            df['gt'] = plot_dict['confusion_matrix'][k]['gt']
            confusion_matrix = pd.crosstab(df['gt'], df['pred'])
            pred_names = [i for i in args.attribute_dict[k] if args.attribute_dict[k].index(i) in df['pred'].tolist()]
            gt_names = [i for i in args.attribute_dict[k] if args.attribute_dict[k].index(i) in df['gt'].tolist()]
            sn.heatmap(confusion_matrix, annot=True, square=True, cbar=False, fmt=".0f", xticklabels=pred_names, yticklabels=gt_names)
            ax.set_ylim(0, len(args.attribute_dict[k]) + 0)
            ax.tick_params(axis='x', labelrotation=60)
            ax.tick_params(axis='y', labelrotation=60)
        except:
            print('attribute {} not availible for confusion matrix plot'.format(k))

    plt.close()
    return f

def get_coverage_dists_plot(args, plot_dict):

    plot_list = plot_dict['dist_concepts']
    if len(plot_list) > 15:
        random.shuffle(plot_list)
        plot_list = plot_list[:15]
    n_x = np.prod([len(v) for k,v in args.attribute_dict.items()])
    f = plt.figure(figsize=(0.3 * n_x, 5 * len(plot_list)))
    gs = gridspec.GridSpec(len(plot_list), 1)
    gs.update(wspace=0.1, hspace=0.35)
    for cix, c in enumerate(plot_list):

        ax = plt.subplot(gs[cix, 0])
        plt.title(c['string'])
        x = c['q_k_comb']
        y = np.array(c['p_k_comb'])
        bins = range(len(x))
        l = [('').join([list(dict.values())[0][0] for dict in cp]) for cp in c['comb_metric_points']]
        plt.bar(bins, x, alpha=0.5, color='b', tick_label=l)
        plt.bar(bins, y, alpha=0.2, color='r')
        plt.ylim(0, 1)
        plt.xticks(rotation='vertical')
        t = 'cons: {0:0.3f} | cov: {1:0.3f} | comb: {2:0.3f}'.format(c['consistency'], c['coverage'], c['comb_metric'])
        ax.text(0, -0.25, t)

    plt.close()
    return f


def get_latent_space_plot(list_of_plot_info):

    from matplotlib.patches import Ellipse, Wedge, Polygon, Circle
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt


    fig = plt.figure(figsize=(30, 30))
    gs = gridspec.GridSpec(3, 3)
    combs = ['xcyl_ycyl', 'xcyl_xsphere', 'ycyl_ysphere']
    for cix, comb in enumerate(combs):
        for dim_ix in range(3):

            ax = plt.subplot(gs[cix, dim_ix])
            patches_cyl = []
            patches_sph = []
            colors_cyl = []
            colors_sph = []
            alphas = []
            xmin, xmax, ymin, ymax = 0, 0, 0, 0
            for center, rad, c, type in list_of_plot_info['circles_list']:
                center = list(center)
                rad = list(rad)
                center.append(center[0])
                rad.append(rad[0])
                x, y = center[0 + dim_ix], center[1 + dim_ix]
                w, h = rad[0 + dim_ix], rad[1 + dim_ix]
                e = Ellipse((x, y), width=w, height=h)
                if (comb == 'xcyl_xsphere' and type == 'x') or (comb == 'ycyl_ysphere' and type == 'y'):
                    if c[0] == 'cylinder':
                        patches_cyl.append(e)
                        colors_cyl.append(c[1])
                    else:
                        patches_sph.append(e)
                        colors_sph.append(c[1])
                if comb == 'xcyl_ycyl':
                    if c[0] == 'cylinder':
                        if type == 'y':
                            patches_cyl.append(e)
                            colors_cyl.append(c[1])
                        else:
                            patches_sph.append(e)
                            colors_sph.append(c[1])


                if x < xmin:
                    xmin = x
                if y < ymin:
                    ymin = y
                if x > xmax:
                    xmax = x
                if y > ymax:
                    ymax = y


            p = PatchCollection(patches_cyl, alpha=0.6, color=colors_cyl)
            ax.add_collection(p)
            p = PatchCollection(patches_sph, alpha=0.3, color=colors_sph)
            ax.add_collection(p)

            for center, info, correct in list_of_plot_info['points_list']:
                center = list (center)
                center.append(center[0])
                x, y = center[0 + dim_ix], center[1 + dim_ix]
                ax.plot(x, y, color=info[1], alpha=0.6 if correct == 1 else 0.3, marker="o")

            ax.set_xlim(xmin - 3, xmax + 3)
            ax.set_ylim(ymin - 3, ymax + 3)
            ax.set_title('{}_{}'.format(comb, dim_ix))

    plt.close()
    return fig

def save_checkpoint(args, net, optimizers):

    state = {
            'epoch': args.current_epoch,
            'state_dict': net.state_dict(),
        }

    for k in optimizers.keys():
        state['optimizer_state_dict_{}'.format(k)] = optimizers[k].state_dict(),

    torch.save(state, os.path.join(args.checkpoint_path))


def load_checkpoint(args, net, optimizers):

    checkpoint_params = torch.load(args.checkpoint_path)

    net.load_state_dict(checkpoint_params['state_dict'])
    for k in optimizers.keys():
        optimizers[k].load_state_dict(checkpoint_params['optimizer_state_dict_{}'.format(k)])
    args.checkpoint_starting_epoch = checkpoint_params['epoch'] + 1
    return net, optimizers


def load_test_params(args, checkpoint_path, net, load_eval=False):

    checkpoint_params = torch.load(checkpoint_path)

    if args.gan_loss and not load_eval:
        enc_list = torch.nn.ModuleList([net.encoder_x,
                                        net.encoder_y,
                                        net.encoder_xy,
                                        net.decoder_y])

        enc_list.load_state_dict(checkpoint_params['enc_state_dict'])
        net.decoder_x.load_state_dict(checkpoint_params['dec_state_dict'])

    else:
        net.load_state_dict(checkpoint_params['state_dict'])



class _AnsiColorizer(object):
    """
    A colorizer is an object that loosely wraps around a stream, allowing
    callers to write text to the stream in a particular color.
    Colorizer classes must implement C{supported()} and C{write(text, color)}.
    """
    _colors = dict(black=30, red=31, green=32, yellow=33,
                   blue=34, magenta=35, cyan=36, white=37, default=39)

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def supported(cls, stream=sys.stdout):
        """
        A class method that returns True if the current platform supports
        coloring terminal output using this method. Returns False otherwise.
        """
        if not stream.isatty():
            return False  # auto color only on TTYs
        try:
            import curses
        except ImportError:
            return False
        else:
            try:
                try:
                    return curses.tigetnum("colors") > 2
                except curses.error:
                    curses.setupterm()
                    return curses.tigetnum("colors") > 2
            except:
                raise
                # guess false in case of error
                return False

    def write(self, text, color):
        """
        Write the given text to the stream in the given color.
        @param text: Text to be written to the stream.
        @param color: A string label for a color. e.g. 'red', 'white'.
        """
        color = self._colors[color]
        self.stream.write('\x1b[%sm%s\x1b[0m' % (color, text))



class ColorHandler(logging.StreamHandler):


    def __init__(self, stream=sys.stdout):
        super(ColorHandler, self).__init__(_AnsiColorizer(stream))

    def emit(self, record):
        msg_colors = {
            logging.DEBUG: "green",
            logging.INFO: "default",
            logging.WARNING: "red",
            logging.ERROR: "red"
        }
        color = msg_colors.get(record.levelno, "blue")
        self.stream.write(record.msg + "\n", color)


def write_halt(args, logger, writer):

    f = open(args.halt_file_path, 'w')
    job_id = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in list(os.environ.keys()) else ''
    f.write('job_id:{} at \n'.format(job_id, time.ctime(int(time.time()))))
    f.close()
    writer.add_scalar('scalars/halt', 1, 0)
    writer.close()
    logger.info('wrote out halts to file and tb.')
