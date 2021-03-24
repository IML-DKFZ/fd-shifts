
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence


def get_loss(args, batch, output_dict):

    out_keys = list(output_dict.keys())
    if 'loss' in out_keys:
        return output_dict


    gt_x = batch['img']
    gt_y = batch['attr']['out_y']
    gt_y_post = batch['attr']['out_y_post']
    gt_yxy = batch['attr']['out_xy']
    y_dropout_mask = batch['attr']['y_dropout_mask']
    reco_x_x = output_dict['reco_x_x']
    reco_y_y = output_dict['reco_y_y']
    q_x = output_dict['q_x']
    q_y = output_dict['q_y']
    q_xy = output_dict['q_xy']
    q_xy_detached = output_dict['q_xy_detached']
    q_x_detached = output_dict['q_x_detached']
    standard_normal_prior = output_dict['sn_prior']
    standard_normal_prior_attr = output_dict['sn_prior_attr']

    # create dummy outputs for simplicity. will be taken out of backprop via loss weights.
    if 'reco_x_xy' in out_keys:
        reco_x_xy = output_dict['reco_x_x']
    else:
        reco_x_xy = [torch.zeros_like(ix).cuda() for ix in reco_x_x]

    if 'reco_y_xy' in out_keys:
        reco_y_xy = output_dict['reco_y_xy']
    else:
        reco_y_xy = torch.zeros_like(reco_y_y).cuda() if args.attribute_decode == 'khot' \
            else [torch.zeros_like(ix).cuda() for ix in reco_y_y]

    if 'reco_y_x' in out_keys:
        reco_y_x = output_dict['reco_y_x']
    else:
        reco_y_x = torch.zeros_like(reco_y_y).cuda() if args.attribute_decode == 'khot' \
            else [torch.zeros_like(ix).cuda() for ix in reco_y_y]

    loss = {}

    # X reconstruction

    if args.x_dec_loss == 'gauss':
        loss['reco_x_x'] = get_likelihood(reco_x_x, gt_x, fixed_variance=args.fixed_model_variance).sum((1, 2, 3)).mean()
        loss['reco_x_xy'] = get_likelihood(reco_x_xy, gt_x, fixed_variance=args.fixed_model_variance).sum((1, 2, 3)).mean()

    else:
        loss['reco_x_x'] = F.binary_cross_entropy(reco_x_x[0], gt_x, reduction='none').sum((1, 2, 3)).mean()
        loss['reco_x_xy'] = F.binary_cross_entropy(reco_x_xy[0], gt_x, reduction='none').sum((1, 2, 3)).mean()


    # Y reconstruction

    if args.attribute_decode == 'categorical':
        # In the categorical case, the output of the model is a list over attribute heads.

        loss_maps_cat_list = []
        for cix, reco_cat in enumerate(reco_y_y):
            loss_maps_cat_list.append(
                F.cross_entropy(reco_cat, gt_y[:, cix].long(), reduction='none').unsqueeze(1))
        # clamp values to avoid exploding gradients on inject bias elements.
        loss['reco_y_y'] = torch.cat(loss_maps_cat_list, 1).clamp(max=5)


        loss_maps_cat_list = []
        for cix, reco_cat in enumerate(reco_y_xy):
            loss_maps_cat_list.append(
                F.cross_entropy(reco_cat, gt_yxy[:, cix].long(), reduction='none').unsqueeze(1))
        loss['reco_y_xy'] = torch.cat(loss_maps_cat_list, 1).sum(1).mean()

    elif args.attribute_decode == 'khot':


        if args.inject_y_noise:
            loss['reco_y_y'] = F.mse_loss(reco_y_y, gt_y, reduction='none')
            loss['reco_y_xy'] = F.mse_loss(reco_y_xy, gt_yxy, reduction='none').sum((1)).mean()
        else:
           # print(reco_y_y[0], gt_y[0])
            loss['reco_y_y'] = F.binary_cross_entropy(torch.sigmoid(reco_y_y), gt_y, reduction='none')
            loss['reco_y_xy'] = F.binary_cross_entropy(torch.sigmoid(reco_y_xy), gt_yxy, reduction='none').sum(1).mean()
            # print(gt_y[0], gt_yxy[0])

        if args.model == 'deterministiccvae' or args.model == 'scan':
            loss['reco_y_x'] = F.binary_cross_entropy(torch.sigmoid(reco_y_x), gt_y_post, reduction='none')
            loss['reco_y_x'] = (loss['reco_y_x'] * y_dropout_mask).sum(1).mean()


    else:
        raise NotImplementedError('attribute code specified in configs is not implemented.')

    loss['reco_y_y'] = (loss['reco_y_y'] * y_dropout_mask).sum(1).mean()

    # KL losses

    if args.model == 'telbo':
        loss['kl_x'] = kl_divergence(q_x, standard_normal_prior).sum(1).mean()
        loss['kl_y'] = kl_divergence(q_y, standard_normal_prior).sum(1).mean()
        loss['kl_xy'] = kl_divergence(q_xy, standard_normal_prior).sum(1).mean()

    elif args.model == 'scan':
        loss['kl_x'] = kl_divergence(q_x, standard_normal_prior).sum(1).mean()
        loss['kl_xy'] = kl_divergence(q_x_detached if args.freeze_kl_flag else q_x, q_y).sum( 1).mean()
        loss['kl_y'] = kl_divergence(q_y, standard_normal_prior).sum(1).mean()

    elif args.model == 'jmvae':
        loss['kl_xy'] = kl_divergence(q_xy, standard_normal_prior).sum(1).mean()
        loss['kl_y'] = kl_divergence(q_xy_detached if args.freeze_kl_flag else q_xy, q_y).sum(1).mean()
        loss['kl_x'] = kl_divergence(q_xy_detached if args.freeze_kl_flag else q_xy, q_x).sum(1).mean()

    elif args.model == 'mpriors':
        if float(args.loss_weights['kl_x'][args.current_epoch - 1]) > 0:
            loss['kl_x'] = kl_divergence(q_x, standard_normal_prior).sum(1).mean()
        if float(args.loss_weights['kl_y'][args.current_epoch - 1]) > 0:
            loss['kl_y'] = torch.FloatTensor([0]).cuda()
            for prior in q_y:
                loss['kl_y'] += kl_divergence(prior, standard_normal_prior_attr).sum(1).mean()

    elif args.model == 'deterministiccvae':

        if args.fader_mode:
            loss['adversarial_disc'] = (F.binary_cross_entropy(torch.sigmoid(output_dict['discriminator_zx_out_detach']), gt_y, reduction='none')).sum(1).mean() * float(args.loss_weights['adversarial_disc'][args.current_epoch - 1])
            loss['adversarial_zx'] = (F.binary_cross_entropy(torch.sigmoid(output_dict['discriminator_zx_out']), batch['attr']['out_y_adversarial'], reduction='none')).sum(1).mean()
        if float(args.loss_weights['kl_x'][args.current_epoch - 1]) > 0:
            loss['kl_x'] = kl_divergence(q_x, standard_normal_prior).sum(1).mean()
        if float(args.loss_weights['kl_xy'][args.current_epoch - 1]) > 0:
            loss['kl_xy'] = kl_divergence(q_x, q_y).sum(1).mean()
        if float(args.loss_weights['kl_y'][args.current_epoch - 1]) > 0:
            loss['kl_y'] = kl_divergence(q_y, standard_normal_prior).sum(1).mean()

    elif args.model == 'double_y':

        loss['adversarial_disc'] = (F.binary_cross_entropy(torch.sigmoid(output_dict['discriminator_zx_out_detach']),
                                                          gt_y, reduction='none')).sum(1).mean() * float(
            args.loss_weights['adversarial_disc'][args.current_epoch - 1])
        loss['adversarial_zx'] = (F.binary_cross_entropy(torch.sigmoid(output_dict['discriminator_zx_out']),
                                                         batch['attr']['out_y_adversarial'], reduction='none')).sum(1).mean()
        loss['kl_x'] = kl_divergence(q_x, standard_normal_prior).sum(1).mean()
        loss['kl_y'] = kl_divergence(output_dict['q_y_spec'], q_y).sum(1).mean()
        loss['reco_y_y_spec'] = F.binary_cross_entropy(torch.sigmoid(output_dict['reco_y_y_spec']), gt_yxy, reduction='none').sum(1).mean()

    else:
        raise NotImplementedError('graphical model specified in configs is not implemented.')

    # Loss weighting and aggregation

    total_loss = torch.zeros(1).cuda()
    for k, v in loss.items():
        if not 'disc' in k:
            if type(args.loss_weights[k]) == float or type(args.loss_weights[k]) == int:
                w = args.loss_weights[k]
            else:
                w = float(args.loss_weights[k][args.current_epoch - 1])
            total_loss += v * w
    loss['total'] = total_loss

    return loss



def get_likelihood(pred, target, fixed_variance): # only works if pred is a list of [mu, sigma]

    if fixed_variance is not None:
        variance = fixed_variance
        log_sigma = 0
    else:
        sigma = pred[1].clamp(min=0.1)
        variance = sigma ** 2
        log_sigma = sigma.log()

    return (((pred[0] - target) ** 2) / (2 * variance)) + log_sigma



def get_results(args, batch, output_dict, loss, write_figures=False):


    results_dict = {}
    results_dict['write_scalars'] = {k: v.item() for k, v in loss.items()}
    results_dict['write_figures'] = {}

    results_dict['logger_string'] = ''
    for k, v in results_dict['write_scalars'].items():
        results_dict['logger_string'] += str(k) + ': {0:0.3f} '.format(v)


    if write_figures:

        in_x = batch['img']
        gt_yxy = batch['attr']['out_xy']
        reco_x_x = output_dict['reco_x_x']
        q_x = output_dict['q_x']
        q_y = output_dict['q_y']
        q_xy = output_dict['q_xy']

        if args.model == 'mpriors':
            q_y = q_y[0]

        reco_y_y = map_y_output(args, output_dict['reco_y_y'])

        results_dict['write_figures']['write_images'] = {'input_x': in_x[:args.n_samples_per_val],
                                                         'input_y': gt_yxy[:args.n_samples_per_val],
                                                         'reco_x_x': reco_x_x[0][:args.n_samples_per_val],
                                                         'reco_y_y': reco_y_y[:args.n_samples_per_val],
                                                        }

        results_dict['write_figures']['write_hists'] = {'mu_x': q_x.loc,
                                                        'sigma_x': q_x.scale,
                                                        'mu_y': q_y.loc,
                                                        'sigma_y': q_y.scale,
                                                        'mu_xy': q_xy.loc,
                                                        'sigma_xy': q_xy.scale}

        # create dummy outputs for simplicity. will be taken out of backprop via loss weights.
        out_keys = list(output_dict.keys())
        for k in ['reco_x_xy', 'reco_x_y', 'sample_x']:
            if k in out_keys:
                x = output_dict[k]
                if not k == 'sample_x':
                    x = x[0]
            else:
                x = torch.zeros_like(reco_x_x[0]).cuda()
            results_dict['write_figures']['write_images'][k] = x[:args.n_samples_per_val]

        for k in ['reco_y_xy', 'reco_y_x', 'sample_y']:
            if k in out_keys:
                y = map_y_output(args, output_dict[k])
            else:
                y = torch.zeros_like(reco_y_y).cuda() if args.attribute_decode == 'khot' \
                    else [torch.zeros_like(ix).cuda() for ix in reco_y_y]
            results_dict['write_figures']['write_images'][k] = y[:args.n_samples_per_val]


    return results_dict



def map_y_output(args, y_out):

    if args.attribute_decode == 'categorical':
        y_out = torch.cat([y.argmax(1).unsqueeze(1) for y in y_out], 1)  # b, attr: argmax
    elif args.attribute_decode == 'khot':
        y_out = torch.sigmoid(y_out)
    return y_out



class NDConvGenerator(object):
    """
    generic wrapper around conv-layers to avoid 2D vs. 3D distinguishing in code.
    """
    def __init__(self, dim=2):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, transpose=False, relu='relu', bn_stats_mode='none'):
        """
        :param c_in: number of in_channels.
        :param c_out: number of out_channels.
        :param ks: kernel size.
        :param pad: pad size.
        :param stride: kernel stride.
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :return: convolved feature_map.
        """
        if not transpose:
            conv = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride, bias=False)

        else:
            conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride, bias=False)

        track_bn_mean = False if bn_stats_mode == 'none' else True
        if norm is not None and norm != 'none':
            if norm == 'instance_norm':
                norm_layer = nn.InstanceNorm2d(c_out)
            elif norm == 'batch_norm':
                norm_layer = nn.BatchNorm2d(c_out, track_running_stats=track_bn_mean) # higher momentum = more history
            else:
                raise ValueError('norm type as specified in configs is not implemented...')
            conv = nn.Sequential(conv, norm_layer)

        if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError('relu type as specified in configs is not implemented...')
            conv = nn.Sequential(conv, relu_layer)

        return conv