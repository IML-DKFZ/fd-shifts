
import src.models.networks as networks
from src.models.networks.residual_flows.resflow import ResidualFlow
from src.models.networks.residual_flows import layers
import src.models.networks.residual_flows.layers.base as base_layers
from torch import nn
from torch.nn import functional as F
import math
import torch
import numpy as np


class ZhangAndEncoder(nn.Module):
    def __init__(self, cf):
        super().__init__()

        network = networks.get_network(cf.model.network.backbone)(cf)  # todo make arguments explcit!
        self.encoder = network.encoder
        self.classifier = network.classifier
        self.zhang_net = ZhangNet(cf) # todo make arguments explcit!

    def forward(self, x):

        x = self.encoder(x)
        pred_class = self.classifier(x)
        pred_confid = self.zhang_net(x)

        return pred_class, pred_confid


class ZhangBackbone(nn.Module):
    def __init__(self, cf):
        super(ZhangBackbone, self).__init__()

        self.encoder = encoder32(cf)
        self.classifier = classifier32(cf.data.num_classes)

    def forward(self, x):

        x = self.encoder(x)
        x = self.classifier(x)

        return x

class ZhangNet(nn.Module):
    def __init__(self, cf):
        super().__init__()

        # seems like the defaults of the function itself are settings from I-ResNet while the argparse defaults are the settings from ResidualFlows.
        self.input_size = (cf.trainer.batch_size, 512, 1, 1) # b, c, h, w

        self.residual_flow = ResidualFlow(
        input_size = self.input_size,
        n_blocks=[6],  #  4 fc_end are added automatically
        intermediate_dim=256,   # also for fc before fc_end!
        factor_out=True, # if False, the spatial dimensions get reduced throughout blocks, so needs to be true here.
        quadratic=False, # False default, prio 1
        init_layer=layers.LogitTransform(0.05), # This is the logitTransform! So no Squeeze! alpha is 0.05 or 1e-5 for mnist! See RealNVP paper
        actnorm=True,
        fc_actnorm=True,
        batchnorm=False,
        dropout=0,
        fc=True,
        coeff=0.98, #  self normalization: only when sigma larger than coeff: default 0.98 / 0.9 / 0.97. ".
            # We also increase the bound on the Lipschitz constants of each weight matrix to 0.98, whereas
            # Behrmann et al. (2019) used 0.90 to reduce the error of the biased estimator. "
        vnorms='2222', # property of the layers in the resblock: default '2222'. those are the "domains" of lipshcitz layers per kernel in the block
        n_lipschitz_iters=None, # Try different random seeds to find the best u and v inside lipschitz layer
        sn_atol=1e-3, # properties of the lipschitz layer: Algorithm from http://www.qetlab.com/InducedMatrixNorm.
        sn_rtol=1e-3, # properties of the lipschitz layer: Algorithm from http://www.qetlab.com/InducedMatrixNorm.
        n_power_series=None, # None means Unbiased estimation (vs. Truncated). see paper. is set automatically to 20 for eval.
        n_dist='geometric', #'poisson', # Alternative to Brute-force compute Jacobian determinant.
        n_samples=1,
        activation_fn='swish',
        fc_end=True,
        fc_idim=128, # this is only for end!
        n_exact_terms=2, # default prio 1
        preact=True,
        neumann_grad=True, # prio1
        grad_in_forward=True, # default true prio 1 # Do backprop-in-forward to save memory. Could be dangerous! connected to mem_efficent flag!
        first_resblock=False, # # immediate invertible downsampling (Dinh et al., 2017) at the image pixel-level. Removing this substantially increases the
                            # amount of memory required (shown in Figure 3) as there are more spatial dimensions at every layer,
                            # but increases the overall performance. Should not be a problem here.
        learn_p=False, # prio1
        classification=False,
        block_type='resblock')

        self.padding = 0 # todo carefull: padding is for channel dimension.
        self.nvals = 1 # norm factor of the padded values. todo weird: also scales loss if no padding is used! Set to 1?
        self.beta = 1
        self.squeeze_first = False
        self.epsilon_noise =cf.model.epsilon_noise

    def forward(self, x):

        x, logpu = self.add_padding(x, self.nvals) # logpu = 0 without padding

        if self.squeeze_first:
            raise NotImplementedError

        x = clamp_to_unit_sphere(x)
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]

        # x = clamp_to_unit_sphere(x)
        # noise = x.new().resize_as_(x).normal_()
        # x += noise * self.epsilon_noise
        # print("CLAMPED X", x.min(), x.max(), x.mean(), x.std())
        x = x.unsqueeze(2).unsqueeze(3)
        # x = torch.linalg.norm(x) #self.clamp_to_unit_sphere(x, components=x.size()[-1]**2)  # this is typically done in add_noise operation as part of data augmentation.
        # print("CHECK X 1", x.shape, x.min(), x.max())
        z, delta_logp = self.residual_flow(x, 0) # logpx=0 means we want to compute logpx later,
        # so we need to track the logdetgrad term throguhout (output: delta_logp) the flow and start by feeding in 0.
        # this is always done except for their hybrid classification case (irrelevant here).

        # print("CHECK NETWORK OUTPUT", "z", z.shape, z.min().item(), z.max().item(), z.mean().item(), z.std().item())

        logpz = self.standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
        norm =  np.log(self.nvals) * (x.size()[-1] * x.size()[-1] * (x.size()[1] + self.padding)) - logpu # todo weird, ijust get an extra norm here for free via nvals?!
        logpx = logpz - self.beta * delta_logp - norm
        bits_per_dim = - logpx / (x.size()[-1] * x.size()[-1] * x.size()[1]) / np.log(2)
        # print("CHECK NETWORK OUTPUT", "norm", norm.min().item(), norm.max().item())
        # print("CHECK NETWORK OUTPUT", "logpz", logpz.shape, logpz.min().item(), logpz.max().item(), logpz.mean().item())
        # print("CHECK NETWORK OUTPUT", "delta_logp", delta_logp.shape, delta_logp.min().item(), delta_logp.max().item())
        # print("CHECK NETWORK OUTPUT", "logpx", logpx.shape, logpx.min().item(), logpx.max().item(), logpx.mean().item())
        # print("CHECK NETWORK OUTPUT", "bits_per_dim", bits_per_dim.shape, bits_per_dim.mean().item(), bits_per_dim.min().item(), bits_per_dim.max().item())
        # if logpx.mean() > 0:
        #     assert 1 == 2
        return bits_per_dim


    def standard_normal_logprob(self, z):
        logZ = -0.5 * math.log(2 * math.pi)
        # print("CHECK NETWORK OUTPUT", "LOGPROB", logZ, (z.pow(2)/2).mean())
        return logZ - z.pow(2) / 2

    def add_padding(self, x, nvals=256):
        # Theoretically, padding should've been added before the add_noise preprocessing.
        # nvals takes into account the preprocessing before padding is added.
        if self.padding > 0:
           raise NotImplementedError
        else:
            return x, torch.zeros(x.shape[0], 1).to(x)

    def update_lipschitz(self):
        with torch.no_grad():
            for m in self.residual_flow.modules():
                if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                    m.compute_weight(update=True)
                if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                    m.compute_weight(update=True)

    def add_noise(self, x, nvals=256):
        """
        [0, 1] -> [0, nvals] -> add noise -> [0, 1]
        """

        noise = x.new().resize_as_(x).normal_()
        x = x * (nvals - 1) + noise
        x = x / nvals
        return x



class encoder32(nn.Module):
    def __init__(self, cf):
        super(self.__class__, self).__init__()

        self.dropout_rate = cf.model.dropout_rate


        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        # Shortcut out of the network at 8x8
        # self.conv_out_6 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        # Shortcut out of the network at 4x4
        # self.conv_out_9 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv10 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        # self.conv_out_10 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.dr1 = nn.Dropout2d(self.dropout_rate)
        self.dr2 = nn.Dropout2d(self.dropout_rate)
        self.dr3 = nn.Dropout2d(self.dropout_rate)
        self.dr4 = nn.Dropout2d(self.dropout_rate)

        self.apply(weights_init)


    def forward(self, x, output_scale=1):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 8 x 8
        # if output_scale == 8:
        #     x = self.conv_out_6(x)
        #     x = x.view(batch_size, -1)
        #     x = clamp_to_unit_sphere(x, 8*8)
        #     return x

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 4x4
        # if output_scale == 4:
        #     x = self.conv_out_9(x)
        #     x = x.view(batch_size, -1)
        #     x = clamp_to_unit_sphere(x, 4*4)
        #     return x

        x = self.dr4(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 2x2
        # if output_scale == 2:
        #     x = self.conv_out_10(x)
        #     x = x.view(batch_size, -1)
        #     x = clamp_to_unit_sphere(x, 2*2)
        #     return x

        x = x.view(batch_size, -1)
        x = clamp_to_unit_sphere(x)
        return x


class classifier32(nn.Module):
    def __init__(self, num_classes):
        super(self.__class__, self).__init__()

        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):

        return self.fc1(x)


def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def clamp_to_unit_sphere(x, components=1):
    # If components=4, then we normalize each quarter of x independently
    # Useful for the latent spaces of fully-convolutional networks
    batch_size, latent_size = x.shape
    latent_subspaces = []
    for i in range(components):
        step = latent_size // components
        left, right = step * i, step * (i + 1)
        subspace = x[:, left:right].clone()
        norm = torch.norm(subspace, p=2, dim=1)
        subspace = subspace / norm.expand(1, -1).t()  # + epsilon
        latent_subspaces.append(subspace)
    # Join the normalized pieces back together
    return torch.cat(latent_subspaces, dim=1)
