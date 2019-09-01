import torch
import torch.nn as nn
import torch.nn.functional as F


def get_padding(kernel_size, stride=None, h_or_w=None):
    unused = (h_or_w - 1) % stride if h_or_w and stride else 0
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg - unused
    return (pad_beg, pad_end, pad_beg, pad_end)


def slice2d(x, padding):
    pad_t, pad_b, pad_l, pad_r = padding
    return x.narrow(2, pad_t, x.size(2) - pad_t - pad_b).narrow(3, pad_l, x.size(3) - pad_l - pad_r)


def conv2d(in_channels, out_channels, kernel_size, stride, bias=True, in_h_or_w=None):
    if kernel_size > 1:
        return nn.Sequential(
            nn.ZeroPad2d(get_padding(kernel_size, stride, in_h_or_w)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias))
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)


def deconv2d(in_channels, out_channels, kernel_size, stride, bias=True, out_h_or_w=None):
    if kernel_size > 1:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias),
            Lambda(slice2d, get_padding(kernel_size, stride, out_h_or_w)))
    else:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)


def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Lambda(nn.Module):
    def __init__(self, fn, *args):
        super(Lambda, self).__init__()
        self.args = args
        self.fn = fn

    def forward(self, x):
        return self.fn(x, *self.args)


def gaussian_kl_divergence(z_mean, z_logvar):
    # 1/B sum_i{ 1/2 * sum_d{ mu_d^2 + sigma_d^2 - log(sigma_d^2) - 1 } }
    kl_i = 0.5 * torch.sum(z_mean * z_mean + z_logvar.exp() - z_logvar - 1, 1)  # B
    return torch.mean(kl_i)


def sample_from_gaussian(z_mean, z_logvar):
    noise = torch.randn_like(z_mean)
    return noise * (z_logvar / 2).exp() + z_mean


def bernoulli_reconstruction_loss(recon_x, x):
    bs = x.size(0)
    x = x.reshape(bs, -1)
    recon_x = recon_x.reshape(bs, -1)
    loss_per_sample = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none').sum(1)
    clamped_x = x.clamp(1e-6, 1-1e-6)
    loss_lower_bound = F.binary_cross_entropy(clamped_x, x, reduction='none').sum(1)
    loss = (loss_per_sample - loss_lower_bound).mean()
    return loss


def sample_from_gumbel_softmax(y_logits, tau=1):
    dim = 1
    gumbels = - torch.log(torch.empty_like(y_logits).exponential_() + 1e-20)  # ~ Gumbel(0,1)
    gumbels = (y_logits + gumbels) / tau  # ~ Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    _, y_index = y_soft.max(dim, keepdim=True)
    y_hard = torch.zeros_like(y_logits).scatter_(dim, y_index, 1.0)
    y_hard = y_hard - y_soft.detach() + y_soft
    return y_index, y_soft, y_hard


def categorical_mmd_from_uniform(y, n):
    B = y.size(0)
    gram = y.mm(y.t())  # B x B
    mmd = gram.sum() / B ** 2 - 1 / n
    return mmd


def pairwise_categorical_hsic(y1, y2):
    B = y1.size(0)
    gram1 = y1.mm(y1.t())  # B x B
    gram2 = y2.mm(y2.t())  # B x B
    hsic = ((gram1 * gram2).sum() / B ** 2
            + gram1.flatten().unsqueeze(1).mm(gram2.flatten().unsqueeze(0)).sum() / B ** 4
            - 2 * gram1.mm(gram2).sum() / B ** 3)
    return hsic


def pairwise_gaussian_hsic(z1, z2):
    B = z1.size(0)
    z1_ss = z1.pow(2).sum(1)  # B
    gram1 = torch.exp(-0.5 * (z1_ss.unsqueeze(0) + z1_ss.unsqueeze(1) - 2 * z1.mm(z1.t())))  # B x B
    z2_ss = z2.pow(2).sum(1)  # B
    gram2 = torch.exp(-0.5 * (z2_ss.unsqueeze(0) + z2_ss.unsqueeze(1) - 2 * z2.mm(z2.t())))  # B x B
    hsic = ((gram1 * gram2).sum() / B ** 2
            + gram1.flatten().unsqueeze(1).mm(gram2.flatten().unsqueeze(0)).sum() / B ** 4
            - 2 * gram1.mm(gram2).sum() / B ** 3)
    return hsic


def conditional_entropy_upper_bound(y_logits, y_logits_2):
    p = F.softmax(y_logits, 1)  # B x n
    log_p_2 = F.log_softmax(y_logits_2, 1)  # B x n
    return (- p * log_p_2).sum(1).mean()


def gather(batch_tensor, dim_index):
    return batch_tensor.gather(1, dim_index)


def get_fc_operator(type_, n_dims, n_dims_sm):
    if type_ == 'I':
        return nn.Sequential(nn.ReLU(), nn.Linear(n_dims, n_dims_sm),
                             nn.ReLU(), nn.Linear(n_dims_sm, n_dims))
    elif type_ == 'II':
        return nn.Sequential(nn.ReLU(), nn.Linear(n_dims, n_dims_sm),
                             nn.ReLU(), nn.Linear(n_dims_sm, n_dims_sm),
                             nn.ReLU(), nn.Linear(n_dims_sm, n_dims))
    else:
        raise ValueError('Invalid fc operator type')


def get_fc_switch(type_, n_dims, n_dims_sm, n_branches):
    if type_ == 'I':
        return nn.Sequential(nn.ReLU(), nn.Linear(n_dims, 3 * n_branches))
    elif type_ == 'II':
        return nn.Sequential(nn.ReLU(), nn.Linear(n_dims, n_dims_sm),
                             nn.ReLU(), nn.Linear(n_dims_sm, 3 * n_branches))
    else:
        raise ValueError('Invalid fc switch type')


def compute_fc_operators(fcs, n_branches, x, y_idx, y, z, backward_on_y):
    sp_sum = None
    for i in range(n_branches):
        sub_idx = y_idx.squeeze(1).eq(i).nonzero().squeeze(1)  # sub_B
        if sub_idx.size(0) == 0:
            continue
        sub_z = z[:, i].index_select(0, sub_idx).unsqueeze(1)  # sub_B x 1
        sub_x = x.index_select(0, sub_idx)  # sub_B x n_dims
        sub_out = fcs[i](sub_x) * sub_z  # sub_B x n_dims
        if (y is not None) and backward_on_y:
            sub_y = y[:, i].index_select(0, sub_idx).unsqueeze(1)  # sub_B x 1
            sub_out = sub_out * sub_y  # sub_B x n_dims

        sp_out = torch.sparse_coo_tensor(sub_idx.unsqueeze(0), sub_out, x.size(), device=sub_out.device)  # (sparse) B x n_dims
        sp_sum = sp_out if sp_sum is None else sp_sum + sp_out
    if sp_sum is None:
        sp_sum = None
    return sp_sum.to_dense()  # B x n_dims
