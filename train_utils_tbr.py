import torch
import torch.nn as nn
from models import *
from utils.statics import AverageMeter, evaluator
from operator import itemgetter
import numpy as np
import math
import tqdm


# Training
def train(epoch, net, num_epochs, data_loader, criterion, optimizer, optim_type='SGD'):
    net.train()
    net.training = True
    iter_loss = AverageMeter('Iter loss')

    print(f'Training Epoch {epoch}')
    pbar = tqdm.tqdm(total=len(data_loader), desc="Training")
    for batch_idx, (sparse_gt,) in enumerate(data_loader):
        sparse_gt = sparse_gt.cuda()
        optimizer.zero_grad()
        sparse_pred = net(sparse_gt)
        loss = criterion(sparse_pred, sparse_gt)
        if optim_type == 'Adahessian':
            loss.backward(create_graph=True)
        else:
            loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update
        iter_loss.update(loss)
        pbar.update(1)
        # sys.stdout.write('\r')
        # sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
        #        %(epoch, num_epochs, batch_idx+1, len(trainloader), loss.item(), 100.*correct/total) )
        # sys.stdout.flush()
    pbar.close()
    return iter_loss.avg


def test(epoch, net, data_loader, criterion):
    net.eval()
    net.training = False
    iter_rho = AverageMeter('Iter rho')
    iter_nmse = AverageMeter('Iter nmse')
    iter_loss = AverageMeter('Iter loss')
    with torch.no_grad():
        for batch_idx, (sparse_gt, raw_gt) in enumerate(data_loader):
            sparse_gt = sparse_gt.cuda()
            sparse_pred = net(sparse_gt)
            loss = criterion(sparse_pred, sparse_gt)
            rho, nmse = evaluator(sparse_pred, sparse_gt, raw_gt)
            iter_loss.update(loss)
            iter_rho.update(rho)
            iter_nmse.update(nmse)
    return iter_loss.avg, iter_rho.avg, iter_nmse.avg


# Return network & file name
def getNetwork(args):
    if args.net_type == 'CLNet':
        net = clnet(reduction=64)
        file_name = 'clnet'
    elif args.net_type == 'CRNet':
        net = crnet(reduction=64)
        file_name = 'crnet'
    elif args.net_type == 'CsiNet':
        net = csinet(encoded_dim=32)
        file_name = 'csinet'
    elif args.net_type == 'CsiNet+':
        net = csinetplus(reduction=64)
        file_name = 'csinetplus'

    return net, file_name


def net_esd_estimator(
        net=None,
        EVALS_THRESH=0.00001,
        bins=100,
        fix_fingers=None,
        xmin_pos=2,
        conv_norm=0.5,
        filter_zeros=False):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha': [],
        'spectral_norm': [],
        'D': [],
        'longname': [],
        'eigs': [],
        'norm': [],
        'alphahat': []
    }
    print("=================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone()
            # i have checked that the multiplication won't affect the weights value
            # print("before", torch.max(m.weight.data))
            # normalization and tranpose Conv2d
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            # ascending order
            eigs, _ = torch.sort(eigs, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()

            if filter_zeros:
                # print(f"{name} Filter Zero")
                nz_eigs = eigs[eigs > EVALS_THRESH]
                N = len(nz_eigs)
                # somethines N may equal 0, if that happens, we don't filter eigs
                if N == 0:
                    # print(f"{name} No non-zero eigs, use original total eigs")
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                # print(f"{name} Skip Filter Zero")
                nz_eigs = eigs
                N = len(nz_eigs)

            log_nz_eigs = torch.log(nz_eigs)

            if fix_fingers == 'xmin_mid':
                i = int(len(nz_eigs) / xmin_pos)
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n).cuda()
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(
                    1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n
                ))
            else:
                alphas = torch.zeros(N - 1)
                Ds = torch.ones(N - 1)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                    boundaries = torch.linspace(min_e, max_e, bins + 1)
                    h = counts, boundaries
                    ih = torch.argmax(h[0])  #
                    xmin2 = 10 ** h[1][ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2

                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break

                    n = float(N - i)
                    seq = torch.arange(n).cuda()
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n
                        ))

                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]

            final_alpha = final_alpha.item()
            final_D = final_D.item()
            final_alphahat = final_alpha * math.log10(spectral_norm)

            results['spectral_norm'].append(spectral_norm)
            results['alphahat'].append(final_alphahat)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.detach().cpu().numpy())

    return results


def evals_esd_estimator(
        eigs_lst=None,
        EVALS_THRESH=0.00001,
        bins=100,
        fix_fingers=None,
        xmin_pos=2):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha': [],
        'spectral_norm': [],
        'D': [],
        'name': []
    }
    # iterate through layers
    for eigs in eigs_lst:
        eigs, _ = torch.sort(eigs)
        spectral_norm = eigs[-1].item()
        results['spectral_norm'].append(spectral_norm)
        nz_eigs = eigs[eigs > EVALS_THRESH]

        N = len(nz_eigs)
        log_nz_eigs = torch.log(nz_eigs)

        if fix_fingers == 'xmin_mid':
            i = len(nz_eigs) // xmin_pos
            xmin = nz_eigs[i]
            n = float(N - i)
            seq = torch.arange(n).cuda()
            final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
            final_D = torch.max(torch.abs(
                1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n
            ))
        else:
            alphas = torch.zeros(N - 1)
            Ds = torch.ones(N - 1)
            if fix_fingers == 'xmin_peak':
                hist_nz_eigs = torch.log10(nz_eigs)
                min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                boundaries = torch.linspace(min_e, max_e, bins + 1)
                h = counts, boundaries
                ih = torch.argmax(h[0])  #
                xmin2 = 10 ** h[1][ih]
                xmin_min = torch.log10(0.95 * xmin2)
                xmin_max = 1.5 * xmin2

            for i, xmin in enumerate(nz_eigs[:-1]):
                if fix_fingers == 'xmin_peak':
                    if xmin < xmin_min:
                        continue
                    if xmin > xmin_max:
                        break

                n = float(N - i)
                seq = torch.arange(n).cuda()
                alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                alphas[i] = alpha
                if alpha > 1:
                    Ds[i] = torch.max(torch.abs(
                        1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n
                    ))

            min_D_index = torch.argmin(Ds)
            final_alpha = alphas[min_D_index]
            final_D = Ds[min_D_index]

        final_alpha = final_alpha.item()
        final_D = final_D.item()

        results['alpha'].append(final_alpha)
        results['D'].append(final_D)

    return results


def get_layer_temps(args, temp_balance, n_alphas, epoch_val):
    """

    Args:
        temp_balance (_type_): method type
        n_alphas (_type_): all the metric values
        epoch_val (_type_): basic untuned learning rate
    """
    n = len(n_alphas)
    idx = [i for i in range(n)]
    temps = np.array([epoch_val] * n)

    if temp_balance == 'tbr':
        print("--------------------> Use tbr method to schedule")
        idx = np.argsort(n_alphas)
        # temps = [2 * epoch_val * (0.35 + 0.15 * 2 * i / n) for i in range(n)]
        temps = [epoch_val * (args.lr_min_ratio + args.lr_slope * i / n) for i in range(n)]
        # print("temps",    args.lr_min_ratio,  args.lr_slope )
        # print("temps", temps)
        # Examples:
        # 4 3 5 -> argsort -> 1 0 2
        # temps = [0.7, 1, 1.3]
        # zip([1, 0, 2], [0.7, 1, 1.3]) -> [(1, 0.7), (0, 1), (2, 1.3)] -> [(0, 1),(1, 0.7),(2, 1.3)]
        return [value for _, value in sorted(list(zip(idx, temps)), key=itemgetter(0))]
    elif temp_balance == 'tb_linear_map':
        lr_range = [args.lr_min_ratio * epoch_val, (args.lr_min_ratio + args.lr_slope) * epoch_val]
        score_range = [min(n_alphas), max(n_alphas)]
        temps = np.interp(n_alphas, score_range, lr_range)
        return temps

    elif temp_balance == 'tb_sqrt':
        temps = np.sqrt(n_alphas) / np.sum(np.sqrt(n_alphas)) * n * epoch_val
        return temps

    elif temp_balance == 'tb_log2':
        temps = np.log2(n_alphas) / np.sum(np.log2(n_alphas)) * n * epoch_val
        return temps
    else:
        raise NotImplementedError
