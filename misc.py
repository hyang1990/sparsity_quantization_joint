import random
import sys
import os
import shutil
import pickle as pkl
import time
import numpy as np
from torch.nn import functional as F

import torch


class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        self.init('/tmp', 'tmp.log')
        self._logger.info(str_info)
logger = Logger()

print = logger.info


def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))


def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ncorrect(output, target, topk=(1,)):
    """Computes the numebr of correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum().item()
        res.append(correct_k)
    return res


def eval_loss_acc1_acc5(model, data_loader, loss_func=None, cuda=True, class_offset=0):
    val_loss = 0.0
    val_acc1 = 0.0
    val_acc5 = 0.0
    num_data = 0
    with torch.no_grad():
        model.eval()
        for data, target in data_loader:
            num_data += target.size(0)
            target.data += class_offset
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            if loss_func is not None:
                val_loss += loss_func(model, data, target).item()
            # val_loss += F.cross_entropy(output, target).item()
            nc1, nc5 = ncorrect(output.data, target.data, topk=(1, 5))
            val_acc1 += nc1
            val_acc5 += nc5
            # print('acc:{}, {}'.format(nc1 / target.size(0), nc5 / target.size(0)))

    val_loss /= len(data_loader)
    val_acc1 /= num_data
    val_acc5 /= num_data

    return val_loss, val_acc1, val_acc5


def validate(val_loader, model, loss_func=None, class_offset=0, verbose=True):
    loss, acc1, acc5 = eval_loss_acc1_acc5(model, val_loader, loss_func=loss_func, cuda=True, class_offset=class_offset)
    if verbose:
        if loss_func is not None:
            print('Loss {:.4f}\t'
                  'Prec@1 {:.4f}\t'
                  'Prec@5 {:.4f}'.format(loss, acc1, acc5))
        else:
            print('Prec@1 {:.4f}\t'
                  'Prec@5 {:.4f}'.format(acc1, acc5))
    return acc1, loss


def apply_weight_decay(weights, weight_decay):
    for p in weights:
        p.grad.data.add_(weight_decay, p.data)


def cross_entropy(input, target):
    return F.cross_entropy(input, target)


def classify_loss(model, data, target, teacher_model, temperature):
    output = model(data)
    if temperature <= 0.0:
        return cross_entropy(output, target)
    else:
        with torch.no_grad():
            teacher_output = teacher_model(data).data
        kd = torch.mean(torch.sum(-F.softmax(teacher_output / temperature, dim=1)
                                  * F.log_softmax(output / temperature, dim=1), dim=1))
        class_loss = cross_entropy(output, target)
        # print("distill loss={:.4e}, class loss={:.4e}".format(kd, class_loss))
        return class_loss + (temperature ** 1) * kd


def load_model_sd(model, sdfilename, strict=True):
    plain_model_sd = torch.load(sdfilename)
    plain_model_sd2 = dict()
    for name in plain_model_sd:
        plain_model_sd2[name.replace('module.', '')] = plain_model_sd[name]
        # if name.startswith('module.'):
        #     warnings.warn("statedict file may saved from nn.dataparallel, auto load")
        #     plain_model_sd2[name[len('module.'):]] = plain_model_sd[name]

    if len(plain_model_sd2) == 0:
        plain_model_sd2 = plain_model_sd

    model.load_state_dict(plain_model_sd2, strict=strict)


def load_pretrained_model(sd_filepath, model, strict=True):
    if sd_filepath is not None and os.path.isfile(sd_filepath):
        print('load pretrained model:{}'.format(sd_filepath))
        # print('current model keys={}'.format(model.state_dict().keys()))
        load_model_sd(model, sdfilename=sd_filepath, strict=strict)
    elif sd_filepath is not None:
        print('fail to load pretrained model: {}'.format(sd_filepath))


def array1d_repr(t, format='{:.3f}'):
    res = ''
    for i in range(len(t)):
        res += format.format(float(t[i]))
        if i < len(t) - 1:
            res += ', '

    return '[' + res + ']'


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def layers_stat(model, param_names=('weight',), param_filter=lambda p: True):
    if isinstance(param_names, str):
        param_names = (param_names,)
    def match_endswith(name):
        for param_name in param_names:
            if name.endswith(param_name):
                return param_name
        return None
    res = "########### layer stat ###########\n"
    for name, W in model.named_parameters():
        param_name = match_endswith(name)
        if param_name is not None:
            if param_filter(W):
                layer_name = name[:-len(param_name) - 1]
                W_nz = torch.nonzero(W.data)
                nnz = W_nz.shape[0] / W.data.numel() if W_nz.dim() > 0 else 0.0
                W_data_abs = W.data.abs()
                res += "{:>20}".format(layer_name) + 'abs(W): min={:.4e}, mean={:.4e}, max={:.4e}, nnz={:.4f}\n'\
                    .format(W_data_abs.min().item(), W_data_abs.mean().item(), W_data_abs.max().item(), nnz)

    res += "########### layer stat ###########"
    return res
