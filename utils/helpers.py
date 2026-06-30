import torch
import collections


def load_state(net, checkpoint):
    """Load model state dict, handling DDP 'module.' prefix."""
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        target_key_new = 'module.' + target_key
        if target_key_new in source_state and source_state[target_key_new].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key_new]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)


def make_optimizer(cfg, parameters):
    optimizer = torch.optim.Adam(
        parameters,
        lr=cfg.base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay
    )
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    w_iters = cfg.warm_up_iters
    w_fac = cfg.warm_up_factor
    max_iter = cfg.max_iter
    lr_lambda = lambda iteration: (
        w_fac + (1 - w_fac) * iteration / w_iters
        if iteration < w_iters
        else 1 - (iteration - w_iters) / (max_iter - w_iters)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    return scheduler
