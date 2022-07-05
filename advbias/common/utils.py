import os
import torch
import contextlib


def rescale_intensity(data, new_min=0, new_max=1, eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs*c, -1)
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values
    new_data = (data - old_min) / (old_max - old_min + eps) * \
        (new_max-new_min)+new_min
    new_data = new_data.view(bs, c, h, w)
    return new_data


def check_dir(dir_path, create=False):
    '''
    check the existence of a dir, when create is True, will create the dir if it does not exist.
    dir_path: str.
    create: bool
    return:
    exists (1) or not (-1)
    '''
    if os.path.exists(dir_path):
        return 1
    else:
        if create:
            os.makedirs(dir_path)
        return -1


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(model, new_state=None, hist_states=None):
        """[summary]

        Args:
            model ([torch.nn.Module]): [description]
            new_state ([bool], optional): [description]. Defaults to None.
            hist_states ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        old_states = {}
        for name, module in model.named_children():
            if hasattr(module, 'track_running_stats'):
                old_state = module.track_running_stats
                if hist_states is not None:
                    module.track_running_stats = hist_states[name]
                else:
                    if new_state is not None:
                        module.track_running_stats = new_state
                old_states[name] = old_state
        return old_states

    old_states = switch_attr(model, False)
    yield
    switch_attr(model, old_states)


def set_grad(module, requires_grad=False):
    for p in module.parameters():  # reset requires_grad
        p.requires_grad = requires_grad
