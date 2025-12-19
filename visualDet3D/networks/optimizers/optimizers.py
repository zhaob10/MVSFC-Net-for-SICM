from ast import keyword
from easydict import EasyDict 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools

def build_optimizer(optim_cfg:EasyDict, model:nn.Module):
    if optim_cfg.type_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), **(optim_cfg.keywords))
    if optim_cfg.type_name.lower() == 'adam':
        return optim.Adam(model.parameters(), **(optim_cfg.keywords))
    if optim_cfg.type_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), **(optim_cfg.keywords))
    raise NotImplementedError(optim_cfg)

def build_optimizer_VCM(optim_cfg:EasyDict, model:nn.Module):
    # if optim_cfg.type_name.lower() == 'sgd':
    #     return optim.SGD(model.parameters(), **(optim_cfg.keywords))
    if optim_cfg.type_name.lower() == 'adam':
        codec = model.core.stereo_compression
        recon = model.core.reconstruction

        codec_params, aux_params = Split_Params(codec)
        detec_params, _ = Split_Params(model)

        codec_optimizer = optim.Adam(params=itertools.chain(codec_params, recon.parameters()), \
            lr=optim_cfg.keywords.lr, weight_decay=optim_cfg.keywords.weight_decay)

        aux_optimizer = optim.Adam(params=aux_params, \
            lr=10 * optim_cfg.keywords.lr, weight_decay=optim_cfg.keywords.weight_decay)

        detector_optimizer = optim.Adam(params=detec_params, \
            lr= optim_cfg.keywords.lr, weight_decay=optim_cfg.keywords.weight_decay)

        return aux_optimizer, codec_optimizer, detector_optimizer
    # if optim_cfg.type_name.lower() == 'adamw':
    #     return optim.AdamW(model.parameters(), **(optim_cfg.keywords))
    raise NotImplementedError(optim_cfg)


def Split_Params(codec:nn.Module):

    assert codec is not None, "Network must be instantiated first"
    parameters = set(n for n, p in codec.named_parameters() if not n.endswith(".quantiles") and p.requires_grad)
    aux_parameters = set(n for n, p in codec.named_parameters() if n.endswith(".quantiles") and p.requires_grad)

    # Make sure there is no intersection of parameters
    params_dict = dict(codec.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters
    assert len(inter_params) == 0, "intersection between main params and auxiliary params"
    assert len(union_params) - len(params_dict.keys()) == 0, "intersection between main params and auxiliary params"

    normal_param = (params_dict[n] for n in sorted(list(parameters)))
    aux_parms = (params_dict[n] for n in sorted(list(aux_parameters)))

    return normal_param, aux_parms

