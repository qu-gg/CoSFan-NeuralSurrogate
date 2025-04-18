"""
@file utils.py

Utility functions across files
"""
from omegaconf import DictConfig, OmegaConf


def flatten_cfg(cfg: DictConfig):
    """ Utility function to flatten the primary submodules of a Hydra config """
    # Disable struct flag on the config
    OmegaConf.set_struct(cfg, False)

    # Loop through each item, merging with the main cfg if its another DictConfig
    for key, value in cfg.copy().items():
        if isinstance(value, DictConfig):            
            cfg.merge_with(cfg.pop(key))

    # Do it a second time for nested cfgs
    for key, value in cfg.copy().items():
            if isinstance(value, DictConfig):            
                cfg.merge_with(cfg.pop(key))

    print(cfg)
    return cfg


def get_model(name):
    """ Import and return the specific latent dynamics function by the given name"""
    # Lowercase name in case of misspellings
    name = name.lower()

    # Feed-Forward Models
    if name == "feedforwardmask":
        from models.FeedForward import FeedForward
        return FeedForward
    
    if name == "feedforwardmask-stationary":
        from models.FeedForwardStationary import FeedForward
        return FeedForward

    # MAML
    if name == "maml":
        from models.MAML import Maml
        return Maml

    # PNS non-meta 
    if name == "pns":
        from models.PNS import PNS
        return PNS

    # Given no correct model type, raise error
    raise NotImplementedError("Model type {} not implemented.".format(name))


def get_memory(name):
    """ Import and return the specific memory module by the given name"""
    if name == "er":
        from memory.ExactReplay import ExactReplay
        return ExactReplay

    if name == "task_agnostic":
        from memory.TaskAgnosticReservoir import TaskAgnosticReservoir
        return TaskAgnosticReservoir
    
    if name == "task_aware":
        from memory.TaskAwareReservoir import TaskAwareReservoir
        return TaskAwareReservoir

    if name == "task_relational":
        from memory.TaskRelationalReservoir import TaskRelationalReservoir
        return TaskRelationalReservoir
