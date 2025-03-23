# adapted from
# https://developmentalsystems.org/sensorimotor-lenia
from addict import Dict
import torch

from ._BoxSpace import BoxSpace
from ._DictSpace import DictSpace


class LeniaInitializationSpace(DictSpace):
    """Class for initialization space that allows to sample and clip the initialization"""

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.neat_config = None
        default_config.cppn_n_passes = 2
        return default_config

    def __init__(self, init_size=40, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        spaces = Dict(
            # cppn_genome = LeniaCPPNInitSpace(self.config)
            init=BoxSpace(
                low=0.0,
                high=1.0,
                shape=(init_size, init_size),
                mutation_mean=torch.zeros((40, 40)),
                mutation_std=torch.ones((40, 40)) * 0.01,
                indpb=0.0,
                dtype=torch.float32,
            )
        )

        DictSpace.__init__(self, spaces=spaces)
