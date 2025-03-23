# adapted from
# https://developmentalsystems.org/sensorimotor-lenia
from addict import Dict
import torch

from ._BoxSpace import BoxSpace
from ._DictSpace import DictSpace
from ._DiscreteSpace import DiscreteSpace
from ._MultiDiscreteSpace import MultiDiscreteSpace


class LeniaUpdateRuleSpace(DictSpace):
    """Space associated to the parameters of the update rule"""

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, nb_k=10, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        spaces = Dict(
            R=DiscreteSpace(
                n=25, mutation_mean=0.0, mutation_std=0.01, indpb=0.01
            ),
            c0=MultiDiscreteSpace(
                nvec=[1] * nb_k,
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.1 * torch.ones((nb_k,)),
                indpb=0.1,
            ),
            c1=MultiDiscreteSpace(
                nvec=[1] * nb_k,
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.1 * torch.ones((nb_k,)),
                indpb=0.1,
            ),
            T=BoxSpace(
                low=1.0,
                high=10.0,
                shape=(),
                mutation_mean=0.0,
                mutation_std=0.1,
                indpb=0.01,
                dtype=torch.float32,
            ),
            rk=BoxSpace(
                low=0,
                high=1,
                shape=(nb_k, 3),
                mutation_mean=torch.zeros((nb_k, 3)),
                mutation_std=0.2 * torch.ones((nb_k, 3)),
                indpb=1,
                dtype=torch.float32,
            ),
            b=BoxSpace(
                low=0.0,
                high=1.0,
                shape=(nb_k, 3),
                mutation_mean=torch.zeros((nb_k, 3)),
                mutation_std=0.2 * torch.ones((nb_k, 3)),
                indpb=1,
                dtype=torch.float32,
            ),
            w=BoxSpace(
                low=0.01,
                high=0.5,
                shape=(nb_k, 3),
                mutation_mean=torch.zeros((nb_k, 3)),
                mutation_std=0.2 * torch.ones((nb_k, 3)),
                indpb=1,
                dtype=torch.float32,
            ),
            m=BoxSpace(
                low=0.05,
                high=0.5,
                shape=(nb_k,),
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.2 * torch.ones((nb_k,)),
                indpb=1,
                dtype=torch.float32,
            ),
            s=BoxSpace(
                low=0.001,
                high=0.18,
                shape=(nb_k,),
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.01 ** torch.ones((nb_k,)),
                indpb=0.1,
                dtype=torch.float32,
            ),
            h=BoxSpace(
                low=0,
                high=1.0,
                shape=(nb_k,),
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.2 * torch.ones((nb_k,)),
                indpb=0.1,
                dtype=torch.float32,
            ),
            r=BoxSpace(
                low=0.2,
                high=1.0,
                shape=(nb_k,),
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.2 * torch.ones((nb_k,)),
                indpb=1,
                dtype=torch.float32,
            ),
            # kn = DiscreteSpace(n=4, mutation_mean=0.0, mutation_std=0.1, indpb=1.0),
            # gn = DiscreteSpace(n=3, mutation_mean=0.0, mutation_std=0.1, indpb=1.0),
        )

        DictSpace.__init__(self, spaces=spaces)

    def mutate(self, x):
        mask = (x["s"] > 0.04).float() * (
            torch.rand(x["s"].shape[0]) < 0.25
        ).float().to(x["s"].device)
        param = []
        for k, space in self.spaces.items():
            if k == "R" or k == "c0" or k == "c1" or k == "T":
                param.append((k, space.mutate(x[k])))
            elif k == "rk" or k == "w" or k == "b":
                param.append((k, space.mutate(x[k], mask.unsqueeze(-1))))
            else:
                param.append((k, space.mutate(x[k], mask)))

        return Dict(param)
