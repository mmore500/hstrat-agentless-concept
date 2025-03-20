from copy import deepcopy
import warnings

from addict import Dict
import torch

from ._Dummy_init_mod import Dummy_init_mod
from ._LeniaInitializationSpace import LeniaInitializationSpace
from ._LeniaStepFFTC import LeniaStepFFTC
from ._LeniaUpdateRuleSpace import LeniaUpdateRuleSpace


class Lenia_C(torch.nn.Module):
    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.version = (
            "pytorch_fft"  # "pytorch_fft", "pytorch_conv2d"
        )
        default_config.SX = 256
        default_config.SY = 256
        default_config.final_step = 40
        default_config.C = 2
        default_config.speed_x = 0
        default_config.speed_y = 0
        return default_config

    def __init__(
        self,
        initialization_space=None,
        update_rule_space=None,
        nb_k=10,
        init_size=40,
        config={},
        device=torch.device("cpu"),
        **kwargs,
    ):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        torch.nn.Module.__init__(self)
        self.device = device
        self.init_size = init_size
        if initialization_space is not None:
            self.initialization_space = initialization_space
        else:
            self.initialization_space = LeniaInitializationSpace(
                self.init_size
            )

        if update_rule_space is not None:
            self.update_rule_space = update_rule_space
        else:
            self.update_rule_space = LeniaUpdateRuleSpace(nb_k)

        self.run_idx = 0
        self.init_wall = torch.zeros((self.config.SX, self.config.SY))
        # reset with no argument to sample random parameters
        self.reset()
        self.to(self.device)

    def reset(
        self, initialization_parameters=None, update_rule_parameters=None
    ):
        # call the property setters
        if initialization_parameters is not None:
            self.initialization_parameters = initialization_parameters
        else:
            self.initialization_parameters = self.initialization_space.sample()

        if update_rule_parameters is not None:
            self.update_rule_parameters = update_rule_parameters
        else:
            policy_parameters = Dict.fromkeys(["update_rule"])
            policy_parameters["update_rule"] = self.update_rule_space.sample()
            # divide h by 3 at the beginning as some unbalanced kernels can easily kill
            policy_parameters["update_rule"].h = (
                policy_parameters["update_rule"].h / 3
            )
            self.update_rule_parameters = policy_parameters["update_rule"]

        # initialize Lenia CA with update rule parameters
        if self.config.version == "pytorch_fft":
            lenia_step = LeniaStepFFTC(
                self.config.C,
                self.update_rule_parameters["R"],
                self.update_rule_parameters["T"],
                self.update_rule_parameters["c0"],
                self.update_rule_parameters["c1"],
                self.update_rule_parameters["r"],
                self.update_rule_parameters["rk"],
                self.update_rule_parameters["b"],
                self.update_rule_parameters["w"],
                self.update_rule_parameters["h"],
                self.update_rule_parameters["m"],
                self.update_rule_parameters["s"],
                1,
                is_soft_clip=False,
                SX=self.config.SX,
                SY=self.config.SY,
                speed_x=self.config.speed_x,
                speed_y=self.config.speed_y,
                device=self.device,
            )
        self.add_module("lenia_step", lenia_step)

        # initialize Lenia initial state with initialization_parameters
        init = self.initialization_parameters["init"]
        # initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, self.initialization_space.config.neat_config, device=self.device)
        self.add_module("initialization", Dummy_init_mod(init))

        # push the nn.Module and the available device
        self.to(self.device)
        self.generate_init_state()

    def random_obstacle(self, nb_obstacle=6):
        self.init_wall = torch.zeros((self.config.SX, self.config.SY))

        x = torch.arange(self.config.SX)
        y = torch.arange(self.config.SY)
        xx = x.view(-1, 1).repeat(1, self.config.SY)
        yy = y.repeat(self.config.SX, 1)
        for i in range(nb_obstacle):
            X = (xx - int(torch.rand(1) * self.config.SX)).float()
            Y = (yy - int(torch.rand(1) * self.config.SY / 2)).float()
            D = torch.sqrt(X**2 + Y**2) / 10
            mask = (D < 1).float()
            self.init_wall = torch.clamp(self.init_wall + mask, 0, 1)

    def random_obstacle_bis(self, nb_obstacle=6):
        self.init_wall = torch.zeros((self.config.SX, self.config.SY))

        x = torch.arange(self.config.SX)
        y = torch.arange(self.config.SY)
        xx = x.view(-1, 1).repeat(1, self.config.SY)
        yy = y.repeat(self.config.SX, 1)
        for i in range(nb_obstacle):
            X = (xx - int(torch.rand(1) * self.config.SX)).float()
            Y = (yy - int(torch.rand(1) * self.config.SY)).float()
            D = torch.sqrt(X**2 + Y**2) / 10
            mask = (D < 1).float()
            self.init_wall = torch.clamp(self.init_wall + mask, 0, 1)
        self.init_wall[95:155, 170:230] = 0

    def generate_init_state(self, X=105, Y=180):
        init_state = torch.zeros(
            1,
            self.config.SX,
            self.config.SY,
            self.config.C,
            dtype=torch.float64,
        )
        init_state[
            0, X : X + self.init_size, Y : Y + self.init_size, 0
        ] = self.initialization.init
        if self.config.C > 1:
            init_state[0, :, :, 1] = self.init_wall
        self.state = init_state.to(self.device)
        self.step_idx = 0

    def update_initialization_parameters(self):
        new_initialization_parameters = deepcopy(
            self.initialization_parameters
        )
        new_initialization_parameters["init"] = self.initialization.init.data
        if not self.initialization_space.contains(
            new_initialization_parameters
        ):
            new_initialization_parameters = self.initialization_space.clamp(
                new_initialization_parameters
            )
            warnings.warn(
                "provided parameters are not in the space range and are therefore clamped"
            )
        self.initialization_parameters = new_initialization_parameters

    def update_update_rule_parameters(self):
        new_update_rule_parameters = deepcopy(self.update_rule_parameters)
        # gather the parameter from the lenia step (which may have been optimized)
        new_update_rule_parameters["m"] = self.lenia_step.m.data
        new_update_rule_parameters["s"] = self.lenia_step.s.data
        new_update_rule_parameters["r"] = self.lenia_step.r.data
        new_update_rule_parameters["rk"] = self.lenia_step.rk.data
        new_update_rule_parameters["b"] = self.lenia_step.b.data
        new_update_rule_parameters["w"] = self.lenia_step.w.data
        new_update_rule_parameters["h"] = self.lenia_step.h.data
        if not self.update_rule_space.contains(new_update_rule_parameters):
            new_update_rule_parameters = self.update_rule_space.clamp(
                new_update_rule_parameters
            )
            warnings.warn(
                "provided parameters are not in the space range and are therefore clamped"
            )
        self.update_rule_parameters = new_update_rule_parameters

    def step(self, intervention_parameters=None):
        self.state = self.lenia_step(self.state)
        self.step_idx += 1
        return self.state

    def forward(self):
        state = self.step(None)
        return state

    def run(self):
        """run lenia for the number of step specified in the config.
        Returns the observations containing the state at each timestep"""
        # clip parameters just in case
        if not self.initialization_space["init"].contains(
            self.initialization.init.data
        ):
            self.initialization.init.data = self.initialization_space[
                "init"
            ].clamp(self.initialization.init.data)
        if not self.update_rule_space["r"].contains(self.lenia_step.r.data):
            self.lenia_step.r.data = self.update_rule_space["r"].clamp(
                self.lenia_step.r.data
            )
        if not self.update_rule_space["rk"].contains(self.lenia_step.rk.data):
            self.lenia_step.rk.data = self.update_rule_space["rk"].clamp(
                self.lenia_step.rk.data
            )
        if not self.update_rule_space["b"].contains(self.lenia_step.b.data):
            self.lenia_step.b.data = self.update_rule_space["b"].clamp(
                self.lenia_step.b.data
            )
        if not self.update_rule_space["w"].contains(self.lenia_step.w.data):
            self.lenia_step.w.data = self.update_rule_space["w"].clamp(
                self.lenia_step.w.data
            )
        if not self.update_rule_space["h"].contains(self.lenia_step.h.data):
            self.lenia_step.h.data = self.update_rule_space["h"].clamp(
                self.lenia_step.h.data
            )
        if not self.update_rule_space["m"].contains(self.lenia_step.m.data):
            self.lenia_step.m.data = self.update_rule_space["m"].clamp(
                self.lenia_step.m.data
            )
        if not self.update_rule_space["s"].contains(self.lenia_step.s.data):
            self.lenia_step.s.data = self.update_rule_space["s"].clamp(
                self.lenia_step.s.data
            )
        # self.generate_init_state()
        observations = Dict()
        observations.timepoints = list(range(self.config.final_step))
        observations.states = torch.empty(
            (
                self.config.final_step,
                self.config.SX,
                self.config.SY,
                self.config.C,
            )
        )
        observations.states[0] = self.state
        for step_idx in range(1, self.config.final_step):
            cur_observation = self.step(None)
            observations.states[step_idx] = cur_observation[0, :, :, :]

        return observations

    def save(self, filepath):
        """
        Saves the system object using torch.save function in pickle format
        Can be used if the system state's change over exploration and we want to dump it
        """
        torch.save(self, filepath)

    def close(self):
        pass
