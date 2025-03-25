# adapted from
# https://developmentalsystems.org/sensorimotor-lenia
from addict import Dict
import cv2
import torch

from ..auxlib._log_context_duration import log_context_duration
from ._LeniaInitializationSpace import LeniaInitializationSpace
from ._Lenia_C import Lenia_C


def generate_observations(
    SX,
    SY,
    *,
    borders,
    crea_file,
    device,
    mode,
    num_frames=1000,
    logger=print,
    list_kernels=range(10),
    zoom=1,
):
    lenia_config = Lenia_C.default_config()
    lenia_config.SX = SX
    lenia_config.SY = SY
    lenia_config.final_step = num_frames
    lenia_config.version = "pytorch_fft"
    lenia_config.nb_kernels = len(list_kernels)
    initialization_space_config = Dict()
    initialization_space = LeniaInitializationSpace(
        config=initialization_space_config
    )
    system = Lenia_C(
        initialization_space=initialization_space,
        config=lenia_config,
        device=device,
    )
    a = torch.load(crea_file, map_location=torch.device(device))

    system.init_loca = []
    for x, y in [
        (1, 1),
        (200, 200),
        (300, 100),
        (200, 100),
        (300, 250),
        (300, 200),
        (100, 350),
    ]:
        for i in range(x, x + 40):
            for j in range(y, y + 40):
                system.init_loca.append((i, j))

    # b=torch.load("run_0000179_data.pickle")
    policy_parameters = Dict.fromkeys(["initialization", "update_rule"])
    policy_parameters["initialization"] = a["policy_parameters"][
        "initialization"
    ]
    policy_parameters["update_rule"] = a["policy_parameters"]["update_rule"]

    policy_parameters["update_rule"]["R"] = (
        policy_parameters["update_rule"]["R"] + 15
    ) * zoom - 15
    init_s = policy_parameters["initialization"].init.cpu().numpy() * 1.0

    width = int(init_s.shape[1] * zoom)
    height = int(init_s.shape[0] * zoom)
    dim = (width, height)
    # resize image
    resized = cv2.resize(init_s, dim)
    init_f = torch.tensor(resized).to(device)

    for k in policy_parameters["update_rule"].keys():
        if k not in ("R", "T"):
            policy_parameters["update_rule"][k] = policy_parameters[
                "update_rule"
            ][k][list_kernels]
        policy_parameters["update_rule"][k] = policy_parameters["update_rule"][
            k
        ].to(device)

    system.reset(
        initialization_parameters=policy_parameters["initialization"],
        update_rule_parameters=policy_parameters["update_rule"],
    )

    if mode == "random":
        nb_obstacle = 30
        system.random_obstacle(nb_obstacle)

    if borders:
        system.init_wall[:, :20] = 1
        system.init_wall[:, -20:] = 1
        system.init_wall[-20:, :] = 1
        system.init_wall[:20, :] = 1

    with log_context_duration("Running Lenia", logger=logger):
        with torch.no_grad():
            system.generate_init_state()
            system.state[0, :, :, 0] = 0
            for loca in system.init_loca:
                i, j = loca
                if True:
                    system.state[
                        0,
                        loca[0] : loca[0] + init_f.shape[0],
                        loca[1] : loca[1] + init_f.shape[1],
                        0,
                    ] = init_f
            return system.run()
