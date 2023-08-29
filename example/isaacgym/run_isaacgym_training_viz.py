# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

# This code implemented based on https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/train.py

import os
import sys
from datetime import datetime

# noinspection PyUnresolvedReferences
import isaacgym
from sim_web_visualizer.isaac_visualizer_client import create_isaac_visualizer, bind_visualizer_to_gym, set_gpu_pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), "IsaacGymEnvs"))

import hydra

from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

################################################################################################
# The following code is only used for visualizer and not presented in the original IsaacGymEnv
################################################################################################
from isaacgymenvs.tasks.base import vec_task
from isaacgym import gymapi


def wrapped_create_sim(
    self: vec_task.VecTask, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams
):
    sim = vec_task._create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()
    self.gym = bind_visualizer_to_gym(self.gym, sim)
    set_gpu_pipeline(sim_params.use_gpu_pipeline)
    return sim


# Reload VecTask function to create a hook for sim_web_visualizer
vec_task.VecTask.create_sim = wrapped_create_sim

# Create web visualizer
create_isaac_visualizer(port=6000, host="localhost", keep_default_viewer=False, max_env=2)


################################################################################################
# End of visualizer code
################################################################################################


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict["params"]["config"]

    train_cfg["device"] = cfg.rl_device

    train_cfg["population_based_training"] = cfg.pbt.enabled
    train_cfg["pbt_idx"] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg["full_experiment_name"] = cfg.get("full_experiment_name")

    print(f"Using rl_device: {cfg.rl_device}")
    print(f"Using sim_device: {cfg.sim_device}")
    print(train_cfg)

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"]["model_size_multiplier"]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}'
            )
    except KeyError:
        pass

    return config_dict


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_isaacgym_env(**kwargs),
        },
    )

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, "dict_obs_cls") and ige_env_cls.dict_obs_cls else False

    if dict_cls:
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec["obs"] = {
            "names": list(actor_net_cfg.inputs.keys()),
            "concat": not actor_net_cfg.name == "complex_net",
            "space_name": "observation_space",
        }
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec["states"] = {
                "names": list(critic_net_cfg.inputs.keys()),
                "concat": not critic_net_cfg.name == "complex_net",
                "space_name": "state_space",
            }

        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs),
        )
    else:
        vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder("amp_continuous", lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder(
            "amp_continuous", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs)
        )
        model_builder.register_model("continuous_amp", lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
        model_builder.register_network("amp", lambda **kwargs: amp_network_builder.AMPBuilder())

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join(
            "runs", cfg.train.params.config.name + "_{date:%d-%H-%M-%S}".format(date=datetime.now())
        )

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run(
        {
            "train": not cfg.test,
            "play": cfg.test,
            "checkpoint": cfg.checkpoint,
            "sigma": cfg.sigma if cfg.sigma != "" else None,
        }
    )


if __name__ == "__main__":
    launch_rlg_hydra()
