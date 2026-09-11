"""PPO agent config for the dog3 locomotion tasks.

Same hyperparameters / config format as the InstinctLab D1 tasks (Instinct-RL PPO).
"""

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class Dog3PolicyCfg(InstinctRlActorCriticCfg):
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


@configclass
class Dog3AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class Dog3NormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class Dog3FlatPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: Dog3PolicyCfg = Dog3PolicyCfg()
    algorithm: Dog3AlgorithmCfg = Dog3AlgorithmCfg()
    normalizers: Dog3NormalizersCfg = Dog3NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    log_interval = 10
    experiment_name = "dog3_locomotion_flat"

    load_run = None

    def __post_init__(self):
        super().__post_init__()  # type: ignore
        self.resume = self.load_run is not None
        self.run_name = ""


@configclass
class Dog3RoughPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: Dog3PolicyCfg = Dog3PolicyCfg()
    algorithm: Dog3AlgorithmCfg = Dog3AlgorithmCfg()
    normalizers: Dog3NormalizersCfg = Dog3NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    log_interval = 10
    experiment_name = "dog3_locomotion_rough"

    load_run = None

    def __post_init__(self):
        super().__post_init__()  # type: ignore
        self.resume = self.load_run is not None
        self.run_name = ""
