"""dog3 flat-ground env config (plane, no height scan)."""

from isaaclab.utils import configclass

from .dog3_env_cfg import Dog3EnvCfg


@configclass
class Dog3FlatEnvCfg(Dog3EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------------- terrain (plane, no height scan) ----------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # critic height_scan obs term references the removed sensor: clear it
        self.observations.critic.height_scan = None
        # base_height_l2 uses the ray-caster to follow rough terrain; on flat use
        # the world-frame target height directly (function supports sensor_cfg=None)
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class Dog3FlatEnvCfg_PLAY(Dog3FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # smaller scene for interactive playback / ONNX export (single env)
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # disable noise / external pushes / domain rand events for stable playback
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
