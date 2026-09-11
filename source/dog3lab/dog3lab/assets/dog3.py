"""dog3 quadruped (12-DOF legged) asset for Isaac Sim.

dog3 URDF (SolidWorks export) is stored under
``assets/resources/dog3_description``. The 4 ``*_foot_joint`` are **fixed**
in the URDF; ``merge_fixed_joints=False`` keeps the foot links so foot-tip
bonuses / contact sensors have a body to read, while DOF stays 12.

Parameters (source = the dog3 package itself, not guesses):
  * joint limits      : hip effort 20 N*m / 30 rad/s; thigh, calf 20 N*m / 17.8 rad/s
                        (urdf/dog3.urdf.xacro)
  * default joint pose: hip 0.0, thigh 0.0, calf 0.0 -- the URDF joint origins were
                        shifted by the original default angles (thigh +0.7, calf -1.44)
                        so that joint zero IS the standing stance (matches the sim2sim
                        URDF; see dog3_sim2sim).  Rear foot links (LH/RH_FOOT) moved
                        back 0.035 m to move the rear contact point back.
  * nominal COM height: 0.30 m (config/reference.info comHeight)
  * base_link mass    : 5.6022 kg (dog3.urdf; csv's 3.12 ignored per user)

STIFFNESS/DAMPING and ARMATURE are NOT in the dog3 package: stiffness=20,
damping=0.5 are the common small-quadruped (Go1-class) values and armature=0.05
is a placeholder -- calibrate these against the real robot.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

DOG3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=os.path.join(__file_dir__, "resources/dog3_description/urdf/dog3.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        # joint zero == standing stance: the URDF joint origins are shifted by the
        # default angles (thigh +0.7, calf -1.44), so the default pose is all zeros.
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.0,
            ".*_calf_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=20.0,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.1,
            armature=0.05,
            min_delay=0,
            max_delay=4,
        ),
        "thigh_calf": DelayedPDActuatorCfg(
            joint_names_expr=[".*_(thigh|calf)_joint"],
            effort_limit=20.0,
            velocity_limit=17.8,
            stiffness=20.0,
            damping=0.5,
            friction=0.1,
            armature=0.05,
            min_delay=0,
            max_delay=4,
        ),
    },
)
