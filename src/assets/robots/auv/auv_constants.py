"""AUV constants (mjlab style, one-to-one with Unitree A2)."""
from pathlib import Path
import mujoco
from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

import numpy as np


# MJCF and assets.
##
AUV_XML: Path = SRC_PATH / "assets" / "robots" / "auv" / "xmls" / "auv.xml"
assert AUV_XML.exists()
def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, AUV_XML.parent / "assets", meshdir)
    return assets
def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(AUV_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec
##
# Actuator config.
# 2个舵机输入：
# - UD: 直接驱动 up（down 由 equality 约束成 -up）
# - LR: 直接驱动 left（right 由 equality 约束成 left）
##
AUV_ACTUATOR_SERVO_UD = BuiltinPositionActuatorCfg(
    target_names_expr=("joint_servo_up",),
    stiffness=20.0,
    damping=4.0,
    effort_limit=10.0,
    armature=0.03,
)
AUV_ACTUATOR_SERVO_LR = BuiltinPositionActuatorCfg(
    target_names_expr=("joint_servo_left",),
    stiffness=20.0,
    damping=4.0,
    effort_limit=10.0,
    armature=0.03,
)
##
# Keyframes.
##
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.5),
    joint_pos={
        "joint_servo_up": 0.0,
        "joint_servo_down": 0.0,
        "joint_servo_left": 0.0,
        "joint_servo_right": 0.0,
    },
    joint_vel={".*": 0.0},
)
##
# Final config.
##
AUV_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        AUV_ACTUATOR_SERVO_UD,
        AUV_ACTUATOR_SERVO_LR,
    ),
    soft_joint_pos_limit_factor=1.0,
)
def get_auv_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(),
        spec_fn=get_spec,
        articulation=AUV_ARTICULATION,
    )
    

if __name__ == "__main__":
    import time
    import numpy as np
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity
    robot = Entity(get_auv_robot_cfg())
    model = robot.spec.compile()
    data = mujoco.MjData(model)
    
    # 映射action到舵机角度范围
    for i in range(model.nu):
        model.actuator_ctrllimited[i] = 1
        model.actuator_ctrlrange[i, 0] = -np.pi / 2
        model.actuator_ctrlrange[i, 1] = np.pi / 2
        
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep)