"""AUV constants (mjlab style)."""
from pathlib import Path
import mujoco
from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

import numpy as np

# 1. 路径与资产配置
AUV_XML: Path = SRC_PATH / "assets" / "robots" / "auv" / "xmls" / "auv.xml"
assert AUV_XML.exists(), f"未找到模型文件: {AUV_XML}"

def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, AUV_XML.parent / "assets", meshdir)
    return assets

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(AUV_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


# 2. 执行器配置 (Actuator config)
# --- 舵机配置 (位置控制) ---
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

# --- 推进器配置 (电机/推力控制) ---
AUV_ACTUATOR_THRUSTERS = BuiltinMotorActuatorCfg(
    target_names_expr=("joint_prop_.*",), 
    gear=0.1,
    effort_limit=20.0,  
)

# 3. 初始状态
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.5),
    joint_pos={
        "joint_servo_up": 0.0,
        "joint_servo_left": 0.0,
        "joint_prop_.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

# 4. 关节与执行器聚合
AUV_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        AUV_ACTUATOR_SERVO_UD,
        AUV_ACTUATOR_SERVO_LR,
        AUV_ACTUATOR_THRUSTERS, # 确保推进器已被注册
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
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity
    
    robot = Entity(get_auv_robot_cfg())
    model = robot.spec.compile()
    data = mujoco.Data(model) 
    
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep)