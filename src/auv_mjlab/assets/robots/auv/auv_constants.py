"""AUV constants (mjlab style)."""
import sys
from pathlib import Path
import numpy as np
import mujoco
current_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parents[5]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.auv_mjlab import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.actuator.actuator import TransmissionType

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
AUV_ACTUATOR_SERVOS = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "joint_servo_up",
        "joint_servo_left",
    ),
    stiffness=20.0,
    damping=4.0,
    effort_limit=10.0,    
    armature=0.03,
)

# --- 推进器配置 (电机/推力控制) ---
AUV_ACTUATOR_THRUSTERS = BuiltinMotorActuatorCfg(
    target_names_expr=(
        "thrust_up_site",
        "thrust_down_site",
        "thrust_left_site",
        "thrust_right_site",
    ),
    transmission_type=TransmissionType.SITE,  
    effort_limit=50.0,
    gear=1.0,   
)

# 3. 初始状态
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.5),
    joint_pos={
        "joint_servo_up": 0.0,
        "joint_servo_left": 0.0,
        # 推进器叶片关节被 range="0 0" 锁定，不需要初始化位姿
    },
    joint_vel={".*": 0.0},
)

# Hydrodynamic parameters
AUV_VOLUME = 0.106  # m³
AUV_MASS = 100.0  # kg
AUV_INERTIA = np.array([19.3, 19.3, 1.125])  # [I_xx, I_yy, I_zz]
AUV_COM_TO_COB_OFFSET = np.array([0.0, 0.0, 0.05])  # m, 质心到浮心的偏移

# 4. 关节与执行器聚合
AUV_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        AUV_ACTUATOR_SERVOS,
        AUV_ACTUATOR_THRUSTERS,
    ),
    soft_joint_pos_limit_factor=1.0,
)

def get_auv_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(), # AUV 外壳碰撞未精细定义前先置空
        spec_fn=get_spec,
        articulation=AUV_ARTICULATION,
    )

if __name__ == "__main__":
    import time
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity
    import torch
    from src.auv_mjlab.tasks.velocity.mdp.hydrodynamics import HydrodynamicForceModels
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
    
    # 创建机器人实体
    robot = Entity(get_auv_robot_cfg())
    model = robot.spec.compile()
    data = mujoco.MjData(model)
    
    # 水动力参数
    water_density = 1000.0  # kg/m³
    water_viscosity = 0.0009  # Pa·s
    g_mag = 9.81  # 重力加速度
    
    device = torch.device('cpu')
    force_model = HydrodynamicForceModels(num_envs=1, device=device, debug=True)
    
    volume_tensor = torch.tensor([[AUV_VOLUME]], dtype=torch.float32, device=device)
    mass_tensor = torch.tensor([[AUV_MASS]], dtype=torch.float32, device=device)
    inertia_tensor = torch.tensor([AUV_INERTIA], dtype=torch.float32, device=device)
    com_to_cob_offset_tensor = torch.tensor([AUV_COM_TO_COB_OFFSET], dtype=torch.float32, device=device)
    
    root_body_id = 0
    
    with viewer.launch_passive(model, data) as v:
        step_count = 0
        while v.is_running():
            # MuJoCo 四元数格式为 (w, x, y, z)，与 mjlab 一致，无需重排
            mujoco_quat = data.xquat[root_body_id].copy()  
            quat_w = torch.tensor([mujoco_quat], dtype=torch.float32, device=device)
            
            linvel_w = torch.tensor([data.cvel[root_body_id][3:6]], dtype=torch.float32, device=device)
            angvel_w = torch.tensor([data.cvel[root_body_id][0:3]], dtype=torch.float32, device=device)
            
            buoyancy_force, buoyancy_torque = force_model.calculate_buoyancy_forces(
                quat_w, water_density, volume_tensor, g_mag, com_to_cob_offset_tensor
            )
            
            f_d, g_d, f_v, g_v = force_model.calculate_density_and_viscosity_forces(
                quat_w, linvel_w, angvel_w,
                inertia_tensor, water_viscosity, water_density, mass_tensor
            )
            
            total_force_b = buoyancy_force + f_d + f_v
            total_torque_b = buoyancy_torque + g_d + g_v
            
            # 将力和扭矩从刚体坐标系转换到世界坐标系 (body → world)
            # xfrc_applied 需要世界坐标系的力和扭矩
            total_force_w = quat_apply(quat_w, total_force_b)
            total_torque_w = quat_apply(quat_w, total_torque_b)
            # 注意：不再手动添加重力，MuJoCo 已自动施加重力
            
            total_force_np = total_force_w[0].cpu().numpy()
            total_torque_np = total_torque_w[0].cpu().numpy()
            
            data.xfrc_applied[root_body_id][:3] = total_force_np
            data.xfrc_applied[root_body_id][3:6] = total_torque_np
            
            if step_count % 100 == 0:
                print(f"Step {step_count}:")
                print(f"  姿态四元数: {mujoco_quat}")
                print(f"  线速度: {linvel_w[0].cpu().numpy()}")
                print(f"  角速度: {angvel_w[0].cpu().numpy()}")
                print(f"  总力 (世界系): {total_force_np}")
                print(f"  总扭矩 (世界系): {total_torque_np}")
                print(f"  位置 z: {data.xpos[root_body_id][2]:.3f}\n")
            
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep)
            step_count += 1