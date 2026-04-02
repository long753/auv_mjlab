"""AUV constants (mjlab style)."""
import sys
from pathlib import Path

# 添加项目根目录到sys.path以便直接运行此脚本
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent  # src/assets/robots/auv -> 项目根目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mujoco
from src.auv_mjlab import SRC_PATH
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

# Hydrodynamic parameters
# AUV近似为圆柱体：直径0.3m，长度1.5m
AUV_VOLUME = 0.106  # m³ (π * R² * L = 3.1416 * 0.15² * 1.5)
AUV_MASS = 100.0  # kg (合理质量)
# 惯性张量对角线元素 (kg·m²)：基于圆柱体计算
# I_xx = I_yy = (1/12) * M * (3R² + L²) = (1/12) * 100 * (0.0675 + 2.25) ≈ 19.3
# I_zz = (1/2) * M * R² = 0.5 * 100 * 0.0225 = 1.125
AUV_INERTIA = np.array([19.3, 19.3, 1.125])  # [I_xx, I_yy, I_zz]
AUV_COM_TO_COB_OFFSET = np.array([0.0, 0.0, 0.05])  # m, 质心到浮心的偏移（假设浮心稍高于质心）

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
    import torch
    import numpy as np
    from src.tasks.velocity.mdp.hydrodynamics import HydrodynamicForceModels
    from mjlab.utils.lab_api.math import quat_apply
    
    # 创建机器人实体
    robot = Entity(get_auv_robot_cfg())
    model = robot.spec.compile()
    data = mujoco.Data(model)
    
    # 水动力参数
    water_density = 1000.0  # kg/m³
    water_viscosity = 0.0009  # Pa·s (MuJoCo默认值)
    g_mag = 9.81  # 重力加速度
    
    # 创建水动力模型
    device = torch.device('cpu')
    force_model = HydrodynamicForceModels(num_envs=1, device=device, debug=True)
    
    # 准备常量张量
    volume_tensor = torch.tensor([[AUV_VOLUME]], dtype=torch.float32, device=device)
    mass_tensor = torch.tensor([[AUV_MASS]], dtype=torch.float32, device=device)
    inertia_tensor = torch.tensor([AUV_INERTIA], dtype=torch.float32, device=device)
    com_to_cob_offset_tensor = torch.tensor([AUV_COM_TO_COB_OFFSET], dtype=torch.float32, device=device)
    
    # 根身体索引（假设为0）
    root_body_id = 0
    
    with viewer.launch_passive(model, data) as v:
        step_count = 0
        while v.is_running():
            # 获取根身体状态
            # 位置 (未使用)
            # pos = data.xpos[root_body_id].copy()
            
            # 四元数 (w, x, y, z) - MuJoCo使用(x, y, z, w)格式，需要转换
            mujoco_quat = data.xquat[root_body_id].copy()  # [x, y, z, w]
            quat_w = torch.tensor([[mujoco_quat[3], mujoco_quat[0], mujoco_quat[1], mujoco_quat[2]]], 
                                  dtype=torch.float32, device=device)
            
            # 线速度和角速度 (世界坐标系)
            linvel_w = torch.tensor([data.cvel[root_body_id][3:6]], dtype=torch.float32, device=device)  # 线速度部分
            angvel_w = torch.tensor([data.cvel[root_body_id][0:3]], dtype=torch.float32, device=device)  # 角速度部分
            
            # 计算浮力
            buoyancy_force, buoyancy_torque = force_model.calculate_buoyancy_forces(
                quat_w, water_density, volume_tensor, g_mag, com_to_cob_offset_tensor
            )
            
            # 计算阻力和粘性力
            f_d, g_d, f_v, g_v = force_model.calculate_density_and_viscosity_forces(
                quat_w, linvel_w, angvel_w,
                inertia_tensor, water_viscosity, water_density, mass_tensor
            )
            
            # 总力和扭矩 (刚体坐标系)
            total_force_b = buoyancy_force + f_d + f_v
            total_torque_b = buoyancy_torque + g_d + g_v
            
            # 添加重力 (世界坐标系向下，需要转换到刚体坐标系)
            gravity_force_w = torch.tensor([[0.0, 0.0, -AUV_MASS * g_mag]], dtype=torch.float32, device=device)
            gravity_force_b = quat_apply(quat_w, gravity_force_w)  # 转换到刚体系
            total_force_b += gravity_force_b
            
            # 转换为numpy数组
            total_force_np = total_force_b[0].cpu().numpy()
            total_torque_np = total_torque_b[0].cpu().numpy()
            
            # 施加外力到MuJoCo (身体局部坐标系)
            data.xfrc_applied[root_body_id][:3] = total_force_np
            data.xfrc_applied[root_body_id][3:6] = total_torque_np
            
            # 每100步打印调试信息
            if step_count % 100 == 0:
                print(f"Step {step_count}:")
                print(f"  姿态四元数: {mujoco_quat}")
                print(f"  线速度: {linvel_w[0].cpu().numpy()}")
                print(f"  角速度: {angvel_w[0].cpu().numpy()}")
                print(f"  总力 (刚体系): {total_force_np}")
                print(f"  总扭矩 (刚体系): {total_torque_np}")
                print(f"  位置 z: {data.xpos[root_body_id][2]:.3f}")
                print()
            
            # 仿真步进
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep)
            step_count += 1