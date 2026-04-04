"""AUV CONSTANTS"""

import sys
import math
from pathlib import Path
import mujoco

AUV_MAX_SERVO_ANGLE = math.pi / 2

current_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parents[5]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.auv_mjlab import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.actuator.actuator import TransmissionType


# ==========================
# 1. 导入模型文件
# ==========================

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


# ==========================
# 2. 执行器配置
# ==========================

# --- 舵机配置 (位置控制) ---
AUV_ACTUATOR_SERVOS = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "joint_servo_up",    # 上下舵机转角
        "joint_servo_left",  # 左右舵机转角
    ),
    stiffness=20.0,
    damping=4.0,
    effort_limit=10.0,
    armature=0.03,
)

# --- 推进器配置 (推力控制) ---
AUV_ACTUATOR_THRUSTERS = BuiltinMotorActuatorCfg(
    target_names_expr=(
        "thrust_up_site",
        "thrust_down_site",
        "thrust_left_site",
        "thrust_right_site",
    ),
    transmission_type=TransmissionType.SITE,
    effort_limit=40.0,
    gear=1.0,
)


# ==========================
# 3. 初始状态
# ==========================

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.0),
    joint_pos={
        "joint_servo_up": 0.0,
        "joint_servo_left": 0.0,
    },
    joint_vel={".*": 0.0},
)


# ==========================
# 4. 实体组装
# ==========================

AUV_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        AUV_ACTUATOR_SERVOS,
        AUV_ACTUATOR_THRUSTERS,
    ),
    soft_joint_pos_limit_factor=1.0,
)


def get_auv_robot_cfg() -> EntityCfg:
    """返回供 RL 环境调用的 AUV 实体配置。"""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(),  # AUV 外壳碰撞未精细定义前先置空
        spec_fn=get_spec,
        articulation=AUV_ARTICULATION,
    )


# ==========================
# 5. 从模型中提取物理参数
# ==========================

def get_auv_physical_params() -> dict:
    """从MuJoCo模型中提取AUV的物理参数。

    返回:
        dict: 包含以下键值:
            - mass: float, 总质量 (kg)
            - volume: float, 排水体积 (m³)
            - inertia: tuple[float, float, float], 主转动惯量 (I_xx, I_yy, I_zz)
            - com_to_cob_offset: tuple[float, float, float], 质心到浮心偏移 (m)
    """
    spec = get_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    # base body id
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    # 总质量
    total_mass = float(model.body_subtreemass[base_body_id])

    # 转动惯量: 仅使用 base body 自身的惯性 (model.body_inertia)
    base_inertia = model.body_inertia[base_body_id]
    I_xx, I_yy, I_zz = float(base_inertia[0]), float(base_inertia[1]), float(base_inertia[2])

    # 排水体积 ,先占位，目前不考虑不重合
    water_density = model.opt.density if model.opt.density > 0 else 1000.0
    volume = total_mass / water_density

    # 质心到浮心偏移 (暂时不考虑，假设重心和浮心重合)
    com_to_cob_offset = (0.0, 0.0, 0.0)

    return {
        "mass": total_mass,
        "volume": volume,
        "inertia": (I_xx, I_yy, I_zz),
        "com_to_cob_offset": com_to_cob_offset,
    }


if __name__ == "__main__":
    import time
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    # 打印从模型提取的物理参数
    params = get_auv_physical_params()
    print("=" * 50)
    print("AUV 物理参数 (从MuJoCo模型提取)")
    print("=" * 50)
    print(f"  总质量:        {params['mass']:.4f} kg")
    print(f"  转动惯量 Ixx:  {params['inertia'][0]:.6f} kg·m²")
    print(f"  转动惯量 Iyy:  {params['inertia'][1]:.6f} kg·m²")
    print(f"  转动惯量 Izz:  {params['inertia'][2]:.6f} kg·m²")
    print("=" * 50)

    # 创建机器人实体
    robot = Entity(get_auv_robot_cfg())
    model = robot.spec.compile()
    data = mujoco.MjData(model)

    root_body_id = 0

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep)
