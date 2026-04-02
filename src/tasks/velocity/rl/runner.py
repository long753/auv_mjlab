import os
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.envs.mdp.actions import JointPositionAction


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    if self.logger.logger_type == "wandb" and wandb.run:
        run_name = wandb.run.name
    else:
        run_name = "local"
    onnx_path = os.path.join(policy_path, filename)
    
    # 自定义metadata获取，避免JointPositionAction假设
    env = self.env.unwrapped
    robot = env.scene["robot"]
    metadata = {
        "run_name": run_name,
        "policy_type": "auv_thruster_allocation",
        "robot_name": "auv",
        "action_dim": 6,
        "observation_dim": 26,
        "num_actuators": len(robot.actuator_names),
        "actuator_names": list(robot.actuator_names),
    }
    
    try:
        attach_metadata_to_onnx(onnx_path, metadata)
    except Exception as e:
        print(f"Warning: Failed to attach metadata: {e}")
    
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
 