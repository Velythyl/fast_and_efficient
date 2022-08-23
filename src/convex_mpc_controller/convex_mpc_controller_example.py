"""Example of MPC controller on A1 robot."""
import numpy as np
from absl import app
from absl import flags

import time

from src.convex_mpc_controller import locomotion_controller
from src.convex_mpc_controller.locomotion_controller import ControllerMode
from src.convex_mpc_controller.locomotion_controller import GaitType
from src.worlds import plane_world, slope_world, stair_world, uneven_world

flags.DEFINE_string("logdir", "logs", "where to log trajectories.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 1., "maximum time to run the robot.")
flags.DEFINE_enum("world", "plane",
                  ["plane", "slope", "stair", "uneven"],
                  "world type to choose from.")
FLAGS = flags.FLAGS

WORLD_NAME_TO_CLASS_MAP = dict(plane=plane_world.PlaneWorld,
                               slope=slope_world.SlopeWorld,
                               stair=stair_world.StairWorld,
                               uneven=uneven_world.UnevenWorld)
class Record:
    def __init__(self, name):
        self.name = name
        self.record_angles = []
        self.record_timesteps = []
    def record(self, controller, timestep):
        angles = controller._robot.motor_angles()
        record_angles.append(angles)
        record_timesteps.append(timestep)

    def save_record(self):
        global record_angles
        global record_timesteps
        np_record_angles = np.array(record_angles)
        np_record_timesteps = np.array(record_timesteps)
        np.save(f"{self.name}_angles.npy", np_record_angles)
        np.save(f"{self.name}_timesteps.npy", np_record_timesteps)

def _update_controller(controller):
  # Update speed
  lin_speed, rot_speed = [0.3, 0.], 0.
  controller.set_desired_speed(lin_speed, rot_speed)
  # Update controller moce
  controller.set_controller_mode(ControllerMode.WALK)
  # Update gait
  controller.set_gait(GaitType.FLYTROT)


def main(argv):
  del argv  # unused
  controller = locomotion_controller.LocomotionController(
      FLAGS.use_real_robot,
      FLAGS.show_gui,
      world_class=WORLD_NAME_TO_CLASS_MAP[FLAGS.world])

  try:
    start_time = controller.time_since_reset
    current_time = start_time

    record = Record("full")

    while current_time - start_time < FLAGS.max_time_secs:
      record.record(controller, current_time)
      current_time = controller.time_since_reset
      time.sleep(0.05)
      _update_controller(controller)
      if not controller.is_safe:
        break

    record.save_record()
  finally:
    controller.set_controller_mode(
        locomotion_controller.ControllerMode.TERMINATE)




if __name__ == "__main__":
  app.run(main)
