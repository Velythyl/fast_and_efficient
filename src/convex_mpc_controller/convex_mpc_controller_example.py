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

#  inline CVector3 BulletToSimulator(const btVector3& c_bt_vector) {
#     return CVector3(c_bt_vector.getX(), -c_bt_vector.getZ(), c_bt_vector.getY());
#  }

#  inline btVector3 SimulatorToBullet(const CVector3& c_a_vector) {
#     return btVector3(c_a_vector.GetX(), c_a_vector.GetZ(), -c_a_vector.GetY());
#  }

#  inline CQuaternion BulletToSimulator(const btQuaternion& c_bt_quaternion) {
#     return CQuaternion(c_bt_quaternion.getW(), c_bt_quaternion.getX(),
#                           -c_bt_quaternion.getZ(), c_bt_quaternion.getY());
#  }

#  inline btQuaternion SimulatorToBullet(const CQuaternion& c_a_quaternion) {
#     return btQuaternion(c_a_quaternion.GetX(), c_a_quaternion.GetZ(),
#                         -c_a_quaternion.GetY(), c_a_quaternion.GetW());
#  }
# ~


class Record:
    def __init__(self, name):
        self.name = name
        self.record_states = []
        self.record_timesteps = []

    def record(self, controller, timestep):
        pos = controller._robot.base_position
        rot = controller._robot.base_orientation_quat
        conv_pos = np.array([pos[0], pos[1], pos[2]])
        conv_rot = np.array([rot[3], rot[0], rot[1], rot[2]])
        state = np.hstack([conv_pos, conv_rot, controller._robot.motor_angles])
        self.record_states.append(state)
        self.record_timesteps.append(timestep)

    def save_record(self):
        global record_angles
        global record_timesteps
        # np_record_angles = np.array(self.record_angles)
        # np_record_timesteps = np.array(self.record_timesteps)
        # np.save(f"{self.name}_angles.npy", np_record_angles)
        # np.save(f"{self.name}_timesteps.npy", np_record_timesteps)
        with open(f"{self.name}_angles.txt", "w") as fp:

            fp.write(
                """
{
    "LoopMode": "Wrap",
    "FrameDuration": 0.01667,
    "EnableCycleOffsetPosition": true,
    "EnableCycleOffsetRotation": false,

    "Frames":
    [
"""
            )
            root_x, root_y = self.record_states[0][:2]
            for row in self.record_states:
                fp.write("        [ {:6.5f}, {:6.5f},".format(row[0]-root_x, row[1]-root_y))
                s = ""
                for val in row[2:]:
                    s = s + " {:6.5f},".format(val)
                fp.write(s + "],\n")
            fp.write(
                """    ]
}
""")


def _update_controller(controller):
    # Update speed
    lin_speed, rot_speed = [0.0, 0.0], 0.3
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

        while current_time - start_time < FLAGS.max_time_secs + 5:
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
