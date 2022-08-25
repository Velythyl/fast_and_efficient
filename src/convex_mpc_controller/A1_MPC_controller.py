"""Example of MPC controller on A1 robot."""
from threading import Lock
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


class Planner:
    def __init__(self, goal, tolerance=0.5) -> None:
        self.goal = np.array(goal).reshape([2, 1])
        self.tolerance = tolerance

    def at_goal(self):
        dist = np.norm(self.goal - self.pos)
        if dist < self.tolerance:
            return True
        return False

    def get_command(self, pos):
        self.pos = pos
        return [0, 0, 0]


class StateEstimator:
    def __init__(self, pos=[0, 0], dt=0.02) -> None:
        self.pos = np.array(pos).reshape([2, 1])
        self.mutex = Lock()

    def update(self, orientation, v, dt=0.02):
        # rotate from the current orientation to the world frame (which should
        # correspond to the initialization direction)
        Rwb = np.array([[np.cos(orientation), -np.sin(orientation)],
                        [np.sin(orientation),  np.cos(orientation)]
                        ])
        v_world = Rwb @ v[:2].reshape([2, 1]) * dt

        self.mutex.acquire()
        try:
            self.pos += v_world
        finally:
            self.mutex.release()

    def get_pos(self):
        self.mutex.acquire()
        try:
            return self.pos
        finally:
            self.mutex.release()


def _update_controller(controller, command):
    # Update speed
    lin_speed, rot_speed = command[:2], command[2]
    controller.set_desired_speed(lin_speed, rot_speed)
    # Update controller moce
    controller.set_controller_mode(ControllerMode.WALK)
    # Update gait
    controller.set_gait(GaitType.FLYTROT)


def main(argv):
    del argv  # unused

    # Dummy state estimator and planner
    state_estimator = StateEstimator(controller._conf.timestep)
    planner = Planner()

    controller = locomotion_controller.LocomotionController(
        FLAGS.use_real_robot,
        FLAGS.show_gui,
        world_class=WORLD_NAME_TO_CLASS_MAP[FLAGS.world],
        state_estimator=state_estimator)

    try:
        start_time = 0  # controller.time_since_reset
        current_time = start_time
        at_goal = False

        while not planner.at_goal():
            current_time = controller.time_since_reset
            command = planner.get_command(state_estimator.get_pos())
            _update_controller(controller, command)

            if not controller.is_safe:
                break

            # sleep until next time
            time.sleep(0.05)

    finally:
        controller.set_controller_mode(
            locomotion_controller.ControllerMode.TERMINATE)


if __name__ == "__main__":
    app.run(main)
