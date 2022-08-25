"""Example of MPC controller on A1 robot."""
from potential_planner import Planner
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

'''
class Planner:
    def __init__(self, pos, goal, tolerance=0.5) -> None:
        self.goal = np.array(goal).reshape([2, 1])
        self.tolerance = tolerance
        self.pos = pos

    def at_goal(self):
        dist = np.linalg.norm(self.goal - self.pos)
        if dist < self.tolerance:
            return True
        return False

    def get_command(self, pos):
        self.pos = pos
        return [0, 0, 0]
'''


class Fixer:
    def __init__(self) -> None:
        self.last_o = 0.0
        self.int_o = 0.0
        self.Kp = 0.8
        self.Kd = 0.1
        self.Ki = 0.0

    def get_fix(self, current_o, dt=0.02):
        err_o = current_o - self.last_o
        d_o = err_o / dt
        self.int_o += err_o
        self.last_o = current_o

        return current_o - (self.Kp * err_o + self.Kd * d_o + self.Ki * self.int_o)


class StateEstimator:
    def __init__(self, pos=[0, 0]) -> None:
        self.pos = np.array(pos).astype(float) .reshape([2, 1])
        self.mutex = Lock()

    def robot2world(self, orientation, v, dt=1.):
        Rwb = np.array([[np.cos(orientation), -np.sin(orientation)],
                        [np.sin(orientation),  np.cos(orientation)]
                        ])
        return Rwb @ np.array(v[:2]).reshape([2, 1]) * dt

    def world2robot(self, orientation, v, dt=1.):
        Rwb = np.array([[np.cos(orientation), -np.sin(orientation)],
                        [np.sin(orientation),  np.cos(orientation)]
                        ])
        return Rwb.T @ v[: 2].reshape([2, 1]) * dt

    def update(self, orientation, v, dt=0.02):
        # rotate from the current orientation to the world frame (which should
        # correspond to the initialization direction)
        v_world = self.robot2world(orientation, v, dt)

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
    lin_speed, rot_speed = command[: 2], command[2]
    controller.set_desired_speed(lin_speed, rot_speed)
    # Update controller moce
    controller.set_controller_mode(ControllerMode.WALK)
    # Update gait
    controller.set_gait(GaitType.FLYTROT)


def main(argv):
    del argv  # unused

    # Dummy state estimator and planner
    state_estimator = StateEstimator(controller._conf.timestep)
    planner = Planner(0.2)

    controller = locomotion_controller.LocomotionController(
        FLAGS.use_real_robot,
        FLAGS.show_gui,
        world_class=WORLD_NAME_TO_CLASS_MAP[FLAGS.world],
        pos_estimator=state_estimator)

    try:
        current_time = time.time()
        last_time = current_time
        current_p = state_estimator.get_pos()

        def is_done(self, goal, robot_pose, delta):
            dist = np.linalg.norm(robot_pose - goal)
            done = (dist < delta).all()
            return done

        while not is_done(goal, current_p, 0.001):
            # update time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # update position
            current_o = controller._robot.base_orientation_rpy[2]
            current_p = state_estimator.get_pos()

            v = planner.get_command(current_p)
            o = Fixer.get_fix(current_o=current_o, dt=dt)
            v = state_estimator.world2robot(o, v)
            command = [v[0], v[1], o]
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
