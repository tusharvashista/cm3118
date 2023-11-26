import math
import time
from functools import wraps
import random
import rospy
from collision_checker import CollisionChecker
from node import Node
from nav_msgs.msg import Path


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        rospy.loginfo(
            f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds"
        )
        return result

    return timeit_wrapper


class RRT:
    """RRT algorithm implementation as a planner to solve start-to-goal queries"""

    def __init__(self):
        self.start: Node = None
        self.goal: Node = None

        self.nodes = {}

        self.step_size = 0.001
        self.expansion_size = 0.5
        self.bias = 0.5

        self.collision_checker: CollisionChecker = None
        self.tolerance = None

        self.solution_path = None

    def set_start(self, start):
        """Set start position of the robot"""
        self.start = start

    def set_goal(self, goal):
        """Set goal of the start to goal query"""
        self.goal = goal

    def set_collision_checker(self, collision_checker):
        """Set collision checker to detect collision free nodes"""
        self.collision_checker = collision_checker

    def set_goal_tolerance(self, tolerance):
        """Set goal tolerance to check when the query has been solved"""
        self.tolerance = tolerance

    def set_step_size(self, step_size):
        """Set the step size to determine if connection between two
        nodes is possible with collision checking"""
        self.step_size = step_size

    def set_expansion_size(self, expansion_size):
        """Set the maximum expansion for a new node in the tree"""
        self.expansion_size = expansion_size

    def set_bias(self, bias):
        """Set the bias of the planner"""
        self.bias = bias

    def get_solution_path(self):
        """Obtain stored solution path"""
        return self.solution_path

    def get_path_length(self):
        """Obtain the length of the solution path"""
        distance = 0

        for i in range(0, len(self.solution_path) - 2):
            distance += math.sqrt(
                math.pow(self.solution_path[i].x - self.solution_path[i + 1].x, 2)
                + math.pow(self.solution_path[i].y - self.solution_path[i + 1].y, 2)
            )
        return distance

    def get_tree(self):
        """Obtain the tree"""
        return self.nodes

    @timeit
    def solve(self, termination_time) -> list:
        """Function to find path from start to goal using Dijkstra algorithm"""

        if not self.collision_checker:
            rospy.logwarn("Collision checker has not been set on the planner")
            return False
        if not self.start:
            rospy.logwarn("Start position has not been set on the planner")
            return False
        if not self.goal:
            rospy.logwarn("Goal position has not been set on the planner")
            return False

        self.nodes[
            self.collision_checker.coordinates_to_indices(self.start.x, self.start.y)
        ] = self.start

        timeout = time.time() + termination_time

        while timeout > time.time():
            temp_val = random.uniform(0, 1)
            if 0 < temp_val < self.bias:
                nrand = self.goal
            else:
                nrand = self.sample()

            nnear = self.nearest(nrand)
            new_node = self.step(nnear, nrand)
            if self.collision_checker.is_node_free(
                new_node
            ) and self.collision_checker.is_connection_free(
                nnear, new_node, self.step_size
            ):
                new_node.parent = nnear
                self.nodes[
                    self.collision_checker.coordinates_to_indices(
                        new_node.x, new_node.y
                    )
                ] = new_node

                if new_node.calculate_distance(self.goal) < self.tolerance:
                    self.solution_path = new_node.backtrack_path()
                    return True

        return False

    def nearest(self, _n: Node):
        """Return the nearest node from _n"""
        nearest_node = None
        dis = float("inf")

        for node in self.nodes:
            cur_dis = _n.calculate_distance(self.nodes[node])
            if cur_dis < dis:
                dis = cur_dis
                nearest_node = self.nodes[node]
        return nearest_node

    def sample(self):
        """Sample node from the workspace"""
        _i = int(random.uniform(0, self.collision_checker.map_width - 1))
        _j = int(random.uniform(0, self.collision_checker.map_height - 1))
        _x, _y = self.collision_checker.indices_to_coordinates(_i, _j)
        node = Node(_x, _y)
        return node

    def step(self, nnear: Node, nrand: Node):
        """Defines the node step from nnear to nrand"""

        if nnear.calculate_distance(nrand) > self.expansion_size:
            theta = math.atan2(nrand.y - nnear.y, nrand.x - nnear.x)
            (_x, _y) = (
                nnear.x + self.expansion_size * math.cos(theta),
                nnear.y + self.expansion_size * math.sin(theta),
            )
            step_node = Node(_x, _y)
            return step_node
        return nrand

    def nodes_to_path_msg(self, path_nodes: list) -> Path:
        """Transforms the list of path nodes into a Path type of object"""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()

        for node in path_nodes:
            path.poses.append(node.to_pose_stamped())

        return path
