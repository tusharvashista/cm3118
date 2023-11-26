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


class RRTstar:
    """RRTstar algorithm implementation as a planner to solve start-to-goal queries"""

    def __init__(self):
        self.start: Node = None
        self.goal: Node = None

        self.nodes = {}

        self.step_size = 0.001
        self.expansion_size = 0.5
        self.bias = 0.5

        self.collision_checker: CollisionChecker = None
        self.tolerance = None

        self.solution_node = None

        self.solution_path = None

        self.path_publisher = rospy.Publisher("/prior_path", Path, queue_size=10)

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
        """Goal tolerance to check when the query has been solved"""
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

        start_coord = self.collision_checker.coordinates_to_indices(
            self.start.x, self.start.y
        )
        self.nodes[start_coord] = self.start

        timeout = time.time() + termination_time

        while timeout > time.time():
            temp_val = random.uniform(0, 1)
            if 0 < temp_val < self.bias:
                nrand = self.goal
            else:
                nrand = self.sample()

            nnear = self.nearest(nrand)
            new_node = self.step(nnear, nrand)
            new_node_coord = self.collision_checker.coordinates_to_indices(
                new_node.x, new_node.y
            )

            if new_node_coord not in self.nodes.keys():
                if self.collision_checker.is_node_free(new_node):
                    neighbors = self.neighbor_nodes(new_node)
                    cur_cost = float("inf")
                    parent_node = None

                    for node in neighbors:
                        new_cost = self.cost(
                            self.nodes[node]
                        ) + new_node.calculate_distance(self.nodes[node])

                        if new_cost < cur_cost:
                            cur_cost = new_cost
                            parent_node = self.nodes[node]

                    if parent_node:
                        new_node.parent = parent_node
                        self.nodes[new_node_coord] = new_node

                        neighbors.remove(
                            self.collision_checker.coordinates_to_indices(
                                parent_node.x, parent_node.y
                            )
                        )

                        if start_coord in neighbors:
                            neighbors.remove(start_coord)

                        self.check_and_reconnect(new_node, neighbors)

                        if new_node.calculate_distance(self.goal) < self.tolerance:
                            self.solution_node = new_node
                            rospy.loginfo(
                                "Initial solution found at: "
                                + str(time.time() - (timeout - termination_time))
                                + " seconds."
                            )
                            self.solution_path = self.get_branch(self.solution_node)
                            init_path_length = self.get_path_length()
                            rospy.loginfo(
                                "Initial solution path length: "
                                + str(init_path_length)
                                + " meters."
                            )

        if self.solution_node:
            self.solution_path = self.get_branch(self.solution_node)
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

    def cost(self, _n: Node):
        """Returns the cost of a node to the start node"""
        if not _n.parent:
            return 0
        cost = 0
        cur_node = _n
        next_node = self.nodes[
            self.collision_checker.coordinates_to_indices(
                cur_node.parent.x, cur_node.parent.y
            )
        ]
        cost += cur_node.calculate_distance(next_node)
        while not next_node == self.start:
            cur_node = next_node
            next_node = self.nodes[
                self.collision_checker.coordinates_to_indices(
                    cur_node.parent.x, cur_node.parent.y
                )
            ]
            cost += cur_node.calculate_distance(next_node)
        return cost

    def neighbor_nodes(self, _n: Node):
        """Returns indexed of nodes inside the range of expansion size"""
        nei_nodes = []
        for node in self.nodes:
            if _n.calculate_distance(
                self.nodes[node]
            ) <= self.expansion_size and self.collision_checker.is_connection_free(
                _n, self.nodes[node], self.step_size
            ):
                nei_nodes.append(node)
        return nei_nodes

    def check_and_reconnect(self, _n: Node, neighbors):
        """Checks and reconnects nearby neighbors with _n depending on cost"""
        for node in neighbors:
            cur_cost = self.cost(self.nodes[node])
            new_cost = self.cost(_n) + _n.calculate_distance(self.nodes[node])
            if new_cost < cur_cost:
                self.nodes[node].parent = _n

    def get_branch(self, _n):
        """Obtain nodes of branch in tree"""
        branch = [_n]
        cur_node = _n
        while cur_node.parent:
            cur_node = self.nodes[
                self.collision_checker.coordinates_to_indices(
                    cur_node.parent.x, cur_node.parent.y
                )
            ]
            branch.append(cur_node)
        branch.reverse()
        return branch

    def nodes_to_path_msg(self, path_nodes: list) -> Path:
        """Transforms the list of path nodes into a Path type of object"""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()

        for node in path_nodes:
            path.poses.append(node.to_pose_stamped())

        return path
