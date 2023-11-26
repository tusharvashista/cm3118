import math
import time
from functools import wraps
import rospy
from node import Node


# Time measurement function
# =============================
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


# =============================


class DStarLite:
    """D* Lite algorithm implementation as a planner to solve start-to-goal queries"""

    def __init__(self):
        self.node_start = None
        self.node_goal = None

        self.s_start = None
        self.s_goal = None

        self.collision_checker = None
        self.tolerance = None

        self.solution_path = None

        self.g, self.rhs, self.U = {}, {}, {}
        self.km = 0

    # SETTERS
    # =============================
    def set_start(self, start):
        """Set start position of the robot"""
        self.node_start = start

    def set_goal(self, goal):
        """Set goal of the start to goal query"""
        self.node_goal = goal

    def set_collision_checker(self, collision_checker):
        """Set collision checker to detect collision free nodes"""
        self.collision_checker = collision_checker
        self.s_start = self.collision_checker.coordinates_to_indices(
            self.node_start.x, self.node_start.y
        )
        self.s_goal = self.collision_checker.coordinates_to_indices(
            self.node_goal.x, self.node_goal.y
        )

    def set_goal_tolerance(self, tolerance):
        """Goal tolerance to check when the query has been solved"""
        self.tolerance = tolerance

    # =============================

    # GETTERS
    # =============================
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

    # =============================

    def get_neighbor(self, s_):
        """Generate neighbors from node s_, neighbors generated manually
        instead of using Node function since indices were desired"""
        nei_list = set()
        if not (s_[0] + 1) > (self.collision_checker.map_height - 1):
            nei_list.add((s_[0] + 1, s_[1]))
        if not (s_[0] - 1) < 0:
            nei_list.add((s_[0] - 1, s_[1]))

        if not (s_[1] + 1) > (self.collision_checker.map_width - 1):
            nei_list.add((s_[0], s_[1] + 1))
        if not (s_[1] - 1) < 0:
            nei_list.add((s_[0], s_[1] - 1))

        if not (s_[0] + 1) > (self.collision_checker.map_height - 1) and not (
            s_[1] + 1
        ) > (self.collision_checker.map_width - 1):
            nei_list.add((s_[0] + 1, s_[1] + 1))

        if not (s_[0] - 1) < 0 and not (s_[1] + 1) > (
            self.collision_checker.map_width - 1
        ):
            nei_list.add((s_[0] - 1, s_[1] + 1))

        if (
            not (s_[0] + 1) > (self.collision_checker.map_height - 1)
            and not (s_[1] - 1) < 0
        ):
            nei_list.add((s_[0] + 1, s_[1] - 1))

        if not (s_[0] - 1) < 0 and not (s_[1] - 1) < 0:
            nei_list.add((s_[0] - 1, s_[1] - 1))

        return nei_list

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_start]
        s = self.s_start

        for k in range(1000):
            g_list = {}
            for nei in self.get_neighbor(s):
                x, y = self.collision_checker.indices_to_coordinates(nei[0], nei[1])
                node = Node(x, y)
                if self.collision_checker.is_node_free(node):
                    g_list[nei] = self.g[nei]
            s = min(g_list, key=g_list.get)
            path.append(s)
            if s == self.s_goal:
                break

        self.solution_path = []

        for i in path:
            x, y = self.collision_checker.indices_to_coordinates(i[0], i[1])
            node = Node(x, y)
            self.solution_path.append(node)

        if len(self.solution_path) > 2:
            self.solution_path = self.solution_path[2:]

    # ==================================================
    # D* LITE COMPLEMENTARY FUNCTIONS FOR SOLVING QUERY
    # ==================================================
    def compute_path(self) -> list:
        while True:
            try:
                s, v = self.top_key()
            except:
                return False
            if (
                v >= self.calculate_key(self.s_start)
                and self.rhs[self.s_start] == self.g[self.s_start]
            ):
                return True

            k_old = v
            self.U.pop(s)

            if k_old < self.calculate_key(s):
                self.U[s] = self.calculate_key(s)
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for nei in self.get_neighbor(s):
                    self.update_vertex(nei)
            else:
                self.g[s] = float("inf")
                self.update_vertex(s)
                for nei in self.get_neighbor(s):
                    self.update_vertex(nei)

    def update_vertex(self, s):
        if s != self.s_goal:
            self.rhs[s] = float("inf")

            for nei in self.get_neighbor(s):
                x, y = self.collision_checker.indices_to_coordinates(nei[0], nei[1])
                node = Node(x, y)
                if self.collision_checker.is_node_free(node):
                    self.rhs[s] = min(
                        self.rhs[s],
                        self.g[nei] + self.cost(s, nei),
                    )
                else:
                    self.rhs[s] = min(
                        self.rhs[s],
                        self.g[nei] + float("inf"),
                    )

        if s in self.U:
            self.U.pop(s)

        if self.g[s] != self.rhs[s]:
            self.U[s] = self.calculate_key(s)

    def calculate_key(self, s):
        return [
            min(self.g[s], self.rhs[s]) + self.cost(self.s_start, s) + self.km,
            min(self.g[s], self.rhs[s]),
        ]

    def top_key(self):
        """
        :return: return the min key and its value.
        """
        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        x0, y0 = self.collision_checker.indices_to_coordinates(s_start[0], s_start[1])
        x1, y1 = self.collision_checker.indices_to_coordinates(s_goal[0], s_goal[1])
        return math.hypot(x1 - x0, y1 - y0)

    # ==========================================================
    # ==========================================================

    @timeit
    def solve(self) -> list:
        """Function to find path from start to goal using D* lite algorithm"""

        # CHECK IF ALL THE REQUIRED ATTRIBUTES HAVE BEEN SET
        if not self.collision_checker:
            rospy.logwarn("Collision checker has not been set on the planner")
            return False
        elif not self.s_start:
            rospy.logwarn("Start position has not been set on the planner")
            return False
        elif not self.s_goal:
            rospy.logwarn("Goal position has not been set on the planner")
            return False
        # =============================

        # START SOLVING THE QUERY
        for i in range(0, self.collision_checker.map_width - 1):
            for j in range(0, self.collision_checker.map_height - 1):
                self.rhs[(i, j)] = float("inf")
                self.g[(i, j)] = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.calculate_key(self.s_goal)

        path_found = False

        try:
            path_found = self.compute_path()
        except:
            return False

        if path_found:
            self.extract_path()
            return True
        return False

    @timeit
    def update_path(self, collision_checker, start, nodes):
        """Function to update path from start to goal using D* lite algorithm when nodes
        corresponds to states that are in collision"""
        self.node_start = start
        self.collision_checker = collision_checker

        self.s_start = self.collision_checker.coordinates_to_indices(
            self.node_start.x, self.node_start.y
        )

        s_last = self.s_start
        s_curr = self.s_start
        i = 0
        path = [self.s_start]

        occ_nodes = nodes

        while s_curr != self.s_goal:
            s_list = {}

            for s in self.get_neighbor(s_curr):
                x, y = self.collision_checker.indices_to_coordinates(s[0], s[1])
                node = Node(x, y)

                if self.collision_checker.is_node_free(node):
                    s_list[s] = self.g[s] + self.cost(s_curr, s)
                else:
                    s_list[s] = self.g[s] + float("inf")
            s_curr = min(s_list, key=s_list.get)
            path.append(s_curr)

            if i < 1:
                self.km += self.cost(s_last, s_curr)
                s_last = s_curr

                for s in occ_nodes:
                    self.update_vertex(s)
                    for nei in self.get_neighbor(s):
                        self.update_vertex(nei)
                i += 1

                if not self.compute_path():
                    return False

        self.extract_path()
        if len(self.solution_path) > 0:
            return True
        return False
