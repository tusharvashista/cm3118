from nav_msgs.msg import OccupancyGrid
from node import Node
import math
import rospy
from geometry_msgs.msg import Point


class CollisionChecker:
    """Collision checker to if nodes are free of collision or not. Uses OccupancyGrid in order to check the collision based on robot radius. Contains additional function to get indexes from the OccupancyGrid and coordinates"""

    def __init__(
        self,
        grid_map: OccupancyGrid,
        robot_radius: float,
        planning_bounds: list = [20, 20],
        opportunistic_cc: bool = True,
    ):
        self.map_data = grid_map.data
        self.map_width = grid_map.info.width
        self.map_height = grid_map.info.height
        self.map_resolution = grid_map.info.resolution
        self.map_origin = grid_map.info.origin.position

        self.robot_radius = robot_radius

        self.planning_bounds = planning_bounds

        self.tolerance = math.ceil(self.robot_radius / self.map_resolution)

        if opportunistic_cc:
            self.expand_grid_map()

    def expand_grid_map(self):
        if len(self.planning_bounds) > 1:
            # new map
            map_width = int(self.planning_bounds[0] / self.map_resolution)
            map_height = int(self.planning_bounds[1] / self.map_resolution)
            map_origin = Point()
            map_origin.x = -map_width * self.map_resolution / 2
            map_origin.y = -map_height * self.map_resolution / 2

            map_data = [0] * map_width * map_height

            for i in range(0, self.map_width):
                for j in range(0, self.map_height):
                    value = self.get_by_indices(i, j)
                    if value == 100:
                        _x, _y = self.indices_to_coordinates(i, j)
                        ni_ = int((_x + map_origin.x) / self.map_resolution)
                        nj_ = int((_y + map_origin.y) / self.map_resolution)
                        index = nj_ * map_width + ni_
                        map_data[index] = 100

            self.map_data = map_data
            self.map_width = map_width
            self.map_height = map_height
            self.map_origin = map_origin

    def get_by_indices(self, i: int, j: int) -> int:
        """Obtain index of OccupancyGrid data according to i and j from grid.

        Args:
            i (int): index in x
            j (int): index in j

        Returns:
            int: index from data list in OccupancyGrid
        """
        index = None
        try:
            index = self.map_data[j * self.map_width + i]
        except:
            rospy.logwarn("Position not available in map")
        return index

    def get_by_coordinates(self, x: float, y: float) -> int:
        """Obtain index of OccupancyGrid data according to i and j from grid.

        Args:
            x (float): x coordinate in the world
            y (float): y coordinate in the world

        Returns:
            int: index from data list in OccupancyGrid
        """
        indices = self.coordinates_to_indices(x, y)
        return self.get_by_indices(indices[0], indices[1])

    def coordinates_to_indices(self, x: float, y: float) -> tuple:
        """Transform coordinates to indices of the grid map

        Args:
            x (float): x coordinate of the grid map
            y (float): y coordinate of the grid map

        Returns:
            i, j (tuple): tuple of the indexes
        """
        i = int((x - self.map_origin.x) / self.map_resolution)
        j = int((y - self.map_origin.y) / self.map_resolution)
        return i, j

    def indices_to_coordinates(self, i: int, j: int) -> tuple:
        """Transforms indices to coordinates of the grid map

        Args:
            i (int): index in x axis grid map
            j (int): index in y axis grid map

        Returns:
            x, y (tuple): tuple of the coordinates
        """
        x = i * self.map_resolution + self.map_origin.x
        y = j * self.map_resolution + self.map_origin.y
        return x, y

    def is_node_free(self, node: Node) -> bool:
        """Check if a node is collision free according to its position in the grid_map

        Args:
            node (Node): the node to check if collision free

        Returns:
            bool: Wether the node is collision free or not
        """
        i, j = self.coordinates_to_indices(node.x, node.y)

        if not 0 <= i < self.map_width or not 0 <= j < self.map_height:
            return False

        for offset_x in range(-self.tolerance, self.tolerance):
            for offset_y in range(-self.tolerance, self.tolerance):
                val = self.get_by_indices(i - offset_x, j - offset_y)
                if val is None:
                    return False
                if not -1 < val < 100:
                    return False
        return True

    def is_path_free(self, path):
        """Checks whether a path is free of collision or not by
        returning the list of nodes that are in collision.

        Args:
            path (list(Node)): list of Node objects

        Returns:
            occ_nodes (list((x, y))): list of coordinates as tuples
        """
        occ_nodes = []

        for node in path:
            if not self.is_node_free(node):
                occ_nodes.append(self.coordinates_to_indices(node.x, node.y))
        if len(occ_nodes) > 0:
            rospy.logwarn("Current path is not collision free, calculating a new path")

        return occ_nodes
