#!/usr/bin/env python3
import time
import rospy
from collision_checker import CollisionChecker
from node import Node
from planners.rrt_star import RRTstar
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool
from tf import TransformListener, ExtrapolationException, LookupException
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker


class Navigator:
    """Manages the incoming start-to-goal query, solves it by requesting a map and sends the path to the controller to follow"""

    def __init__(self):
        #! Start and goal nodes for the query
        self.start = None
        self.goal = None

        self.collision_checker = None

        # ========================================

        #! Flags to check the status of the robot, the query and the map
        self.is_goal_cancelled = False
        self.is_goal_reached = False
        self.is_map_loaded = False
        self.is_robot_moving = False

        # ! Planner parameters
        self.step_size = rospy.get_param("~step_size", 0.001)
        self.expansion_size = rospy.get_param("~expansion_size", 0.5)
        self.solve_time = rospy.get_param("~solve_time", 5)
        self.bias = rospy.get_param("~bias", 0.5)

        self.goal_tolerance = rospy.get_param(
            "~goal_tolerance", 0.1
        )  # tolerance of the solution path to the goal

        self.max_planning_bounds = rospy.get_param("~max_planning_bounds", [20, 20])

        # ? flag to decide use of path smoother
        self.use_path_smoother = rospy.get_param("~use_path_smoother", False)

        # ========================================

        #! Information about the robot
        self.robot_position = None
        self.robot_orientation = None
        self.robot_radius = rospy.get_param("~robot_radius", 0.15)
        # ========================================

        self.map_topic = rospy.get_param("~map_topic", default="/map")

        self.transform_listener = TransformListener()

        time.sleep(1)

        #! SUBSCRIBERS
        self.start_subscriber = rospy.Subscriber(
            "/tf", TFMessage, self.robot_pose_callback
        )
        self.goal_subscriber = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.goal_query_callback
        )

        # ? subscribers to communicate with the controller
        self.robot_moving_subscriber = rospy.Subscriber(
            "/robot_is_moving", Bool, self.robot_moving_callback
        )

        self.goal_reached_subscriber = rospy.Subscriber(
            "/goal_reached", Bool, self.goal_reached_callback
        )

        # =======================================

        #! PUBLISHERS
        # ? publishers to communicate with the controller
        self.path_publisher = rospy.Publisher("/path", Path, queue_size=10)
        self.stop_motion_publisher = rospy.Publisher(
            "/stop_motion", Bool, queue_size=10
        )
        self.goal_publisher = rospy.Publisher(
            "/goal_controller", PoseStamped, queue_size=10
        )
        self.tree_publisher = rospy.Publisher("/rrt", Marker, queue_size=10)

        # ========================================

    def goal_reached_callback(self, goal_reached: Bool):
        """Listens from the controller wether the robot has reached the goal"""
        self.is_goal_reached = goal_reached

    def robot_moving_callback(self, robot_moving: Bool):
        """Listens from the controller wether the robot is moving or not"""
        self.is_robot_moving = robot_moving

    def robot_pose_callback(self, data: TFMessage):
        """Listen to robot current position"""
        try:
            position, quaternion = self.transform_listener.lookupTransform(
                "/map", "/base_link", rospy.Time()
            )
        except (ExtrapolationException, LookupException):
            return

        self.robot_position = position
        self.robot_orientation = quaternion

    def goal_query_callback(self, data: PoseStamped) -> bool:
        """Obtain goal query and ask for the planning callback"""
        if self.is_robot_moving:
            self.cancel_goal()

        self.goal = data.pose
        self.is_goal_cancelled = False
        self.is_goal_reached = False

        rospy.loginfo("Received new goal")
        self.goal_publisher.publish(data)
        return self.planning_callback()

    def planning_callback(self):
        """Main planning function where the whole process is carried,
        definition of the start and goal, map and solving of the query"""
        #! Creating Variables to store the total time taken to for the planning and reach the goal
        startTime = rospy.Time.now().to_sec() # Recording the starting time 

        #! Defining the goal states as nodes, as goal will remain the same
        goal = Node.from_pose(self.goal)

        #! Starting with an empty map
        map = None

        #! Creating a loop for the planning function, to allow for replanning
        while not self.is_goal_reached and not rospy.is_shutdown():

            #! 1. Define start nodes, as start position will change, in other words getting the current position
            start = Node.from_tf(self.robot_position, self.robot_orientation)
            

            #! 2. Updating the map every iteration
            
            map = rospy.wait_for_message(self.map_topic, OccupancyGrid, timeout=5)

            if map:
                self.is_map_loaded = True

            #! 3. Redefining Collision Checker, every iteration

            self.collision_checker = CollisionChecker(map, self.robot_radius, self.max_planning_bounds)

            #! 4. Check if Start and Goal states are valid every iteration

            if (
                (
                    not self.collision_checker.is_node_free(goal)
                    and not self.collision_checker.is_node_free(start)
                )
                or not self.collision_checker
                or not start
            ):
                rospy.logwarn("Goal can't be reached, either start or goal are in collision") # Sending a warning if the goal or the start position are invalid
                self.stop_motion_publisher.publish(True)
                self.path_publisher.publish(self.nodes_to_path_msg([]))  # Clearing path from RViz
                return False

            #! 5. Define planner
         
            planner = RRTstar()
            planner.set_start(start)  # define start robot position
            planner.set_goal(goal)  # define goal
            planner.set_collision_checker(self.collision_checker)  # set collision node checker
            planner.set_goal_tolerance(self.goal_tolerance)  # tolerance to find the goal
            planner.set_expansion_size(self.expansion_size)
            planner.set_step_size(self.step_size)
            planner.set_bias(self.bias)

            #! 6. Solve the query

            rospy.loginfo("Searching solution path...")
            solved = planner.solve(self.solve_time)

            if solved:  # if the query is solved
                #self.stop_motion_publisher.publish(False)
                path_length = planner.get_path_length()
                rospy.loginfo(f"Path length: {path_length} meters")
                solution_path = planner.get_solution_path()
                rospy.loginfo("Solution path found")
                if self.use_path_smoother:
                    solution_path = self.moving_average(solution_path)
                path_msg = self.nodes_to_path_msg(solution_path)
                
                
                #! 7. Send path to controller
                self.path_publisher.publish(path_msg)
                
                # publish tree
                rrt = planner.get_tree()
                self.publish_tree(rrt)
                
                 #! checking if we have reached the goal
                if self.is_goal_reached:
                    self.path_publisher.publish(self.nodes_to_path_msg([]))  # Clearing path from RViz
                    #! calculating and displaying the total time taken for planning and reaching the goal for the ease of storing and using the information to compare with different param values
                    elapsedTime = rospy.Time.now().to_sec() - startTime
                    rospy.loginfo(f"Total Time taken: {elapsedTime} seconds")
                    return True
                
            else:
                
                # publish tree
                rrt = planner.get_tree()
                self.publish_tree(rrt)
                self.stop_motion_publisher.publish(True)
                rospy.logwarn("No path found")
                self.path_publisher.publish(self.nodes_to_path_msg([]))  # Clearing path from RViz
                
            
           
            
            rospy.sleep(3)
        return False

    def moving_average(self, path: list, window: int = 4) -> list:
        """Smoothes the path obtained by finding an average"""

        window_queue = []
        smoothed_path = [path[0]]

        for node in path:
            if len(window_queue) == window:
                smoothed_path.append(sum(window_queue) / window)  # Mean
                window_queue.pop(0)

            window_queue.append(node)
        goal = Node.from_pose(self.goal)
        return smoothed_path + [goal]

    def nodes_to_path_msg(self, path_nodes: list) -> Path:
        """Transforms the list of path nodes into a Path type of object"""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()

        for node in path_nodes:
            path.poses.append(node.to_pose_stamped())

        return path

    def publish_tree(self, tree):
        visual_rrt = Marker()
        visual_rrt.header.frame_id = "map"
        visual_rrt.header.stamp = rospy.Time.now()
        visual_rrt.action = Marker.ADD

        # visual_rrt.ns = "planner_rrt"
        visual_rrt.type = Marker.LINE_LIST

        visual_rrt.pose.orientation.w = 1.0

        visual_rrt.id = 0

        visual_rrt.scale.x = 0.03

        visual_rrt.color.b = 1.0
        visual_rrt.color.a = 1.0

        for branch in tree:
            node = tree[branch]

            if node.parent:
                p = Point()
                p.x = node.x
                p.y = node.y
                p.z = 0.1

                visual_rrt.points.append(p)

                p = Point()

                p.x = node.parent.x
                p.y = node.parent.y
                p.z = 0.1
                visual_rrt.points.append(p)

        self.tree_publisher.publish(visual_rrt)

    def wait_for_result(self, duration: float) -> bool:
        sending_time = rospy.Time.now().to_sec()
        rospy.sleep(1)

        while (
            self.is_robot_moving
            and not self.is_goal_cancelled
            and not rospy.is_shutdown()
        ):
            if rospy.Time.now().to_sec() - sending_time > duration:
                self.cancel_goal()
                return False

        return self.is_goal_reached

    def send_goal(self, pose: Pose) -> bool:
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose = pose

        return self.goal_query_callback(goal)

    def cancel_goal(self):
        self.stop_motion_publisher.publish(True)
        self.is_goal_cancelled = True


################################################################
# MAIN DEPLOY


if __name__ == "__main__":
    navigator = None

    def goto_goal(pos_x: float, pos_y: float):
        goal = Pose()
        goal.position.x = pos_x
        goal.position.y = pos_y
        goal.orientation.w = 1.0

        navigator.send_goal(goal)

        # Is result successful
        return navigator.wait_for_result(120)

    def on_shutdown():
        navigator.cancel_goal()

    try:
        rospy.init_node("navigator_node", anonymous=True)

        navigator = Navigator()
        rospy.on_shutdown(on_shutdown)

        while not rospy.is_shutdown():
            pos_x = float(input("Input X coordinate:"))
            pos_y = float(input("Input Y coordinate:"))

            goto_goal(pos_x, pos_y)
    except rospy.ROSInterruptException:
        pass
    except ValueError:
        print("Error: not a number")
