#include <iostream>
#include <vector>

#include <boost/bind.hpp>

// standard OMPL
#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/objectives/StateCostIntegralObjective.h>
#include <ompl/base/objectives/MaximizeMinClearanceObjective.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/config.h>

// ROS
#include <ros/ros.h>
#include <ros/package.h>
// ROS services
#include <std_srvs/Empty.h>
// ROS markers rviz
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <nav_msgs/Path.h>
// ROS tf
#include <tf/message_filter.h>
#include <tf/transform_listener.h>
// action server
#include <actionlib/server/simple_action_server.h>
#include <actionlib_msgs/GoalID.h>

// Planner
#include <state_validity_checker_octomap_fcl_R2.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

//!  PlannFramework class.
/*!
 * Planning Framework.
 * Setup a sampling-based planner for computation of collision-free paths.
 * C-Space: R2
 * Workspace is represented with Octomaps
 */
class PlannFramework
{
public:
    //! Constructor
    PlannFramework();
    //! Planner setup
    void run();
    //! Periodic callback to solve the query.
    void planningCallback();
    //! Callback for getting current vehicle odometry
    void odomCallback(const nav_msgs::OdometryConstPtr &odom_msg);
    //! Callback for getting the 2D navigation goal
    void queryGoalCallback(const geometry_msgs::PoseStampedConstPtr &nav_goal_msg);
    //! Procedure to visualize the resulting path
    void visualizeRRT(og::PathGeometric &geopath);

private:
    // ROS
    ros::NodeHandle nh_, local_nh_;
    ros::Timer timer_;
    ros::Subscriber odom_sub_, nav_goal_sub_;
    ros::Publisher solution_path_rviz_pub_, solution_path_pub_, num_nodes_pub_, goal_pub_, stop_motion_pub_;

    // ROS TF
    tf::Pose last_robot_pose_;
    tf::TransformListener tf_listener_;

    // OMPL planner
    og::SimpleSetupPtr simple_setup_;
    double timer_period_, solving_time_, goal_tolerance_, yaw_goal_tolerance_, robot_base_radius;
    bool odom_available_, goal_available_, visualize_tree_;
    std::vector<double> planning_bounds_x_, planning_bounds_y_, start_state_, goal_map_frame_, goal_odom_frame_;
    double goal_radius_;
    std::string planner_name_, odometry_topic_, query_goal_topic_, solution_path_topic_, world_frame_, octomap_service_;
    std::vector<const ob::State *> solution_path_states_;
};

//!  Constructor.
/*!
 * Load planner parameters from configuration file.
 * Publishers to visualize the resulting path.
 */
PlannFramework::PlannFramework()
    : local_nh_("~")
{
    //=======================================================================
    // Get parameters
    //=======================================================================
    planning_bounds_x_.resize(2);
    planning_bounds_y_.resize(2);
    start_state_.resize(2);
    goal_map_frame_.resize(3);
    goal_odom_frame_.resize(3);

    local_nh_.param("world_frame", world_frame_, world_frame_);
    local_nh_.param("planning_bounds_x", planning_bounds_x_, planning_bounds_x_);
    local_nh_.param("planning_bounds_y", planning_bounds_y_, planning_bounds_y_);
    local_nh_.param("start_state", start_state_, start_state_);
    local_nh_.param("goal_state", goal_map_frame_, goal_map_frame_);
    local_nh_.param("timer_period", timer_period_, timer_period_);
    local_nh_.param("solving_time", solving_time_, solving_time_);
    local_nh_.param("planner_name", planner_name_, planner_name_);
    local_nh_.param("odometry_topic", odometry_topic_, odometry_topic_);
    local_nh_.param("query_goal_topic", query_goal_topic_, query_goal_topic_);
    local_nh_.param("solution_path_topic", solution_path_topic_, solution_path_topic_);
    local_nh_.param("goal_tolerance", goal_tolerance_, 0.2);
    local_nh_.param("yaw_goal_tolerance", yaw_goal_tolerance_, 0.1);
    local_nh_.param("visualize_tree", visualize_tree_, false);
    local_nh_.param("robot_base_radius", robot_base_radius, robot_base_radius);

    goal_radius_ = goal_tolerance_;
    goal_available_ = false;

    //=======================================================================
    // Subscribers
    //=======================================================================
    // Odometry data
    odom_sub_ = nh_.subscribe(odometry_topic_, 1, &PlannFramework::odomCallback, this);
    odom_available_ = false;

    // 2D Nav Goal
    nav_goal_sub_ = local_nh_.subscribe(query_goal_topic_, 1, &PlannFramework::queryGoalCallback, this);

    //=======================================================================
    // Publishers
    //=======================================================================
    solution_path_rviz_pub_ = local_nh_.advertise<visualization_msgs::Marker>("solution_path", 10, true);
    solution_path_pub_ =
        local_nh_.advertise<nav_msgs::Path>("/path", 10, true);
    goal_pub_ = local_nh_.advertise<geometry_msgs::PoseStamped>("/goal_controller", 10, true);
    stop_motion_pub_ = local_nh_.advertise<std_msgs::Bool>("/stop_motion", 10, true);

    //=======================================================================
    // Waiting for odometry
    //=======================================================================
    ros::Rate loop_rate(10);
    while (ros::ok() && !odom_available_)
    {
        ros::spinOnce();
        loop_rate.sleep();
        ROS_WARN("%s:\n\tWaiting for vehicle's odometry\n", ros::this_node::getName().c_str());
    }
    ROS_WARN("%s:\n\tOdometry received\n", ros::this_node::getName().c_str());
}

//! Odometry callback.
/*!
 * Callback for getting updated vehicle odometry
 */
void PlannFramework::odomCallback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    if (!odom_available_)
        odom_available_ = true;
    tf::poseMsgToTF(odom_msg->pose.pose, last_robot_pose_);

    double useless_pitch, useless_roll, yaw;
    last_robot_pose_.getBasis().getEulerYPR(yaw, useless_pitch, useless_roll);

    if ((goal_available_) &&
        sqrt(pow(goal_odom_frame_[0] - last_robot_pose_.getOrigin().getX(), 2.0) +
             pow(goal_odom_frame_[1] - last_robot_pose_.getOrigin().getY(), 2.0)) < (goal_radius_ + 0.3) &&
        abs(yaw - goal_odom_frame_[2]) < (yaw_goal_tolerance_ + 0.08))
    {
        goal_available_ = false;
    }
}

//! Navigation goal callback.
/*!
 * Callback for getting the 2D navigation goal
 */
void PlannFramework::queryGoalCallback(const geometry_msgs::PoseStampedConstPtr &query_goal_msg)
{
    double useless_pitch, useless_roll, yaw;
    yaw = tf::getYaw(tf::Quaternion(query_goal_msg->pose.orientation.x, query_goal_msg->pose.orientation.y,
                                    query_goal_msg->pose.orientation.z, query_goal_msg->pose.orientation.w));

    goal_map_frame_[0] = query_goal_msg->pose.position.x; // x
    goal_map_frame_[1] = query_goal_msg->pose.position.y; // y
    goal_map_frame_[2] = yaw;

    //=======================================================================
    // Transform from map to odom
    //=======================================================================
    ros::Time t;
    std::string err = "";
    tf::StampedTransform tf_map_to_fixed;
    tf_listener_.getLatestCommonTime("map", "odom", t, &err);
    tf_listener_.lookupTransform("map", "odom", t, tf_map_to_fixed);
    tf_map_to_fixed.getBasis().getEulerYPR(yaw, useless_pitch, useless_roll);

    tf::Point goal_point_odom_frame(goal_map_frame_[0], goal_map_frame_[1], 0.0);
    goal_point_odom_frame = tf_map_to_fixed.inverse() * goal_point_odom_frame;
    goal_odom_frame_[0] = goal_point_odom_frame.getX();
    goal_odom_frame_[1] = goal_point_odom_frame.getY();
    goal_odom_frame_[2] = goal_map_frame_[2] - yaw;

    //=======================================================================
    // Clean and merge octomap
    //=======================================================================
    std_srvs::Empty::Request req;
    std_srvs::Empty::Response resp;

    solution_path_states_.clear();
    goal_available_ = true;

    goal_pub_.publish(query_goal_msg);

    planningCallback();
    goal_available_ = false;
}

//!  Planner setup.
/*!
 * Setup a sampling-based planner using OMPL.
 */
void PlannFramework::run()
{

    ros::Rate loop_rate(1 / (timer_period_ - solving_time_)); // 10 hz

    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}

//!  Periodic callback to solve the query.
/*!
 * Solve the query.
 */
void PlannFramework::planningCallback()
{

    //=======================================================================
    // Transform from map to odom
    //=======================================================================
    double useless_pitch, useless_roll, yaw;
    ros::Time t;
    std::string err = "";
    tf::StampedTransform tf_map_to_fixed;
    tf_listener_.getLatestCommonTime("map", "odom", t, &err);
    tf_listener_.lookupTransform("map", "odom", t, tf_map_to_fixed);
    tf_map_to_fixed.getBasis().getEulerYPR(yaw, useless_pitch, useless_roll);

    tf::Point goal_point_odom_frame(goal_map_frame_[0], goal_map_frame_[1], 0.0);
    goal_point_odom_frame = tf_map_to_fixed.inverse() * goal_point_odom_frame;
    goal_odom_frame_[0] = goal_point_odom_frame.getX();
    goal_odom_frame_[1] = goal_point_odom_frame.getY();
    goal_odom_frame_[2] = goal_map_frame_[2] - yaw;

    // ! 1. configuration space definition
    //=======================================================================
    // Instantiate the state space
    //=======================================================================
    ob::StateSpacePtr space = ob::StateSpacePtr(new ob::RealVectorStateSpace(2));

    //=======================================================================
    // Set the bounds for the state space
    //=======================================================================
    ob::RealVectorBounds bounds(2);

    if (last_robot_pose_.getOrigin().getX() < goal_odom_frame_[0])
    {
        if (last_robot_pose_.getOrigin().getX() - 10.0 < planning_bounds_x_[0])
            bounds.setLow(0, planning_bounds_x_[0]);
        else
            bounds.setLow(0, last_robot_pose_.getOrigin().getX() - 10.0);

        if (goal_odom_frame_[0] + 5.0 > planning_bounds_x_[1])
            bounds.setHigh(0, planning_bounds_x_[1]);
        else
            bounds.setHigh(0, goal_odom_frame_[0] + 10.0);
    }
    else
    {
        if (last_robot_pose_.getOrigin().getX() + 10.0 > planning_bounds_x_[1])
            bounds.setHigh(0, planning_bounds_x_[1]);
        else
            bounds.setHigh(0, last_robot_pose_.getOrigin().getX() + 10.0);

        if (goal_odom_frame_[0] - 10.0 < planning_bounds_x_[0])
            bounds.setLow(0, planning_bounds_x_[0]);
        else
            bounds.setLow(0, goal_odom_frame_[0] - 10.0);
    }

    if (last_robot_pose_.getOrigin().getY() < goal_odom_frame_[1])
    {
        if (last_robot_pose_.getOrigin().getY() - 10.0 < planning_bounds_y_[0])
            bounds.setLow(1, planning_bounds_y_[0]);
        else
            bounds.setLow(1, last_robot_pose_.getOrigin().getY() - 10.0);

        if (goal_odom_frame_[1] + 10.0 > planning_bounds_y_[1])
            bounds.setHigh(1, planning_bounds_y_[1]);
        else
            bounds.setHigh(1, goal_odom_frame_[1] + 10.0);
    }
    else
    {
        if (last_robot_pose_.getOrigin().getY() + 10.0 > planning_bounds_y_[1])
            bounds.setHigh(1, planning_bounds_y_[1]);
        else
            bounds.setHigh(1, last_robot_pose_.getOrigin().getY() + 10.0);

        if (goal_odom_frame_[1] - 10.0 < planning_bounds_y_[0])
            bounds.setLow(1, planning_bounds_y_[0]);
        else
            bounds.setLow(1, goal_odom_frame_[1] - 10.0);
    }

    space->as<ob::RealVectorStateSpace>()->setBounds(bounds);
    //=======================================================================
    // Define a simple setup class
    //=======================================================================
    simple_setup_ = og::SimpleSetupPtr(new og::SimpleSetup(space));
    ob::SpaceInformationPtr si = simple_setup_->getSpaceInformation();

    // ! 2. Define start and goals
    //=======================================================================
    // Create a start and goal states
    //=======================================================================
    start_state_[0] = double(last_robot_pose_.getOrigin().getX()); // x
    start_state_[1] = double(last_robot_pose_.getOrigin().getY()); // y

    // create a start state
    ob::ScopedState<> start(space);

    start[0] = double(start_state_[0]); // x
    start[1] = double(start_state_[1]); // y

    // create a goal state
    ob::ScopedState<> goal(space);

    goal[0] = double(goal_map_frame_[0]); // x
    goal[1] = double(goal_map_frame_[1]); // y

    // ! 3. Set state validity checker
    //=======================================================================
    // Set state validity checking for this space
    //=======================================================================
    ob::StateValidityCheckerPtr om_stat_val_check;
    om_stat_val_check = ob::StateValidityCheckerPtr(
        new OmFclStateValidityCheckerR2(simple_setup_->getSpaceInformation(),
                                        planning_bounds_x_, planning_bounds_y_));

    // ! 4. set optimization objective
    //=======================================================================
    // Set optimization objective
    //=======================================================================
    simple_setup_->getProblemDefinition()->setOptimizationObjective(ob::OptimizationObjectivePtr(new ob::PathLengthOptimizationObjective(si)));

    // ! 5. Define planner
    //=======================================================================
    // Create a planner for the defined space
    //=======================================================================

    ob::PlannerPtr planner;
    if (planner_name_.compare("RRT") == 0)
        planner = ob::PlannerPtr(new og::RRT(si));
    if (planner_name_.compare("PRMstar") == 0)
        planner = ob::PlannerPtr(new og::PRMstar(si));
    else if (planner_name_.compare("RRTstar") == 0)
        planner = ob::PlannerPtr(new og::RRTstar(si));
    else if (planner_name_.compare("PRM") == 0)
        planner = ob::PlannerPtr(new og::PRM(si));

    // ! 6. Set attributes to simple setup
    //=======================================================================
    // Set the setup planner
    //=======================================================================
    simple_setup_->setPlanner(planner);

    //=======================================================================
    // Set the start and goal states
    //=======================================================================
    simple_setup_->setStartState(start);
    simple_setup_->setGoalState(goal, goal_radius_);
    simple_setup_->setStateValidityChecker(om_stat_val_check);
    simple_setup_->getStateSpace()->setValidSegmentCountFactor(15.0);

    //=======================================================================
    // Perform setup steps for the planner
    //=======================================================================
    simple_setup_->setup();

    // ! 7. attempt to solve
    //=======================================================================
    // Attempt to solve the problem within one second of planning time
    //=======================================================================
    ob::PlannerStatus solved = simple_setup_->solve(solving_time_);

    if (solved && simple_setup_->haveExactSolutionPath())
    {
        // get the goal representation from the problem definition (not the same as the goal state)
        // and inquire about the found path

        og::PathGeometric path = simple_setup_->getSolutionPath();

        // generates varios little segments for the waypoints obtained from the planner
        path.interpolate(int(path.length() / 0.2));

        // path_planning_msgs::PathConstSpeed solution_path;
        ROS_INFO("%s:\n\tpath with cost %f has been found with simple_setup\n",
                 ros::this_node::getName().c_str(),
                 path.cost(simple_setup_->getProblemDefinition()->getOptimizationObjective()).value());

        std::vector<ob::State *> path_states;
        path_states = path.getStates();

        double distance_to_goal =
            sqrt(pow(goal_odom_frame_[0] - path_states[path_states.size() - 1]
                                               ->as<ob::RealVectorStateSpace::StateType>()
                                               ->values[0],
                     2.0) +
                 pow(goal_odom_frame_[1] - path_states[path_states.size() - 1]
                                               ->as<ob::RealVectorStateSpace::StateType>()
                                               ->values[1],
                     2.0));

        if (simple_setup_->haveExactSolutionPath() || distance_to_goal <= goal_radius_)
        {
            // path.interpolate(int(path.length() / 1.0));
            visualizeRRT(path);

            //=======================================================================
            // Controller
            //=======================================================================
            if (path_states.size() > 0)
            {
                nav_msgs::Path solution_path_for_control;
                solution_path_for_control.header.frame_id = world_frame_;
                for (unsigned int i = 0; i < path_states.size(); i++)
                {
                    geometry_msgs::PoseStamped p;
                    p.pose.position.x = path_states[i]->as<ob::RealVectorStateSpace::StateType>()->values[0];
                    p.pose.position.y = path_states[i]->as<ob::RealVectorStateSpace::StateType>()->values[1];

                    if (i == (path_states.size() - 1))
                    {
                        if (goal_available_)
                        {
                            ros::Time t;
                            std::string err = "";
                            tf::StampedTransform tf_map_to_fixed;
                            tf_listener_.getLatestCommonTime("map", "odom", t, &err);
                            tf_listener_.lookupTransform("map", "odom", t, tf_map_to_fixed);

                            tf_map_to_fixed.getBasis().getEulerYPR(yaw, useless_pitch, useless_roll);

                            tf2::Quaternion myQuaternion;

                            myQuaternion.setRPY(useless_roll, useless_pitch, goal_map_frame_[2] - yaw);

                            myQuaternion = myQuaternion.normalize();

                            p.pose.orientation.x = myQuaternion.getX();
                            p.pose.orientation.y = myQuaternion.getY();
                            p.pose.orientation.z = myQuaternion.getZ();
                            p.pose.orientation.w = myQuaternion.getW();
                        }
                    }
                    solution_path_for_control.poses.push_back(p);
                }
                solution_path_pub_.publish(solution_path_for_control);
            }
            //========================
            // SAVE THE PATH
            // =======================
            solution_path_states_.clear();
            for (int i = 0; i < path_states.size(); i++)
            {
                ob::State *s = space->allocState();
                space->copyState(s, path_states[i]);
                solution_path_states_.push_back(s);
            }
        }
        //=======================================================================
        // Clear previous solution path
        //=======================================================================
        simple_setup_->clear();
    }
    else
    {
        ROS_INFO("%s:\n\tpath has not been found\n", ros::this_node::getName().c_str());
    }
}

//! Resulting path visualization.
/*!
 * Visualize resulting path.
 */
void PlannFramework::visualizeRRT(og::PathGeometric &geopath)
{
    // %Tag(MARKER_INIT)%
    tf::Quaternion orien_quat;
    visualization_msgs::Marker visual_rrt, visual_result_path;
    visual_result_path.header.frame_id = visual_rrt.header.frame_id = world_frame_;
    visual_result_path.header.stamp = visual_rrt.header.stamp = ros::Time::now();
    visual_rrt.ns = "planner_rrt";
    visual_result_path.ns = "planner_result_path";
    visual_result_path.action = visual_rrt.action = visualization_msgs::Marker::ADD;

    visual_result_path.pose.orientation.w = visual_rrt.pose.orientation.w = 1.0;
    // %EndTag(MARKER_INIT)%

    // %Tag(ID)%
    visual_rrt.id = 0;
    visual_result_path.id = 1;
    // %EndTag(ID)%

    // %Tag(TYPE)%
    visual_rrt.type = visual_result_path.type = visualization_msgs::Marker::LINE_LIST;
    // %EndTag(TYPE)%

    // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
    visual_rrt.scale.x = 0.03;
    visual_result_path.scale.x = 0.08;
    // %EndTag(SCALE)%

    // %Tag(COLOR)%
    // Points are green
    visual_result_path.color.g = 1.0;
    visual_result_path.color.a = 1.0;

    // Line strip is blue
    visual_rrt.color.b = 1.0;
    visual_rrt.color.a = 1.0;

    const ob::RealVectorStateSpace::StateType *state_r2;

    geometry_msgs::Point p;

    ob::PlannerData planner_data(simple_setup_->getSpaceInformation());
    simple_setup_->getPlannerData(planner_data);

    std::vector<unsigned int> edgeList;
    int num_parents;
    ROS_DEBUG("%s: number of states in the tree: %d", ros::this_node::getName().c_str(),
              planner_data.numVertices());

    if (visualize_tree_)
    {
        for (unsigned int i = 1; i < planner_data.numVertices(); ++i)
        {
            if (planner_data.getVertex(i).getState() && planner_data.getIncomingEdges(i, edgeList) > 0)
            {
                state_r2 = planner_data.getVertex(i).getState()->as<ob::RealVectorStateSpace::StateType>();
                p.x = state_r2->values[0];
                p.y = state_r2->values[1];
                p.z = 0.1;

                visual_rrt.points.push_back(p);

                state_r2 =
                    planner_data.getVertex(edgeList[0]).getState()->as<ob::RealVectorStateSpace::StateType>();
                p.x = state_r2->values[0];
                p.y = state_r2->values[1];
                p.z = 0.1;

                visual_rrt.points.push_back(p);
            }
        }
        solution_path_rviz_pub_.publish(visual_rrt);
    }

    std::vector<ob::State *> states = geopath.getStates();
    for (uint32_t i = 0; i < geopath.getStateCount(); ++i)
    {
        // extract the component of the state and cast it to what we expect

        state_r2 = states[i]->as<ob::RealVectorStateSpace::StateType>();
        p.x = state_r2->values[0];
        p.y = state_r2->values[1];
        p.z = 0.1;

        if (i > 0)
        {
            visual_result_path.points.push_back(p);

            state_r2 = states[i - 1]->as<ob::RealVectorStateSpace::StateType>();
            p.x = state_r2->values[0];
            p.y = state_r2->values[1];
            p.z = 0.1;

            visual_result_path.points.push_back(p);
        }
    }
    solution_path_rviz_pub_.publish(visual_result_path);
}

//! Main function
int main(int argc, char **argv)
{
    ros::init(argc, argv, "ompl_planning_demo");

    ROS_INFO("%s:\n\tplanner (C++), using OMPL version %s\n", ros::this_node::getName().c_str(),
             OMPL_VERSION);

    PlannFramework planning_framework;
    planning_framework.run();
    ros::spin();
    return 0;
}
