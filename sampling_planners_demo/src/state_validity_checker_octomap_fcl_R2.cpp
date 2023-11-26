/*! \file state_validity_checker_octomap_fcl_R2.cpp
 * \brief State validity checker.
 *
 * \date March 5, 2015
 * \author Juan David Hernandez Vega, juandhv@rice.edu
 *
 * \details Check is a given configuration R2 is collision-free.
 *  The workspace is represented by an Octomap and collision check is done with FCL.
 *
 * Based on Juan D. Hernandez Vega's PhD thesis, University of Girona
 * http://hdl.handle.net/10803/457592, http://www.tdx.cat/handle/10803/457592
 */

#include <state_validity_checker_octomap_fcl_R2.h>

OmFclStateValidityCheckerR2::OmFclStateValidityCheckerR2(const ob::SpaceInformationPtr &si,
                                                         std::vector<double> planning_bounds_x,
                                                         std::vector<double> planning_bounds_y)
    : ob::StateValidityChecker(si), local_nh_("~"), robot_base_radius_(0.4), robot_base_height_(2.0)
{
    GetOctomap::Request req;
    GetOctomap::Response resp;

    planning_bounds_x_ = planning_bounds_x;
    planning_bounds_y_ = planning_bounds_y;

    local_nh_.param("robot_base_radius", robot_base_radius_, robot_base_radius_);
    local_nh_.param("robot_base_height", robot_base_height_, robot_base_height_);
    local_nh_.param("octomap_service", octomap_service_, octomap_service_);
    local_nh_.param("odometry_topic", odometry_topic, odometry_topic);
    local_nh_.param("main_frame", main_frame, main_frame);

    octree_ = NULL;

    ROS_DEBUG("%s: requesting the map to %s...", ros::this_node::getName().c_str(),
              nh_.resolveName(octomap_service_).c_str());

    while ((nh_.ok() && !ros::service::call(octomap_service_, req, resp)) || resp.map.data.size() == 0)
    {
        ROS_WARN("Request to %s failed; trying again...", nh_.resolveName(octomap_service_).c_str());
        usleep(1000000);
    }
    if (nh_.ok())
    { // skip when CTRL-C
        abs_octree_ = octomap_msgs::msgToMap(resp.map);
        std::cout << std::endl;
        if (abs_octree_)
        {
            octree_ = dynamic_cast<octomap::OcTree *>(abs_octree_);
            tree_ = new fcl::OcTreef(std::shared_ptr<const octomap::OcTree>(octree_));
            tree_obj_ = new fcl::CollisionObjectf((std::shared_ptr<fcl::CollisionGeometryf>(tree_)));
        }

        robot_collision_solid_.reset(new fcl::Cylinderf(robot_base_radius_, robot_base_height_));

        octree_res_ = octree_->getResolution();
        octree_->getMetricMin(octree_min_x_, octree_min_y_, octree_min_z_);
        octree_->getMetricMax(octree_max_x_, octree_max_y_, octree_max_z_);

        if (octree_)
            ROS_DEBUG("%s: Octomap received (%zu nodes, %f m res)", ros::this_node::getName().c_str(),
                      octree_->size(), octree_->getResolution());
        else
            ROS_ERROR("Error reading OcTree from stream");
    }

    ROS_INFO_STREAM("Retrieving robot odometry.");
    odomData = ros::topic::waitForMessage<nav_msgs::Odometry>(odometry_topic);
}

bool OmFclStateValidityCheckerR2::isValid(const ob::State *state) const
{
    const ob::RealVectorStateSpace::StateType *state_r2 = state->as<ob::RealVectorStateSpace::StateType>();

    if (state_r2->values[0] < octree_min_x_ || state_r2->values[1] < octree_min_y_ ||
        state_r2->values[0] > octree_max_x_ || state_r2->values[1] > octree_max_y_)
    {
        // ompl::tools::Profiler::End("collision");
        return true;
    }
    // FCL
    fcl::Transform3f robot_tf;
    robot_tf.setIdentity();
    robot_tf.translate(fcl::Vector3f(state_r2->values[0], state_r2->values[1], robot_base_height_ / 2.0));

    fcl::CollisionObjectf vehicle_co(robot_collision_solid_, robot_tf);

    fcl::CollisionRequestf collision_request;
    fcl::CollisionResultf collision_result;

    fcl::collide(tree_obj_, &vehicle_co, collision_request, collision_result);

    if (collision_result.isCollision())
    {
        return false;
    }

    return true;
}

bool OmFclStateValidityCheckerR2::isValidPoint(const ob::State *state) const
{
    OcTreeNode *result;
    point3d query;
    double node_occupancy;

    // extract the component of the state and cast it to what we expect
    const ob::RealVectorStateSpace::StateType *state_r2 = state->as<ob::RealVectorStateSpace::StateType>();

    query.x() = state_r2->values[0];
    query.y() = state_r2->values[1];
    query.z() = 0.0;

    result = octree_->search(query);

    if (result == NULL)
    {
        return false;
    }
    else
    {
        node_occupancy = result->getOccupancy();
        if (node_occupancy <= 0.4)
            return true;
    }
    return false;
}

OmFclStateValidityCheckerR2::~OmFclStateValidityCheckerR2()
{
    delete octree_;
    //    delete tree_;
    //    delete tree_obj_;
}
