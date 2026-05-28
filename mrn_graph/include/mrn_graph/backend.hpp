#pragma once

#include <memory>
#include <string>
#include <vector>

#include <mrn_msgs/msg/agent_state.hpp>
#include <mrn_msgs/msg/clock_offset_estimate.hpp>
#include <mrn_msgs/msg/comm_status.hpp>
#include <mrn_msgs/msg/constraint_graph.hpp>
#include <mrn_msgs/msg/cooperative_pose.hpp>
#include <mrn_msgs/msg/relative_pose_constraint.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace mrn_graph
{

struct GraphInputs
{
  std::vector<mrn_msgs::msg::AgentState> agent_states;
  std::vector<mrn_msgs::msg::RelativePoseConstraint> constraints;
  std::vector<mrn_msgs::msg::ClockOffsetEstimate> clock_offsets;
  std::vector<mrn_msgs::msg::CommStatus> comm_status;
};

struct GraphOutputs
{
  std::vector<mrn_msgs::msg::CooperativePose> cooperative_poses;
  std::vector<nav_msgs::msg::Odometry> cooperative_odoms;
  mrn_msgs::msg::ConstraintGraph status;
  visualization_msgs::msg::MarkerArray markers;
};

class Backend
{
public:
  virtual ~Backend() = default;

  virtual std::string name() const = 0;

  virtual void configure(rclcpp::Node & node) = 0;

  virtual GraphOutputs step(const GraphInputs & inputs, const rclcpp::Time & stamp) = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

}  // namespace mrn_graph
