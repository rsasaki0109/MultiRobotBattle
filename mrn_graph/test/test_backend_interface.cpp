#include <gtest/gtest.h>

#include <mrn_graph/backend.hpp>

namespace
{

class NoOpBackend : public mrn_graph::Backend
{
public:
  std::string name() const override { return "noop"; }

  void configure(rclcpp::Node & /*node*/) override { configured_ = true; }

  mrn_graph::GraphOutputs step(
    const mrn_graph::GraphInputs & inputs, const rclcpp::Time & stamp) override
  {
    last_stamp_ = stamp;
    last_input_constraint_count_ = inputs.constraints.size();
    mrn_graph::GraphOutputs out;
    out.status.backend_name = name();
    out.status.accepted_constraint_count = 0;
    out.status.rejected_constraint_count = 0;
    out.status.stale_constraint_count = 0;
    return out;
  }

  bool configured_ = false;
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};
  std::size_t last_input_constraint_count_ = 0;
};

}  // namespace

TEST(BackendInterface, ConcreteBackendReportsName)
{
  NoOpBackend backend;
  EXPECT_EQ(backend.name(), "noop");
}

TEST(BackendInterface, StepProducesStatusWithBackendName)
{
  NoOpBackend backend;
  mrn_graph::GraphInputs inputs;
  inputs.constraints.emplace_back();
  inputs.constraints.emplace_back();

  rclcpp::Time stamp(1, 500'000'000, RCL_ROS_TIME);
  const auto outputs = backend.step(inputs, stamp);

  EXPECT_EQ(outputs.status.backend_name, "noop");
  EXPECT_EQ(outputs.status.accepted_constraint_count, 0u);
  EXPECT_EQ(outputs.status.rejected_constraint_count, 0u);
  EXPECT_EQ(outputs.status.stale_constraint_count, 0u);
  EXPECT_TRUE(outputs.cooperative_poses.empty());
  EXPECT_TRUE(outputs.cooperative_odoms.empty());
  EXPECT_TRUE(outputs.markers.markers.empty());
  EXPECT_EQ(backend.last_input_constraint_count_, 2u);
  EXPECT_EQ(backend.last_stamp_.nanoseconds(), stamp.nanoseconds());
}

TEST(BackendInterface, GraphInputsCarryAllStreams)
{
  mrn_graph::GraphInputs inputs;
  inputs.agent_states.emplace_back();
  inputs.clock_offsets.emplace_back();
  inputs.comm_status.emplace_back();

  EXPECT_EQ(inputs.agent_states.size(), 1u);
  EXPECT_EQ(inputs.clock_offsets.size(), 1u);
  EXPECT_EQ(inputs.comm_status.size(), 1u);
  EXPECT_TRUE(inputs.constraints.empty());
}

TEST(BackendInterface, BackendPtrIsSharedOwning)
{
  mrn_graph::BackendPtr backend = std::make_shared<NoOpBackend>();
  EXPECT_EQ(backend->name(), "noop");
  EXPECT_EQ(backend.use_count(), 1);
  auto alias = backend;
  EXPECT_EQ(backend.use_count(), 2);
}
