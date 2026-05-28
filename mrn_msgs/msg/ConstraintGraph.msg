std_msgs/Header header

string graph_id
string[] agent_ids
geometry_msgs/PoseWithCovariance[] agent_poses
mrn_msgs/RelativePoseConstraint[] constraints

uint32 accepted_constraint_count
uint32 rejected_constraint_count
uint32 stale_constraint_count

string[] rejection_reasons
uint32[] rejection_reason_counts
string last_rejection_reason

string backend_name
