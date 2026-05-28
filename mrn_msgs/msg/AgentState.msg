mrn_msgs/V2VPacketHeader packet

string agent_id
string map_frame
string odom_frame
string base_frame

geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist

uint8 STATUS_OK=0
uint8 STATUS_DEGRADED=1
uint8 STATUS_STALE=2
uint8 STATUS_INVALID=3
uint8 status

float32 quality
