mrn_msgs/V2VPacketHeader packet

string from_agent_id
string to_agent_id
string from_frame
string to_frame

builtin_interfaces/Time from_state_time
builtin_interfaces/Time to_state_time

geometry_msgs/PoseWithCovariance relative_pose

uint8 SOURCE_FAKE_GROUND_TRUTH=0
uint8 SOURCE_LIDAR_REGISTRATION=1
uint8 SOURCE_VISUAL_MARKER=2
uint8 SOURCE_UWB=3
uint8 SOURCE_V2V_GNSS=4
uint8 source_type

float32 confidence
float32 registration_score
