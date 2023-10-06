# SLAM-evaluation-tool
Performance evaluation for SLAM

Trajectory file and GT file should be in the TUM format
eval_ExtractImage calculates the absolute relative pose between each estimated pose from its associated GT. Then it outputs the pose with the maximum error and saves the images at the time where maximum error occurs.  
