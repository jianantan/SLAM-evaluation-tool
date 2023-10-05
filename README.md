# SLAM-evaluation-tool
Performance evaluation for SLAM

Trajectory file and GT file should be in the TUM format
eval_ExtractImage calculates the Euclidean distance of x and y of each estimated pose in the Trajectory file with its corresponding GT. Then it outputs the pose with the maximum error and saves the images at the time where maximum error occurs.  
