import sys 
import os
import argparse
import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pandas as pd
import numpy as np
import math
import warnings
from tqdm import tqdm
from tf.transformations import quaternion_matrix
from tf import transformations

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Location of image to be saved.")
    parser.add_argument("gt_file", help="GT of the trajectory")
    parser.add_argument("tj_file", help="Localized trajectory")

    args = parser.parse_args()
    gt_Df = pd.read_csv(args.gt_file, delimiter=' ', names=['Timestamp', 't_x', 't_y', 't_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w'], header=None)
    tj_Df = pd.read_csv(args.tj_file, delimiter=' ', names=['Timestamp', 't_x', 't_y', 't_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w', 'Timestamp_ref','t_x_ref', 't_y_ref', 't_z_ref', 'rot_x_ref', 'rot_y_ref', 'rot_z_ref', 'rot_w_ref', 'error'], header=None)
    nTracked_Df = pd.read_csv(args.tj_file[:len(args.tj_file)-4] + "_nTracked.txt", delimiter=' ', names=['Timestamp', 'nTracked'], header=None)
    frame_rate = 30.0
    max_diff = 0.01
    max_error = 0
    max_error_i = 0
    
    gt_i = 0
    print("Calculating absolute relative pose between each estimated pose from its associated GT...")
    for index, row in tqdm(tj_Df.iterrows(), total=tj_Df.shape[0]):
        timestamp = row['Timestamp']

        Pe_wc = quaternion_matrix(np.array([row['rot_x'], row['rot_y'], row['rot_z'], row['rot_w']]))
        Pe_wc[0,3] = row['t_x']
        Pe_wc[1,3] = row['t_y']
        Pe_wc[2,3] = row['t_z']

        gt_i = (gt_Df['Timestamp'] - timestamp).abs().argsort()[:1].values[0]
        '''
        Check difference between timstamp of est_pose and ref_pose
        If the timestamps are too far apart, i.e. greater than max_diff
        The two poses are considered non-associative and the est_pose is discarded for error calculation
        '''
        if (abs(gt_Df.loc[gt_i, 'Timestamp'] - timestamp)) < max_diff:
            t_x_ref = gt_Df.loc[gt_i, 't_x']
            t_y_ref = gt_Df.loc[gt_i, 't_y']
            t_z_ref = gt_Df.loc[gt_i, 't_z']
            r_x_ref = gt_Df.loc[gt_i, 'rot_x']
            r_y_ref = gt_Df.loc[gt_i, 'rot_y']
            r_z_ref = gt_Df.loc[gt_i, 'rot_z']
            r_w_ref = gt_Df.loc[gt_i, 'rot_w']

            Pr_wc = quaternion_matrix(np.array([r_x_ref, r_y_ref, r_z_ref, r_w_ref]))
            Pr_wc[0,3] = t_x_ref
            Pr_wc[1,3] = t_y_ref
            Pr_wc[2,3] = t_z_ref
            Pr_cw = transformations.inverse_matrix(Pr_wc)

            E = np.dot(Pr_cw, Pe_wc)
            row['error'] = math.sqrt((E[0,3])**2 + (E[1,3])**2 + (E[2,3])**2)
            row['Timestamp_ref'] = gt_Df.loc[gt_i, 'Timestamp']
            row['t_x_ref'] = t_x_ref
            row['t_y_ref'] = t_y_ref
            row['t_z_ref'] = t_z_ref
            row['rot_x_ref'] = r_x_ref
            row['rot_y_ref'] = r_y_ref
            row['rot_z_ref'] = r_z_ref
            row['rot_w_ref'] = r_w_ref

            if row['error'] > max_error:
                max_error = row['error']
                max_error_i = index
                max_error_x = t_x_ref
                max_error_y = t_y_ref
                max_error_z = t_z_ref
                max_error_timestamp = timestamp

    nTracked_min = nTracked_Df["nTracked"].min()
    nTracked_max = nTracked_Df["nTracked"].max()
    nTracked_mean = nTracked_Df["nTracked"].mean()
    nTracked_median = nTracked_Df["nTracked"].median()

    max_error_statement = "Max error occurred at: "+str(max_error_timestamp)+" ["+str(max_error_x)+ " "+ str(max_error_y)+ " " + str(max_error_z) +"]:  "
    print("max\t: "+str(tj_Df.loc[max_error_i, 'error']))
    print("mean\t: "+str(tj_Df['error'].mean()))
    print("median\t: "+str(tj_Df['error'].median()))
    print("min\t: "+str(tj_Df['error'].min()))
    print("rmse\t: "+str(math.sqrt(tj_Df['error'].pow(2).sum()/tj_Df['error'].count())))
    print("std\t: "+str(tj_Df['error'].std()))

    isExist = os.path.exists(args.output_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(args.output_dir)
        print(str(args.output_dir) + " created!")

    print(max_error_statement)
    with open(os.path.join(args.output_dir, 'max_error.txt'), 'w') as f:
        f.write(max_error_statement + "\n")
        f.write("max_error\n")
        f.write("mean_error\n")
        f.write("median_error\n")
        f.write("min_error\n")
        f.write("rmse\n")
        f.write("std\n")
        f.write(str(tj_Df.loc[max_error_i, 'error'])+"\n")
        f.write(str(tj_Df['error'].mean())+"\n")
        f.write(str(tj_Df['error'].median())+"\n")
        f.write(str(tj_Df['error'].min())+"\n")
        f.write(str(math.sqrt(tj_Df['error'].pow(2).sum()/tj_Df['error'].count()))+"\n")
        f.write(str(tj_Df['error'].std())+"\n")
        f.write("\nnTracked Map Points: "+ "\n")
        f.write("\tmin\t\t: "+str(nTracked_min)+ "\n")
        f.write("\tmean\t\t: "+str(nTracked_mean)+ "\n")
        f.write("\tmedian\t\t: "+str(nTracked_median)+ "\n")
        f.write("\tmax\t\t: "+str(nTracked_max)+ "\n")
        f.write("\tnTracked at max_error: "+str(nTracked_Df.loc[max_error_i, 'nTracked'])+ "\n")

    tj_Df.to_csv(os.path.join(args.output_dir, 'trajectory_gt_synced.txt'), sep='\t', index=False)

    time_of_interest = tj_Df.loc[max_error_i, 'Timestamp']

    print("Extracting image...")
    bridge = CvBridge()
    for filename in os.scandir(args.bag_file):
        if filename.is_file():
            for topic, msg, t in rosbag.Bag(filename.path).read_messages(topics="/camera/color/image_raw", start_time=rospy.Time(time_of_interest - 5/frame_rate), end_time=rospy.Time(time_of_interest + 5/frame_rate)):
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) 
                print("Saving ", os.path.join(args.output_dir, "%06i.png" % msg.header.seq))
                cv2.imwrite(os.path.join(args.output_dir, "%06i.png" % msg.header.seq), cv_img)
                
if __name__ == '__main__':
    main()