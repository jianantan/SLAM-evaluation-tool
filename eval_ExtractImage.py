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

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Location of image to be saved.")
    parser.add_argument("gt_file", help="GT of the trajectory")
    parser.add_argument("tj_file", help="Localized trajectory")

    args = parser.parse_args()
    gt_Df = pd.read_csv(args.gt_file, delimiter=' ', names=['Timestamp', 't_x', 't_y', 't_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w'], header=None)
    tj_Df = pd.read_csv(args.tj_file, delimiter=' ', names=['Timestamp', 't_x', 't_y', 't_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w', 'dist'], header=None)
    frame_rate = 30.0
    #gt_file=str(sys.argv[1])
    #tj_file=str(sys.argv[2])
    max_error = 0
    max_error_i = 0
    
    gt_i = 0
    inSync = False
    print("Calculating Euclidean distance of each estimated pose from its corresponding GT...")
    for index, row in tqdm(tj_Df.iterrows(), total=tj_Df.shape[0]):
        timestamp = row['Timestamp']
        t_x_est = row['t_x']
        t_y_est = row['t_y']
        t_x_ref = 0.0
        t_y_ref = 0.0
        '''
        last_gt_i = gt_i
        while gt_i >= 0 and gt_i < (len(gt_Df.index) - 2):
            if timestamp >= gt_Df.loc[gt_i, 'Timestamp'] and timestamp < gt_Df.loc[gt_i + 1, 'Timestamp']:
                inSynce = True
                break
            elif timestamp >= gt_Df.loc[gt_i + 1, 'Timestamp']:
                gt_i += 1
            elif timestamp < gt_Df.loc[gt_i, 'Timestamp']:
                gt_i -= 1
        
        if inSync is True:
            
            if abs(timestamp - gt_Df.loc[gt_i, 'Timestamp']) < abs(timestamp - gt_Df.loc[gt_i + 1, 'Timestamp']):
                if abs(timestamp - gt_Df.loc[gt_i, 'Timestamp']) < 1.0/frame_rate:
                    t_x_ref = gt_Df.loc[gt_i, 't_x']
                    t_y_ref = gt_Df.loc[gt_i, 't_y']
            else:
                if abs(timestamp - gt_Df.loc[gt_i + 1, 'Timestamp']) < 1.0/frame_rate:
                    t_x_ref = gt_Df.loc[gt_i + 1, 't_x']
                    t_y_ref = gt_Df.loc[gt_i + 1, 't_y']
        else:
            gt_i = last_gt_i
        '''
        gt_i = (gt_Df['Timestamp'] - timestamp).abs().argsort()[:1].values[0]
        t_x_ref = gt_Df.loc[gt_i, 't_x']
        t_y_ref = gt_Df.loc[gt_i, 't_y']
        row['dist'] = math.sqrt((t_x_est - t_x_ref)**2 + (t_y_est - t_y_ref)**2)
        if row['dist'] > max_error:
            max_error = row['dist']
            max_error_i = index
            max_error_x = t_x_ref
            max_error_y = t_y_ref

    max_error_statement = "Max error occurred at: ["+str(max_error_x)+ " "+ str(max_error_y)+ "]"
    print(max_error_statement)
    with open(os.path.join(args.output_dir, 'max_error.txt'), 'w') as f:
        f.write(max_error_statement)

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