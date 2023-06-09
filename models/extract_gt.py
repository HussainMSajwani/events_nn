#!/home/hussain/miniconda3/envs/tactile/bin/python
import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    print("Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for j, (topic, msg, t) in enumerate(bag.read_messages(topics=[args.image_topic])):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if not cv_img.shape == (260, 346, 3):
            print(cv_img.shape, "frame_%i.png" % t.to_nsec())
            continue
        cv2.imwrite(os.path.join(args.output_dir, "frame_%i.png" % t.to_nsec()), cv_img)
        #print("Wrote image %i" % count)

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()