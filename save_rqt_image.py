import rospy
from sensor_msgs.msg import Image
import cv2
import sys
from cv_bridge import CvBridge, CvBridgeError
import glob

# def files_in_folder(folder_path):


def main(args):
    rospy.init_node("save_rqt_image", anonymous=True)
    data = rospy.wait_for_message("/R1/pi_camera/image_raw",Image)
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    # add unique identifier
    num_imgs = len(glob.glob(args[1] +"_*"+".png"))
    print("num_imgs: {}".format(num_imgs))
    filename = "{}_{}.png".format(args[1], num_imgs+1)
    cv2.imwrite(filename, cv_image)
    print("Saved image to: " + filename)


if __name__ == '__main__':
    main(sys.argv)
