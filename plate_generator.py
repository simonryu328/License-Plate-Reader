#!/usr/bin/env python
import sys
import cv2
import csv
import numpy as np
import os
import pyqrcode
import random
import string
import glob

from random import randint
from PIL import Image, ImageFont, ImageDraw

path = os.path.dirname(os.path.realpath(__file__)) + "/"
outpath = 'generated_data/'

def segment_characters(img, label):
    ytop1 = 0
    ytop2 = 590
    dy1 = 460
    dx1 = 300
    dy2 = 230
    dx2 = 135
    target_width = 37
    target_height = 27
    dim = (target_width, target_height)

    pix_map = np.array([[[ytop1,15],[ytop1+dy1,15+dx1]],
                        [[ytop1,270],[ytop1+dy1,270+dx1]],
                        [[ytop2,10],[ytop2+dy2,10+dx2]],
                        [[ytop2,130],[ytop2+dy2,130+dx2]],
                        [[ytop2,320],[ytop2+dy2,320+dx2]],
                        [[ytop2,430],[ytop2+dy2,430+dx2]]])

    char_imgs = []
    for i in range(len(pix_map)):
        top = pix_map[i,0]
        bot = pix_map[i,1]
        char_img = img[top[0]:bot[0],top[1]:bot[1]]

        # Resize Parking ID imgs to have same size as plate characters
        # cv2.imshow("char_img",char_img)
        # cv2.waitKey(0)
        char_img = cv2.resize(char_img, dim, interpolation =cv2.INTER_AREA)
        char_imgs.append(char_img)
    return char_imgs
def generate_plate_data(args):
    for i in range(int(args[1])):

        # Pick two random letters
        plate_alpha = ""
        for _ in range(0, 2):
            plate_alpha += (random.choice(string.ascii_uppercase))
        num = randint(0, 99)

        # Pick two random numbers
        plate_num = "{:02d}".format(num)

        # # Save plate to file
        # csvwriter.writerow([plate_alpha+plate_num])

        # Write plate to image
        blank_plate = cv2.imread(path+'blank_plate.png')

        # To use monospaced font for the license plate we need to use the PIL
        # package.
        # Convert into a PIL image (this is so we can use the monospaced fonts)
        blank_plate_pil = Image.fromarray(blank_plate)
        # Get a drawing context
        draw = ImageDraw.Draw(blank_plate_pil)
        monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 200)
        draw.text((48, 105),plate_alpha + " " + plate_num, (255,0,0), font=monospace)
        # Convert back to OpenCV image and save
        blank_plate = np.array(blank_plate_pil)

        # cv2.putText(blank_plate,
        #             plate_alpha + " " + plate_num, (45, 360),
        #             cv2.FONT_HERSHEY_PLAIN, 11, (255, 0, 0), 7, cv2.LINE_AA)

        # Create QR code image
        # spot_name = "P" + str(i)
        # qr = pyqrcode.create(spot_name+"_" + plate_alpha + plate_num)
        # qrname = path + "QRCode_" + str(i) + ".png"
        # qr.png(qrname, scale=20)
        # QR_img = cv2.imread(qrname)
        # QR_img = cv2.resize(QR_img, (600, 600), interpolation=cv2.INTER_AREA)

        # Create parking spot label
        num = randint(1, 8)
        s = "P" + str(num)
        parking_spot = 255 * np.ones(shape=[600, 600, 3], dtype=np.uint8)
        cv2.putText(parking_spot, s, (30, 450), cv2.FONT_HERSHEY_PLAIN, 28,
                    (0, 0, 0), 30, cv2.LINE_AA)
        spot_w_plate = np.concatenate((parking_spot, blank_plate), axis=0)
        labels = s + plate_alpha+str(plate_num)
        # Merge labelled or unlabelled images and save
        # labelled = np.concatenate((QR_img, spot_w_plate), axis=0)
        unlabelled = np.concatenate((255 * np.ones(shape=[600, 600, 3],
                                    dtype=np.uint8), spot_w_plate), axis=0)
        unlabelled = unlabelled[700:,:,:]

        # grayscale
        img_gray = cv2.cvtColor(unlabelled, cv2.COLOR_BGR2GRAY)
        threshold = 100
        _, img_bin = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

        # cv2.imshow("unlabelled",img_bin)
        # cv2.waitKey(0)
        seg_chars = segment_characters(img_bin, labels)

        print("length of labels: {}".format(len(labels)))
        print(labels)

        for i in range(1,6):
            print(i)
            char_img = seg_chars[i]
            label = labels[i]

            if i == 1:
                path2 = path+outpath+"segmented_parkingID/"+label +"_*"+".png"
            else:
                path2 = path+outpath+"segmented_plate/"+label +"_*"+".png"

            num_imgs = len(glob.glob(path2))
            print("num_imgs: {}".format(num_imgs))
            filename = path2.replace("*",str(num_imgs+1))
            cv2.imwrite(filename, char_img)
            print("Saved image to: " + filename)

if __name__ == '__main__':
    generate_plate_data(sys.argv)
