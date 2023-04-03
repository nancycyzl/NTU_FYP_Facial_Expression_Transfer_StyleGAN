'''
This script will detect, crop and align human face in the image. Result image will be resized to 1024x1024.

Usage: python face_alignment.py input_path output_path --enable_smooth

input_path: an image or a folder of images
output_path: an image filename or a folder
enable_smooth: only applicable for a folder of images, use coordinates of the first frame to process all frames
'''


import numpy as np
import argparse
import dlib
import tqdm
import cv2
import sys
import os


def parse_args():
    parse = argparse.ArgumentParser(description='face alignment for images')
    parse.add_argument('input_path', type=str, help='input: an image or a folder of images')
    parse.add_argument('output_path', type=str, help='output: an image file or a folder')
    parse.add_argument('--enable_smooth', action='store_true', help='use coordinates of the first image to process all frames')
    args = parse.parse_args()
    return args


def shape_to_np(shape, dtype='int'):
    coords = np.zeros((68,2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_eye(lm):
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_left_right = [sum(x) for x in zip(mouth_left, mouth_right)]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    return eye_left, eye_right, mouth_avg


def get_rotation_matrix(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    angle = np.degrees(np.arctan(tan))
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)

    return M


def get_crop_rect(left_eye, right_eye, mouth):
    eye_to_eye = right_eye - left_eye
    eye_center = (right_eye + left_eye) * 0.5
    eye_to_mouth = mouth - eye_center
    center = eye_center + eye_to_mouth * 0.1
    cx = center[0]
    cy = center[1]

    side = max(np.hypot(*eye_to_eye)*3, np.hypot(*eye_to_mouth)*3.5)

    top_left = (cx-side/2, cy-side/2)
    bot_right = (cx+side/2, cy+side/2)

    return top_left, bot_right


def crop_image(img, rect):
    left = int(rect[0][0])
    right = int(rect[1][0])
    top = int(rect[0][1])
    bottom = int(rect[1][1])

    # print('width: ', right-left)
    # print('height: ',bottom-top)

    cropped = img[top:bottom, left:right]
    return cropped


def process_one_image(img):
    face_rec = detector(img)[0]  # rectangle of the face
    face_shape = predictor(img, face_rec)
    lm = shape_to_np(face_shape)

    left_eye, right_eye, mouth = get_eye(lm)
    M = get_rotation_matrix(left_eye, right_eye)
    crop_rect = get_crop_rect(left_eye, right_eye, mouth)

    return M, crop_rect


def check_paths(input, output):
    if not os.path.exists(input):
        print(f'input path {input} does not exist!!')
        sys.exit()

    if not output.endswith(('.jpg', '.png')):
        if not os.path.exists(output):
            os.mkdir(output)
            print('Creating output folder')
        else:
            print(f'output path {output} already exists!')
            sys.exit()


def process(input_path, output_path, enable_smooth):
    # if is image: process this image
    # if is folder & enable_smooth: process the first frame to get the coordinates
    if os.path.isfile(input_path) or enable_smooth:
        if os.path.isfile(input_path):
            img_file = input_path
        else:
            img_file = os.path.join(input_path,os.listdir(input_path)[0])  # get the first frame

        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        assert img is not None, 'Fails to read image: ' + os.path.join(os.getcwd(), img_file)
        M, crop_rect = process_one_image(img)

        if os.path.isfile(input_path):   # if input is image
            img_rgb = cv2.imread(input_path)
            rotated = cv2.warpAffine(img_rgb, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)  # here :))
            # print('After rotation, shape is', rotated.shape)
            cropped = crop_image(rotated, crop_rect)

            resized = cv2.resize(cropped, (img_size, img_size))
            cv2.imwrite(output_path, resized)

    if os.path.isdir(input_path):    # when input is a folder
        for img_filename in tqdm.tqdm(os.listdir(input_path)):
            # print(img_filename)
            img_path = os.path.join(input_path,img_filename)
            out_path = os.path.join(output_path,img_filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if not enable_smooth:
                M, crop_rect = process_one_image(img)

            img_rgb = cv2.imread(img_path)
            rotated = cv2.warpAffine(img_rgb, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_CUBIC)   # here :))
            # print('After rotation, shape is', rotated.shape)
            cropped = crop_image(rotated, crop_rect)

            if abs(cropped.shape[0]-cropped.shape[1]) > 5:
                print('WARNING! The result image might be distorted!')
                print('shape: ', cropped.shape)
            resized = cv2.resize(cropped,(img_size,img_size))
            cv2.imwrite(out_path, resized)


if __name__ == '__main__':
    args = parse_args()
    check_paths(args.input_path, args.output_path)

    img_size = 1024
    detector_path = 'resources/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(detector_path)

    process(args.input_path, args.output_path, args.enable_smooth)
    print('Done! Result saved at:', args.output_path)
