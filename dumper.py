#!/usr/bin/python

from __future__ import print_function
import cv2
import glob
import math
import numpy as np
import os
import sys
from scipy import ndimage
import random
from PIL import Image
from PIL import ImageOps
import struct
import argparse

'''
Dump samples for cascade training from images folder or movie file
Markup (regions with objects) can be given in .gt format
Output object sample is specified by 4 coordinates (row, column, width and height; all in pixels)
'''

parser = argparse.ArgumentParser(description='Object samples dumper from pictures to binary uncompressed dumps.')
parser.add_argument('media', help='Images folder or movie file')
parser.add_argument('--gt', default=None, help='Ground truth file (if not specified, write whole image)')
parser.add_argument('--plot', action='store_true', help='Show dumped images')
parser.add_argument('--object', default=None, type=int, help='Object type for dumping, default is all')
parser.add_argument('--out', default='images.dump', help='Output data file (default is images.dump)')
parser.add_argument('--nrands', default=7, type=int,
                    help='Number of variations for each sample (default is 7)')
parser.add_argument('--rotate-jitter', default=True, type=bool,
                    help='Add small rotation to variations')
parser.add_argument('--rotate', default=None, type=int,
                    help='Image rotate angle in degrees (default is 0)')
parser.add_argument('--obj-part', choices=['whole', 'upper'], default='whole',
                    help='Part of object to export (whole object, upper half)')
args = parser.parse_args()

if args.plot:
    import matplotlib.pyplot
    import matplotlib.image
    import matplotlib.cm


def write_rid(im, out_file):
    # raw intensity data

    h = im.shape[0]
    w = im.shape[1]

    hw = struct.pack('ii', h, w)

    tmp = [None] * w * h
    for y in range(0, h):
        for x in range(0, w):
            tmp[y * w + x] = im[y, x]

    pixels = struct.pack('%sB' % w * h, *tmp)

    out_file.write(hw)
    out_file.write(pixels)


def export(im, r, c, w, h, out_file):
    nrows = im.shape[0]
    ncols = im.shape[1]

    # crop
    r0 = max(int(r - 0.75 * h), 0)
    r1 = min(int(r + 0.75 * h), nrows)
    c0 = max(int(c - 0.75 * w), 0)
    c1 = min(int(c + 0.75 * w), ncols)

    im = im[r0:r1, c0:c1]

    nrows = im.shape[0]
    ncols = im.shape[1]

    r -= r0
    c -= c0

    # resize, if needed
    maxwsize = 192.0
    wsize = max(nrows, ncols)

    ratio = maxwsize / wsize

    if ratio < 1.0:
        im = np.asarray(Image.fromarray(im).resize((int(ratio * ncols), int(ratio * nrows))))

        r *= ratio
        c *= ratio
        w *= ratio
        h *= ratio

    lst = []

    for i in range(args.nrands):
        wtmp = w * random.uniform(0.9, 1.1)
        htmp = h * random.uniform(0.9, 1.1)
        rtmp = r + h * random.uniform(-0.05, 0.05)
        ctmp = c + w * random.uniform(-0.05, 0.05)

        if args.plot:
            #cv2.imshow("test", im)
            #cv2.waitKey()
            matplotlib.pyplot.cla()
            matplotlib.pyplot.plot([ctmp - wtmp / 2, ctmp + wtmp / 2], [rtmp - htmp / 2, rtmp - htmp / 2], 'b',
                                   linewidth=3)
            matplotlib.pyplot.plot([ctmp + wtmp / 2, ctmp + wtmp / 2], [rtmp - htmp / 2, rtmp + htmp / 2], 'b',
                                   linewidth=3)
            matplotlib.pyplot.plot([ctmp + wtmp / 2, ctmp - wtmp / 2], [rtmp + htmp / 2, rtmp + htmp / 2], 'b',
                                   linewidth=3)
            matplotlib.pyplot.plot([ctmp - wtmp / 2, ctmp - wtmp / 2], [rtmp + htmp / 2, rtmp - htmp / 2], 'b',
                                   linewidth=3)
            matplotlib.pyplot.imshow(im, cmap=matplotlib.cm.Greys_r)
            matplotlib.pyplot.show()

        lst.append((int(ctmp), int(rtmp), int(wtmp), int(htmp)))

    write_rid(im, out_file)
    out_file.write(struct.pack('i', args.nrands))

    for i in range(args.nrands):
        out_file.write(struct.pack('iiii', lst[i][0], lst[i][1], lst[i][2], lst[i][3]))


def mirror_and_export(im, r, c, w, h, out_file):
    # exploit mirror symmetry of the face

    # flip image
    im = np.asarray(ImageOps.mirror(Image.fromarray(im)))

    # flip column coordinate of the object
    c = im.shape[1] - c

    # export
    export(im, r, c, w, h, out_file)


def rotate_coords(x, y, angle, cx=0, cy=0):
    x -= cx
    y -= cy
    new_x = x * math.cos(angle) + y * math.sin(angle)
    new_y = -x * math.sin(angle) + y * math.cos(angle)
    return new_x + cx, new_y + cy


def whole_rect_from_gt(gt_vals, shape):
    cx = float(gt_vals[1]) / 100.0 * shape[1]
    cy = float(gt_vals[2]) / 100.0 * shape[0]
    w = float(gt_vals[3]) / 100.0 * shape[1]
    h = float(gt_vals[4]) / 100.0 * shape[0]
    return cx, cy, w, h


def upper_rect_from_gt(gt_vals, shape):
    w = float(gt_vals[3]) / 100.0 * shape[1]
    h = float(gt_vals[4]) / 100.0 * shape[0] / 2
    cx = float(gt_vals[1]) / 100.0 * shape[1]
    cy = float(gt_vals[2]) / 100.0 * shape[0] - h / 2
    return cx, cy, w, h


def scan_image_folder(media, gt_file, object_type):
    total = 0
    total_records = 0
    line_no = 0
    if args.rotate_jitter:
        angles = [0, -5, 5]
    else:
        angles = [0]
    gt_lines = [line.strip() for line in open(gt_file, 'r').readlines() if line]
    for line in gt_lines:
        line_no += 1
        # construct full image path
        gt_vals = line.split()
        path = media + '/' + gt_vals[0].strip()

        print('%s (%d/%d)\r' % (path, line_no, len(gt_lines)), end="")
        sys.stdout.flush()
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("can't open image '" + path + "'")
            continue

        # cv2.imshow("test1", image)
        # cv2.waitKey()

        if args.obj_part == 'whole':
            cx, cy, w, h = whole_rect_from_gt(gt_vals, image.shape)
        else:
            cx, cy, w, h = upper_rect_from_gt(gt_vals, image.shape)
        for angle in angles:
            if angle == 0:
                image_cur = image
                cx_cur = cx
                cy_cur = cy
            else:
                image_cur = ndimage.interpolation.rotate(image, angle, reshape=False)
                angle = math.pi * angle / 180
                cx_cur, cy_cur = rotate_coords(cx, cy, angle, image.shape[1] / 2, image.shape[0] / 2)

            total_records += 1

            if cx_cur - w / 2 < 0 or cx_cur + w / 2 >= image.shape[1] or \
                    cy_cur - h / 2 < 0 or cy_cur + h / 2 >= image.shape[0]:
                continue

            if object_type is None or object_type == int(gt_vals[-1]):
                export(image_cur, cy_cur, cx_cur, w, h, out)
                # faces are symmetric and we exploit this here
                mirror_and_export(image_cur, cy_cur, cx_cur, w, h, out)
                total += 1
    return total, total_records


def scan_movie(media_path, gt_file, object_type):
    gt_lines = [line.strip() for line in open(gt_file, 'r').readlines() if line]
    gt_in_frame = {}
    for line in gt_lines:
        gt_vals = line.split()
        frame_num = int(gt_vals[0])
        if frame_num in gt_in_frame:
            gt_in_frame[frame_num].append(gt_vals)
        else:
            gt_in_frame[frame_num] = [gt_vals]
    total = 0
    total_records = 0
    cur_frame_num = 0
    if args.rotate_jitter:
        angles = [0, -5.0, 5.0]
    else:
        angles = [0]
    media = cv2.VideoCapture(media_path)
    if media is None:
        raise RuntimeError("can't open '" + media_path + "'")

    total_frames = media.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    while True:
        ret_val, image = media.read()
        if not ret_val:
            break
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cur_frame_num += 1

        if cur_frame_num not in gt_in_frame:
            continue

        if cur_frame_num % 20 == 0:
            print('frame %d/%d\r' % (cur_frame_num, total_frames), end="")
            sys.stdout.flush()
        for angle in angles:
            if angle == 0:
                image_cur = image
            else:
                image_cur = ndimage.interpolation.rotate(image, angle, reshape=False)
                angle = math.pi * angle / 180

            for record in gt_in_frame[cur_frame_num]:
                gt_vals = record

                if args.obj_part == 'whole':
                    cx, cy, w, h = whole_rect_from_gt(gt_vals, image.shape)
                else:
                    cx, cy, w, h = upper_rect_from_gt(gt_vals, image.shape)

                if angle == 0:
                    cx_cur = cx
                    cy_cur = cy
                else:
                    cx_cur, cy_cur = rotate_coords(cx, cy, angle, image.shape[1] / 2, image.shape[0] / 2)

                total_records += 1

                if cx_cur - w / 2 < 0 or cx_cur + w / 2 >= image.shape[1] or \
                        cy_cur - h / 2 < 0 or cy_cur + h / 2 >= image.shape[0]:
                    continue

                if object_type is None or object_type == int(gt_vals[-1]):
                    export(image_cur, cy_cur, cx_cur, w, h, out)
                    # faces are symmetric and we exploit this here
                    mirror_and_export(image_cur, cy_cur, cx_cur, w, h, out)
                    total += 1
    return total, total_records


out = open(args.out, 'wb')
if args.gt:
    if os.path.isdir(args.media):
        total, total_records = scan_image_folder(args.media, args.gt, args.object)
    else:
        total, total_records = scan_movie(args.media, args.gt, args.object)

    print('Exported %d (%d * %d * 2) objects from %d records' %\
          (total * args.nrands * 2, total, args.nrands, total_records))
else:
    total = 0
    all_images = glob.glob(args.media + '/*')
    for cur_file in all_images:
        print(cur_file + ' (' + str(total) + '/' + str(len(all_images)) + ')\r', end="")
        sys.stdout.flush()
        image = cv2.imread(cur_file, cv2.IMREAD_GRAYSCALE)
        if args.rotate is None:
            pass
        elif args.rotate == 90:
            image = np.rot90(image)
        elif args.rotate == 180:
            image = np.rot90(image, 2)
        elif args.rotate == 270:
            image = np.rot90(image, 3)
        else:
            image = ndimage.interpolation.rotate(image, args.rotate)
        if image is None:
            print("can't open image '" + cur_file + "'")
            continue
        write_rid(image, out)
        out.write(struct.pack('i', 0))
        if args.plot:
            cv2.imshow('pic', image)
            cv2.waitKey(50)
        total += 1
    print('Exported %d images' % total)

out.close()
