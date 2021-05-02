# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import time
import argparse

import posenet

#model=101 fs= 0.36-0.66

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default= 101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--view', type=bool, default=False)
args = parser.parse_args()


def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        print(output_stride)

        cap = cv2.VideoCapture(args.cam_id)
        _, _ = cap.read()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width) # カメラ画像の横幅を1280に設定
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*args.scale_factor)
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*args.scale_factor)

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image, result_keypoints = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            if args.view:
                cv2.imshow('posenet', overlay_image)
            print(result_keypoints)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count == 50:
                break
        print(args.scale_factor,args.model)
        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()