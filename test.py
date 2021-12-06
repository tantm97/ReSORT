import os
import argparse
import numpy as np
import cv2
from tracker.resort.resort import ReSort
from tracker.sort.sort import Sort
from tracker.deepsort.deepsort import DeepSort
from detector.detector import RetinafaceDetector
from tracker.visualization import *

parser = argparse.ArgumentParser(description='Face tracking')

parser.add_argument('--video', default='data/multi_cam.mp4', help='video path')
parser.add_argument('--tracker', default='resort', help='Tracker: sort, deepsort, resort')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--resize', default=1, type=int, help='resize')
parser.add_argument('--scale', default=1, type=int, help='scale fps')
parser.add_argument('--save_video', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.8, type=float, help='visualization_threshold')
args = parser.parse_args()

def load_tracker(tracker, max_age=1, min_hits=3, iou_threshold=0.3):
    mot_tracker = None
    if tracker == 'sort':
        mot_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    elif tracker == 'resort':
        mot_tracker = ReSort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    elif tracker == 'deepsort':
        mot_tracker = DeepSort()

    return mot_tracker


if __name__ == '__main__':
    w_thresh = 20
    h_thresh = 20
    max_age = 1
    min_hits = 3
    iou_threshold = 0.3
    frame_rate = 1
    
    detector = RetinafaceDetector(args.network, args.cpu)

    mot_tracker = load_tracker(args.tracker, max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    
    scale = args.scale

    cap = cv2.VideoCapture(args.video)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    fps = cap.get(cv2.CAP_PROP_FPS)

    print("FPS = ", fps/scale)

    if args.save_video:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), int(fps/scale), (frame_width,frame_height))

    while(cap.isOpened()):
        ret, img_raw = cap.read()
        if frame_rate % scale == 0:
            if ret == True:
                dets = detector.detect_faces(img_raw, confidence_threshold=args.confidence_threshold, top_k=args.top_k, 
                                    nms_threshold=args.nms_threshold, keep_top_k=args.keep_top_k, resize=args.resize) 

                detections, facial_landmarks = convert_detection_to_tracker(dets, args.vis_thres, w_thresh, h_thresh, frame_width, frame_height)
                if len(detections) > 0:
                    if args.tracker == 'sort':
                        trackers = mot_tracker.update(np.array(detections))
                        # draw trackers
                        img_raw = draw_box_sort_trackers(img_raw, trackers, frame_width, frame_height)
                    elif args.tracker == 'resort':
                        trackers, change_id_dict = mot_tracker.update(np.asarray(detections), facial_landmarks, img_raw)
                        if change_id_dict:
                            print(change_id_dict)
                        # draw trackers
                        img_raw = draw_box_resort_trackers(img_raw, trackers, frame_width, frame_height)
                    elif args.tracker == 'deepsort':
                        trackers, detections_class = mot_tracker.update(img_raw, detections, facial_landmarks)
                        img_raw = draw_box_deepsort_trackers(img_raw, trackers, detections_class, frame_width, frame_height)
                    
                # draw detections
                img_raw = draw_box_and_landmarks_detections(img_raw, dets, args.vis_thres, w_thresh, h_thresh)

                cv2.imshow("result", img_raw)
                if args.save_video:
                    out.write(img_raw)
                    
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else: 
                break
        frame_rate += 1

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()