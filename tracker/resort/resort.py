from __future__ import print_function
import os
import numpy as np

import cv2
import torch
from filterpy.kalman import KalmanFilter
import shutil

from .utils.associate_detection_trackers import associate_detections_to_trackers
from tracker.extractor.config import Config
from .utils.loader import load_model
from .utils.metrics import calculate_similarity_cosine, save_feature, calculate_similarity_euclid
from .utils.faces_processing import check_frontal_face, preprocess_image, alignment


class KalmanTracker(object):
    counter = 1
    def __init__(self, bbox, state='unconfirmed', enable_counter=True):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        # self.kf.x[:4] = np.array([bbox[0], bbox[1], bbox[2], bbox[3]]).reshape((4, 1))
        self.time_since_update = 0
        self.id = KalmanTracker.counter
        if enable_counter:
            KalmanTracker.counter += 1

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.prob = -1.0
        self.state = state

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        # self.kf.update(measurement)

    def __call__(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        # self.history.append(self.kf.x)
        return self.history[-1]

    def get_current_x(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
        # bbox = (np.array([self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]]).reshape((1, 4)))
        # return bbox


class ReSort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, sim_threshold=0.44):
        self.opt = Config()
        self.sim_threshold = sim_threshold
        self.iou_threshold = iou_threshold
        self.current_trackers = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
        self.model = load_model(self.opt.test_model_path, self.opt.backbone, self.opt.use_cpu)
        self.model.eval()
        self.tracking_src = 'tracking_features'
        self.create_tracking_src(self.tracking_src)


    def create_tracking_src(self, tracking_src):
        if os.path.isdir(tracking_src):
            shutil.rmtree(tracking_src)
        os.makedirs(tracking_src)


    def extract_features(self, im):
        image = preprocess_image(im, self.opt.use_cpu)
        self.model.eval()
        with torch.no_grad():
            if not self.opt.use_cpu:
                features = self.model(image).cpu().numpy()
            else:
                features = self.model(image).numpy()
        # data = torch.from_numpy(image)
        # data = data.to(torch.device("cuda"))
        # output = self.model(data)
        # features = output.data.cpu().numpy()
        return features


    def find_existed_id(self, im, detections, facial_landmarks):
        # print(detections)
        # roi = im[int(detections[1]):int(detections[3]), int(detections[0]):int(detections[2])]
        roi = alignment(im, facial_landmarks)
        # cv2.imwrite("a.jpg", roi)
        det_feature = self.extract_features(roi)
        recover_id, prob = calculate_similarity_cosine(det_feature, self.tracking_src, self.sim_threshold)
        # recover_id, prob = calculate_similarity_euclid(det_feature, self.tracking_src)
        return recover_id, prob, det_feature


    def check_current_state_of_face(self, im, detections, facial_landmarks):
        state = 'unconfirmed'
        recover_id = -1
        prob = 0
        det_feature = []
        if check_frontal_face(facial_landmarks):
            recover_id, prob, det_feature = self.find_existed_id(im, detections,facial_landmarks)
            state = 'confirmed'
        return state, recover_id, prob, det_feature


    def create_tracker(self, detections, state='unconfirmed', enable_counter=True):
        tracker = KalmanTracker(detections, state=state, enable_counter=enable_counter)
        # measurement = np.array((4,1), np.float32)
        # measurement = np.array([[int(detections[0])], [int(detections[1])], [int(detections[2])],
        #                         [int(detections[3])]], np.float32)
        tracker.update(detections)
        return tracker


    def init_tracker(self, im, detections, facial_landmarks):
        state, recover_id, prob, det_feature = self.check_current_state_of_face(im, detections,facial_landmarks)

        enable_counter = False
        if recover_id == -1:
            enable_counter = True
        
        tracker = self.create_tracker(detections[:-1], state=state, enable_counter=enable_counter)

        if enable_counter and state == 'confirmed':
            save_feature(self.tracking_src + '/' + str(tracker.id) + '.npy', det_feature)
        elif not enable_counter:
            tracker.id = recover_id
            tracker.prob = prob

        return tracker


    def get_predicted_tracker(self, current_trackers):
        predicted_trackers = []
        for t in range(len(current_trackers)):
            predictions = current_trackers[t]()[:4]
            predicted_trackers.append(predictions[0])

        predicted_trackers = np.asarray(predicted_trackers)
        return predicted_trackers


    def update_matched_trackers(self, im, detections, facial_landmarks, matched, unmatched_trackers):
        change_id_dict = {}
        for t in range(len(self.current_trackers)):
            if(t not in unmatched_trackers):
                d = matched[np.where(matched[:,1]==t)[0], 0]
                self.current_trackers[t].update(np.array([detections[d, 0], detections[d, 1], 
                                        detections[d, 2], detections[d, 3]]).reshape((4, 1)))
                if self.current_trackers[t].state == 'unconfirmed' and check_frontal_face(facial_landmarks[d[0]]):
                    recover_id, prob, det_feature = self.find_existed_id(im, detections[d[0]],facial_landmarks[d[0]])
                    if recover_id != -1:
                        change_id_dict[self.current_trackers[t].id] = recover_id
                        self.current_trackers[t].id = recover_id
                        self.current_trackers[t].prob = prob
                    else:
                        save_feature(self.tracking_src + '/' + str(self.current_trackers[t].id) + '.npy', det_feature)
                    self.current_trackers[t].state = 'confirmed'
        return change_id_dict


    def update_unmatched_detections(self, im, detections, facial_landmarks, unmatched_detections, unmatched_trackers):
        for i in unmatched_detections:
            state, recover_id, prob, unmatched_det_feature = self.check_current_state_of_face(im, detections[i],facial_landmarks[i])
            if recover_id == -1:
                tracker = self.create_tracker(detections[i, :-1], state=state)
                if state == 'confirmed':
                    save_feature(self.tracking_src + '/' + str(tracker.id) + '.npy', unmatched_det_feature)
                self.current_trackers.append(tracker)
            else:
                existed = False
                if len(unmatched_trackers) > 0 and unmatched_trackers is not None:
                    for index, t in enumerate(self.current_trackers):
                        if t.id == recover_id:
                            if len(np.where(unmatched_trackers == index)[0]) > 0:
                                existed = True
                                t.kf.x[:4] = np.array([detections[i, :-1][0], detections[i, :-1][1], detections[i, :-1][2], detections[i, :-1][3]]).reshape((4, 1))
                                t.update(np.array([detections[i, 0], detections[i, 1], 
                                    detections[i, 2], detections[i, 3]]).reshape((4, 1)))
                                t.time_since_update = 0
                                t.prob = prob
                                t.state = state
                                # print(unmatched_trackers)
                                # print(np.where(unmatched_trackers == index)[0])
                                unmatched_trackers = np.delete(unmatched_trackers, np.where(unmatched_trackers == index)[0][0])
                            break
                if not existed:
                    tracker = self.create_tracker(detections[i, :-1], state=state)
                    tracker.id = recover_id
                    tracker.prob = prob
                    self.current_trackers.append(tracker)
        return unmatched_trackers


    def update_unmatched_trackers(self, unmatched_trackers):
        for index in sorted(unmatched_trackers, reverse=True):
            self.current_trackers[index].time_since_update += 1
            if self.current_trackers[index].time_since_update > self.max_age:
                self.current_trackers.pop(index)


    def update(self, detections, facial_landmarks, image):
        self.frame_count += 1
        retain_trackers = []
        change_id_dict = {}
        if len(self.current_trackers) == 0:
            self.current_trackers = []
            for d in range(len(detections)):
                tracker = self.init_tracker(image, detections[d], facial_landmarks[d])
                self.current_trackers.append(tracker)
        else:
            predicted_trackers = self.get_predicted_tracker(self.current_trackers)
            matched, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections[:, :-1], 
                                                                                                predicted_trackers, 
                                                                                                iou_threshold=self.iou_threshold)
            # print ('Matched Detections & Trackers', len(matched))
            # print ('Unmatched Detections', len(unmatched_detections))
            # print ('Unmatched Trackers', len(unmatched_trackers))
            # print ('Current Trackers', len(self.current_trackers))
            change_id_dict = self.update_matched_trackers(image, detections, facial_landmarks, matched, unmatched_trackers)
            unmatched_trackers = self.update_unmatched_detections(image, detections, facial_landmarks, unmatched_detections, unmatched_trackers)
            self.update_unmatched_trackers(unmatched_trackers)

        for trk in self.current_trackers:
            d = trk.get_current_x()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                retain_trackers.append(np.concatenate((np.concatenate((d[0], [trk.id])), 
                                        np.concatenate(([trk.prob], [trk.state]))), axis=None).reshape(1,-1))

        if(len(retain_trackers) > 0):
            return np.concatenate(retain_trackers), change_id_dict

        return np.empty((0,7)), change_id_dict


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))
