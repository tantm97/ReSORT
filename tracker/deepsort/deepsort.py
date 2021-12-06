import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

from tracker.extractor.config import Config
from .utils import nn_matching
from .utils.tracker import Tracker 
from .utils.detection import Detection
from .utils.loader import load_model
from .utils.faces_processing import check_frontal_face, preprocess_image, alignment


class DeepSort(object):
    def __init__(self, max_cosine_distance=0.44, nn_budget=100):	
        self.opt = Config()
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.trackers = Tracker(self.metric)
        self.model = load_model(self.opt.test_model_path, self.opt.backbone, self.opt.use_cpu)
        self.model.eval()
        
    def preprocessing(self, frame, facial_landmarks):
        crops = []
        for facial_landmark in facial_landmarks:
            roi = alignment(frame, facial_landmark)
            crop = preprocess_image(roi, self.opt.use_cpu)
            crop = crop.squeeze(0).float()
            crops.append(crop)
            
        if len(crops) == 0:
            return None
        crops = torch.stack(crops)
        return crops


    def update(self, frame, detections, facial_landmarks):
        if len(detections[:, :-1]) == 0:			
            self.trackers.predict()
            # print('No detections')
            trackers = self.trackers.tracks
            return trackers

        dets_bb = torch.Tensor(np.array(detections[:, :-1]))
        preprocessed_face = self.preprocessing(frame, facial_landmarks)
        
        if preprocessed_face is None:
            return None, None

        with torch.no_grad():
            if not self.opt.use_cpu:
                device = torch.cuda.current_device()
                preprocessed_face = preprocessed_face.to(device)
                features = self.model(preprocessed_face).cpu().numpy()
            else:
                features = self.model(preprocessed_face).numpy()

        if len(features.shape)==1:
            features = np.expand_dims(features,0)

        dets = [Detection(bbox, score, feature)
                    for bbox,score, feature in
                    zip(convert_tlbr_to_tlwh(dets_bb), np.array(detections[:, -1]).reshape(-1, 1), features)]
        
        # outboxes = np.array([d.tlwh for d in dets])
        # outscores = np.array([d.confidence for d in dets])
        self.trackers.predict()
        self.trackers.update(dets)

        return self.trackers, dets


def convert_tlbr_to_tlwh(coor):
    coor = coor.detach().cpu().numpy()
    ret = coor.copy()
    ret[:, 2:] -= ret[:, :2]
    return torch.Tensor(ret)
