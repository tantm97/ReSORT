from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

class Tracker:
    def __init__(self, metric, max_iou_distance = 0.7, max_age = 70, n_init = 3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            idx for idx, track in enumerate(self.tracks) if track.is_confirmed()]
        unconfirmed_tracks = [
            idx for idx, track in enumerate(self.tracks) if not track.is_confirmed()]
        # Associate confirmed tracks using appearance features.
        matches_1, unmatched_tracks_1, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            track_indice for track_indice in unmatched_tracks_1 if
            self.tracks[track_indice].time_since_update == 1]

        unmatched_tracks_1 = [
            track_indice for track_indice in unmatched_tracks_1 if
            self.tracks[track_indice].time_since_update != 1]

        matches_2, unmatched_tracks_2, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_1 + matches_2
        unmatched_tracks = list(set(unmatched_tracks_1 + unmatched_tracks_2))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        new_track = Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature)
        self.tracks.append(new_track)
        self._next_id += 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [track for track in self.tracks if not track.is_deleted()]

        # Update distance metric.
        active_targets = [track.track_id for track in self.tracks if track.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = [track.features[-1]]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)