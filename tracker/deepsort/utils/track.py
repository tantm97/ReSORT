track_status = {
    "unconfirmed": 1,
    "confirmed": 2,
    "deleted": 3
}

class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature = None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = track_status["unconfirmed"]
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == track_status["unconfirmed"] and self.hits >= self._n_init:
            self.state = track_status["confirmed"]

    def mark_missed(self):
        if self.state == track_status["unconfirmed"]:
            self.state = track_status["deleted"]
        elif self.time_since_update > self._max_age:
            self.state = track_status["deleted"]

    def is_tentative(self):
        return self.state == track_status["unconfirmed"]

    def is_confirmed(self):
        return self.state == track_status["confirmed"]

    def is_deleted(self):
        return self.state == track_status["deleted"]