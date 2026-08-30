"""
Forward Collision Warning (FCW) System.
Uses the ego-lane lead vehicle's filtered range / range-rate when available.
"""

from typing import Dict, List, Optional, Tuple
from drivecv.config import ADASConfig
from drivecv.core.geometry import CameraGeometry
from drivecv.types import ADASAlertLevel, LaneBoundaries, Track, TrackLifecycle


class ForwardCollisionWarning:
    """
    Monocular Forward Collision Warning on the single ego-lane lead vehicle.
    Prefers lane-calibrated range already stored on the track; falls back to pinhole.
    """

    def __init__(
        self,
        config: Optional[ADASConfig] = None,
        camera_geom: Optional[CameraGeometry] = None,
    ):
        self.config = config or ADASConfig()
        self.camera_geom = camera_geom or CameraGeometry(self.config.camera)
        self.prev_distances: Dict[int, float] = {}

    def _ensure_range(self, track: Track, lanes: Optional[LaneBoundaries]) -> None:
        dist_m, lat_x_m = self.camera_geom.estimate_range_from_lanes(track.bbox, lanes)
        track.kinematics.distance_m = dist_m
        track.kinematics.lateral_offset_m = lat_x_m

    def update(
        self,
        tracks: List[Track],
        lanes: Optional[LaneBoundaries],
        timestamp: float,
        dt: float = 0.04,
    ) -> Tuple[ADASAlertLevel, Optional[Track], Optional[float], Optional[float], Optional[float]]:
        """Returns (alert_level, lead_track, distance_m, rel_speed_kmh, ttc_s)."""
        del timestamp
        self.camera_geom.calibrate_from_lanes(lanes)

        candidates: List[Track] = []
        for track in tracks:
            if track.lifecycle == TrackLifecycle.DELETED:
                continue
            self._ensure_range(track, lanes)
            track.kinematics.is_lead_vehicle = False
            candidates.append(track)

        pool = candidates
        if lanes is not None and lanes.is_valid:
            in_lane = [
                t
                for t in candidates
                if lanes.contains_contact(
                    t.bbox.bottom_center[0],
                    t.bbox.bottom_center[1],
                    margin_frac=self.config.fcw_in_lane_margin_frac,
                )
            ]
            if in_lane:
                pool = in_lane

        lead_track: Optional[Track] = None
        if pool:
            scored = [t for t in pool if t.kinematics.distance_m > 1.5]
            if scored:
                lead_track = min(scored, key=lambda t: t.kinematics.distance_m)

        if lead_track is None:
            self.prev_distances = {}
            return ADASAlertLevel.SAFE, None, None, None, None

        lead_track.kinematics.is_lead_vehicle = True
        lead_dist = lead_track.kinematics.distance_m

        if abs(lead_track.kinematics.rel_speed_mps) < 1e-6 and lead_track.track_id in self.prev_distances and dt > 0.0:
            prev_d = self.prev_distances[lead_track.track_id]
            closing = (prev_d - lead_dist) / dt
            lead_track.kinematics.rel_speed_mps = float(closing)
            lead_track.kinematics.rel_speed_kmh = float(closing * 3.6)
            if closing > 0.5:
                lead_track.kinematics.ttc_seconds = float(lead_dist / closing)

        self.prev_distances = {lead_track.track_id: lead_dist}

        lead_speed_kmh = lead_track.kinematics.rel_speed_kmh
        lead_ttc = lead_track.kinematics.ttc_seconds

        alert = ADASAlertLevel.SAFE
        if lead_dist <= 6.0:
            alert = ADASAlertLevel.CRITICAL
        elif lead_ttc is not None:
            if lead_ttc <= self.config.fcw_critical_ttc_s:
                alert = ADASAlertLevel.CRITICAL
            elif lead_ttc <= self.config.fcw_warning_ttc_s:
                alert = ADASAlertLevel.WARNING
            elif lead_ttc <= self.config.fcw_caution_ttc_s:
                alert = ADASAlertLevel.CAUTION

        return alert, lead_track, lead_dist, lead_speed_kmh, lead_ttc
