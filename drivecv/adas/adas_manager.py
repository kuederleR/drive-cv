"""
Unified ADAS Safety Manager coordinating LDW and FCW systems.
"""

from typing import List, Optional
from drivecv.config import ADASConfig
from drivecv.core.geometry import CameraGeometry
from drivecv.adas.fcw import ForwardCollisionWarning
from drivecv.adas.ldw import LaneDepartureWarning
from drivecv.types import ADASAlert, ADASAlertLevel, LaneBoundaries, LDWState, Track


class ADASManager:
    """
    Central ADAS coordinator.
    Runs Lane Departure Warning (LDW) and Forward Collision Warning (FCW) per frame.
    """

    def __init__(self, config: Optional[ADASConfig] = None):
        self.config = config or ADASConfig()
        self.camera_geom = CameraGeometry(self.config.camera)
        self.ldw = LaneDepartureWarning(self.config)
        self.fcw = ForwardCollisionWarning(self.config, self.camera_geom)

    def update_resolution(self, width: int, height: int):
        """Updates camera model resolution."""
        self.camera_geom.update_resolution(width, height)

    def process(
        self,
        tracks: List[Track],
        lanes: Optional[LaneBoundaries],
        frame_width: int,
        frame_height: int,
        timestamp: float,
        dt: float = 0.04,
    ) -> ADASAlert:
        """
        Executes ADAS safety analysis on current scene state.
        """
        # 1. Update Lane Departure Warning
        ldw_state, ldw_offset_m, ldw_tlc_s = self.ldw.update(
            lanes=lanes,
            frame_width=frame_width,
            dt=dt,
        )

        # 2. Update Forward Collision Warning
        fcw_level, lead_track, lead_dist, lead_rel_speed, lead_ttc = self.fcw.update(
            tracks=tracks,
            lanes=lanes,
            timestamp=timestamp,
            dt=dt,
        )

        # 3. Construct advisory/warning message
        msg_parts = []
        if ldw_state == LDWState.WARNING_LEFT:
            msg_parts.append("LANE DEPARTURE: LEFT")
        elif ldw_state == LDWState.WARNING_RIGHT:
            msg_parts.append("LANE DEPARTURE: RIGHT")

        if fcw_level == ADASAlertLevel.CRITICAL:
            msg_parts.append(f"BRAKE! TTC: {lead_ttc:.1f}s" if lead_ttc else "BRAKE! COLLISION IMMINENT")
        elif fcw_level == ADASAlertLevel.WARNING:
            msg_parts.append(f"COLLISION WARNING: TTC {lead_ttc:.1f}s" if lead_ttc else "COLLISION WARNING")
        elif fcw_level == ADASAlertLevel.CAUTION:
            msg_parts.append(f"CAUTION: LEAD VEHICLE CLOSING")

        warning_message = " | ".join(msg_parts) if msg_parts else None

        return ADASAlert(
            ldw_state=ldw_state,
            ldw_offset_m=ldw_offset_m,
            ldw_tlc_s=ldw_tlc_s,
            fcw_level=fcw_level,
            fcw_lead_track_id=lead_track.track_id if lead_track else None,
            fcw_lead_distance_m=lead_dist,
            fcw_lead_rel_speed_kmh=lead_rel_speed,
            fcw_lead_ttc_s=lead_ttc,
            warning_message=warning_message,
        )
