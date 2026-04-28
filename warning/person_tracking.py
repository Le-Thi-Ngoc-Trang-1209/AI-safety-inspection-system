from dataclasses import dataclass
from datetime import datetime
import json
from typing import List, Tuple, Dict, Any
from pathlib import Path

# ----------------- Utility -----------------
def iou_xyxy(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    """IoU giữa 2 bbox [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter + 1e-9
    return inter / union

# for data class (auto call init, repr, eq)
@dataclass(slots=True)
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    last_update_frame: int
    hits: int = 0  
    state_cur: str = ""
    state_cnt: int = 0
    state_official: str = ""


# ----------------- Simple Tracker -----------------
class SimpleTracker:
    def __init__(
        self, 
        iou_thresh: float = 0.3, 
        max_age: int = 60,
        log_dir: str = "Frame_log",
        save_log: bool = False
    ):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks: List[Track] = [] # first track
        self.next_id: int = 1 # track_ID
        self.log_dir = Path(log_dir)
        self.K: int = 5    # K continuous frames are needed to change the state.
        self.save_log = save_log
        

    # --------- Input Normalization ---------
    # data_dict = {"data": [[[x1,y1,x2,y2], confidence, state],]}
    # detections = [[x1,y1,x2,y2], confidence, state]
    def _normalize_from_dict(self, data_dict: Dict[str, Any]) -> Tuple[Tuple[List[float], float, str]]:
        rows = data_dict.get("data", []) or []
        dets: Tuple[Tuple[List[float], float, str]] = []

        for row in rows:
            # check each row: [[x1,y1,x2,y2], confidence, detection_id]
            if not (isinstance(row, (list, tuple)) and len(row) >= 3):
                continue
            xyxy, conf, det_id = row[:3]

            # Check bounding box
            if not (isinstance(xyxy, (list, tuple)) and len(xyxy) == 4):
                continue
            x1, y1, x2, y2 = map(float, xyxy)
            if x2 <= x1 or y2 <= y1:
                continue

            dets.append(((x1, y1, x2, y2), float(conf), str(det_id)))
        return dets

    # --------- Main update per frame ---------
    # data_dict = {"data": [[class_id, class_name, [x1,y1,x2,y2], confidence, state],]}
    # [frame_id, track_id, state, confidence, [x1,y1,x2,y2]]
    def update(self, frame_idx: int, data_dict: Dict[str, Any], frame_time: datetime) -> List[Dict[str, Any]]:
        newtrack = []
        # 0) Delete Overdue track (>30frs))
        """self.tracks = [
                tr for tr in self.tracks
                if (frame_idx - tr.last_update_frame) <= self.max_age
            ]       
        
        """
        # 0) Delete overdue tracks: thu thập danh sách
        removed = []
        alive_tracks = []
        for tr in self.tracks:
            if (frame_idx - tr.last_update_frame) <= self.max_age:
                alive_tracks.append(tr)
            else:
                if tr.hits >= 15:
                    removed.append(tr)
        self.tracks = alive_tracks

        
        # 1) Take the detections
        dets = self._normalize_from_dict(data_dict)
        if len(dets) == 0:
           return [], [tr.track_id for tr in removed], []
        
        # 2) Matching persons to tracking (reID step)
        if len(self.tracks) == 0:
            matches = []
            unmatched_dets = list(range(len(dets)))
        else:
            # Iou (track-detection)
            candidates = []  # (iou, t_idx, d_idx)
            # t_idx, tr: trackID, trackdata
            for t_idx, tr in enumerate(self.tracks):
                for d_idx, (det_box, _, _) in enumerate(dets):
                    iou = iou_xyxy(tr.bbox, det_box)
                    if iou >= self.iou_thresh:
                        candidates.append((iou, t_idx, d_idx))

            # Greedy: sort
            candidates.sort(key=lambda x: x[0], reverse=True)
            matched_tracks, matched_dets = set(), set()
            matches: List[Tuple[int, int]] = []              
            max_possible = min(len(self.tracks), len(dets))
            for iou, t_idx, d_idx in candidates:
                # skip matched track and detection
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue           
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                matches.append((t_idx, d_idx))
                if len(matches) == max_possible:
                    break

            # Update unmatched detections.
            unmatched_dets = [di for di in range(len(dets)) if di not in matched_dets]

        outputs: List[Dict[str, Any]] = []     
        # 3) Update tracks matched: update bbox/last frame, hits
        for t_idx, d_idx in matches:
            tr = self.tracks[t_idx]
            det_box, conf, state = dets[d_idx]
            # Update old track
            tr.bbox = det_box
            tr.last_update_frame = frame_idx
            tr.hits += 1

            # State smoothing using debounce
            if state == tr.state_cur:
                tr.state_cnt += 1
            else:
                tr.state_cur = state
                tr.state_cnt = 1
            if tr.state_cnt >= self.K:
                tr.state_official = tr.state_cur
            state_out = tr.state_official if tr.state_official else tr.state_cur

            # Update output
            outputs.append({
                            "time": frame_time,
                            "frame_id": frame_idx,
                            "personID": tr.track_id,
                            "frames_total": tr.hits,
                            "state": state_out,
                            "confidence": conf,
                            "xyxy": list(det_box)                           
                        })

        # 4) Create new track for new detection (unmatched_dets)
        for d_idx in unmatched_dets:
            det_box, conf, state = dets[d_idx] # bbox
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    bbox=det_box,
                    last_update_frame=frame_idx,
                    hits=1,
                    state_cur=state,
                    state_cnt=0,
                    state_official=state
                )
            )
            track = {
                "time": frame_time,
                "frame_id": frame_idx,
                "personID": self.next_id,
                "frames_total": 1,
                "state": state,
                "confidence": conf,
                "xyxy": list(det_box)
            }
            outputs.append(track)
            newtrack.append(track)
            self.next_id += 1

        if self.save_log:
            record = {
                "time": frame_time,
                "data": dets
            }
            txt_path = self._build_path_txt_for_frame(frame_time)
            self._save_frame_log(txt_path, record)

        return outputs, [tr.track_id for tr in removed], newtrack
        #return outputs
    

    # Create the frame log path
    def _build_path_txt_for_frame(self, frame_time: datetime, **kwargs) -> Path:
        # ex: reports/frame_log_2026-02-27.txt
        date_str = frame_time.strftime("%Y-%m-%d")
        return self.log_dir / f"frame_log_{date_str}.txt"
    
    # Write the outputs to the frame log.
    def _save_frame_log(self, txt_path: Path, record: dict):
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")



