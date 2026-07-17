import math
class InspectionSelector:
    def __init__(
        self,
        zone_xyxy,
        head_tolerance=30
    ):
        """
        zone_xyxy = [x1, y1, x2, y2]
        head_line_y = green threshold
        """
        self.zone = zone_xyxy
        self.zone_cx = (zone_xyxy[0] + zone_xyxy[2]) / 2
        self.zone_cy = (zone_xyxy[1] + zone_xyxy[3]) / 2
        self.head_line_y = zone_xyxy[1]
        self.head_tolerance = head_tolerance

    # ------------ Check center in zone---------------------------------
    def is_inside_zone(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        zx1, zy1, zx2, zy2 = self.zone
        return (zx1 <= cx <= zx2 and zy1 <= cy <= zy2)

    # --------- Distance to zone center----------------
    def center_distance(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return math.sqrt((cx - self.zone_cx) ** 2 + (cy - self.zone_cy) ** 2)

    # -------------Check distance using head position-------------------------------
    def check_head_position(self, bbox):
        x1, y1, x2, y2 = bbox
        dy = y1 - self.head_line_y
        if dy > self.head_tolerance*2:
            return False, "Please move closer to the camera"
            #return False, "もう少し前にお進みください"
        if dy < -self.head_tolerance:
            return False, "Please move back."
            #return False, "少し後ろに下がってください"

        return True, "READY"

    # --------------------- Main selection------------------------------------
    def select(self, tracks):
        # track in green zone
        candidates = []
        for tr in tracks:
            bbox = tr["xyxy"]
            # Check within the green zone
            if self.is_inside_zone(bbox):
                candidates.append(tr)

        if len(candidates) == 0:
            return False, None, "Please stand within the green frame"
            #return None, "緑色の枠内に立ってください"

        # d1, d2,... _> return tr
        target = min(
            candidates,
            key=lambda t: self.center_distance(t["xyxy"])
        )

        ready, msg = self.check_head_position(target["xyxy"])

        # if in valid, return track. Else: return message
        if not ready:
            return ready, target, msg

        return True, target, "READY"