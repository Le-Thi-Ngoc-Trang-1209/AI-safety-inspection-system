import depthai as dai
import cv2
import numpy as np
from ultralytics import YOLO
from savevid import RTSPReader
from person_tracking import SimpleTracker
from datetime import datetime
import time
from inspection_filter import InspectionSelector
from inspection_agg import InspectionAggregator
from PIL import Image, ImageDraw, ImageFont



det_model = YOLO("weight/fill.engine")

frame_w = 640
frame_h = 640

RTSP = "rtsp://169.254.150.5:8554/preview"
reader = RTSPReader(RTSP)


# ========= Utils =========
IOU_MATCH_THRES_HELMET = 0.10      
IOU_MATCH_THRES_HARNESS   = 0.20      
USE_REGIONS = True 
CONE_ID = 0                
HARNESS_ID = 1
HELMET_ID = 2
PERSON_ID = 3


# Màu vẽ
COLOR_OK      = (0, 200, 0)        # xanh lá
COLOR_NG      = (0, 0, 255)        # đỏ
COLOR_HELMET  = (0, 200, 255)      # cam
COLOR_HARNESS    = (23, 53, 180)     # xanh dương nhạt/da cam
COLOR_GROUND = (255, 255, 255)
COLOR_REMINDER = (0, 0, 0)
COLOR_Pending = (255, 120, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
overlay = cv2.imread("Picture2.png", cv2.IMREAD_UNCHANGED)

def draw_human_shape(frame, color=(0, 0, 255)):
    """
    color là BGR:
    (255,0,0) : xanh dương
    (0,255,0) : xanh lá
    (0,0,255) : đỏ
    (0,255,255) : vàng
    """
    overlay_resize = cv2.resize(overlay, (320, 600))
    # đổi màu
    alpha_mask = overlay_resize[:, :, 3] > 0
    overlay_resize[alpha_mask, 0] = color[0]
    overlay_resize[alpha_mask, 1] = color[1]
    overlay_resize[alpha_mask, 2] = color[2]
    x = (640 - 320) // 2
    y = 40
    alpha = overlay_resize[:, :, 3] / 255.0
    roi = frame[y:y+600, x:x+320]
    for c in range(3):
        roi[:, :, c] = (
        roi[:, :, c] * (1 - alpha)
        + overlay_resize[:, :, c] * alpha
        )
    frame[y:y+600, x:x+320] = roi
    return frame

def draw_jp_text(img, text, x, y, color=(0,255,0), size=23):
    font = ImageFont.truetype(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    size
    )
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(
        (x, y),
        text,
        font=font,
        fill=color
        )
    return cv2.cvtColor(
        np.array(pil),
        cv2.COLOR_RGB2BGR
        )

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
    
def head_region(bbox, top_ratio=0.3):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    return [x1, y1, x2, y1 + int(h * top_ratio)]

# Take the bbox of torso region
def torso_region(bbox, top=0.2, bottom=0.90):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    t1 = y1 + int(h * top)
    t2 = y1 + int(h * bottom)
    return [x1, t1, x2, t2]

# Process matching EQ with person
def match_eq_to_person(eq, person, region_fn, iou_thresh):
    used = set()
    has_eq = []
    eq_bbox = []
    eq_conf = []
    # Iterate through each person:
    for p in person:
        pbox = p["bbox"]
        target_region = region_fn(pbox)
        # The index/IoU of the item with the highest score.
        best_iou = 0.0
        best_idx = -1

        # Iterate through all items:
        for i, itm in enumerate(eq):
            if i in used:
                continue
            iou = iou_xyxy(target_region, itm["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
                # => Ensure: 1 person - only 1 best item

        if best_idx != -1 and best_iou >= iou_thresh:
            has_eq.append(True)
            eq_bbox.append(eq[best_idx]["bbox"])
            eq_conf.append(eq[best_idx]["conf"])
            used.add(best_idx)
            # => Ensure: 1 item - only 1 person
        else:
            has_eq.append(False)
            eq_bbox.append(None)
            eq_conf.append(None)
    return has_eq, eq_bbox, eq_conf

#------------------------------ Connect Post processing logic step -----------------------------
# tracker + smoother
tracker = SimpleTracker()
# filter
zone_xyxy = [100,50,500,600]
filter = InspectionSelector(zone_xyxy)
# minute summary  
agg = InspectionAggregator(window_sec=3)
fps_counter = 0
start_time = time.time()
    
# ================== OAK PIPELINE ==================
frame_id = 0
latest_result = "Checking...."
#latest_result = "                確認中..."
color_result = COLOR_Pending
while True:
    ret, frame = reader.read()
    frame_time = datetime.now().astimezone()  

    #print(frame_time) 
    if not ret:
        continue
	
    
    t0 = time.perf_counter()

    output = frame.copy()
    #cv2.rectangle(output, (zone_xyxy[0], zone_xyxy[1]), (zone_xyxy[2], zone_xyxy[3]), COLOR_NG, 7)
    output = draw_human_shape(output)
    cv2.putText(output, "Please align your body with the line", (30, 40), FONT, 0.8, COLOR_NG, 2, cv2.LINE_AA)
    #output = draw_jp_text(output, "頭のてっぺんを緑色のラインに合わせてください", 30, 10, COLOR_REMINDER)
    
    t1 = time.perf_counter()
    det_result = det_model(
        frame,
        conf=0.6,
        verbose=False
    )[0]
    t2 = time.perf_counter()

    cones, persons, helmets, harnesses = [], [], [], []
    if det_result.boxes is not None:
        for b in det_result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            entry = {
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            }
            if cls_id == PERSON_ID:
                persons.append(entry)
            elif cls_id == HELMET_ID and conf > 0.72:
                helmets.append(entry)
            elif cls_id == HARNESS_ID and conf > 0.7:
                harnesses.append(entry)
        
    if USE_REGIONS:
        has_helmet, helmet_bbox, helmet_conf = match_eq_to_person(helmets, persons, head_region, IOU_MATCH_THRES_HELMET)
        has_harness, harness_bbox, harness_conf = match_eq_to_person(harnesses, persons, torso_region, IOU_MATCH_THRES_HARNESS)
    else:
        has_helmet, helmet_bbox, helmet_conf = match_eq_to_person(helmets, persons, lambda b: b, IOU_MATCH_THRES_HELMET)
        has_harness, harness_bbox, harness_conf = match_eq_to_person(harnesses, persons, lambda b: b, IOU_MATCH_THRES_HARNESS)

    # Count & Draw
    outlist = []
    for i, p in enumerate(persons):
        #px1, py1, px2, py2 = map(int, p["bbox"])
        #conf = p["conf"]
        #ok_person = bool(has_helmet[i]) and bool(has_harness[i]) # T or F
        
        if has_helmet[i] and has_harness[i]:
           status = 'OK'
        elif not has_helmet[i] and has_harness[i]:
            status = 'NG_helmet'
        elif has_helmet[i] and not has_harness[i]:
            status = 'NG_harness'
        else:
            status = 'NG_H&N'
            
        outlist.append([
            p["bbox"],
            p["conf"],
            status,])

   
    #print(outlist)
    # Tracking + State Smoothing
    active_pids, lost_pids, new_pids = tracker.update(frame_id, outlist, frame_time=frame_time)
    #print("Pids:", active_pids)
    
    # Draw simple bbox
    """valid_pids = []
    for p in active_pids:
        state = str(p["state"])
        px1, py1, px2, py2 = map(int, p["xyxy"])
        
        cv2.rectangle(output, (px1, py1), (px2, py2), COLOR_HARNESS, 3)
        #cv2.putText(output, label, ((px1 + 6), (py2 - 10)), FONT, 0.8, color, 2, cv2.LINE_AA)"""

    cv2.rectangle(output, (0, 600), (640, 640), COLOR_GROUND, -1)
    if active_pids == []:
        target_person = None
        ready = False
        reminder = "Waiting..."
    else:
        ready, target_person, reminder = filter.select(active_pids)
        #print(target_person)
        if target_person is not None: 
            px1, py1, px2, py2 = map(int, target_person["xyxy"])
            cv2.rectangle(output, (px1, py1), (px2, py2), COLOR_HELMET, 3)
    
    # Flush the previous mininute: Save the image 
    if not ready: 
        agg.reset() 
        latest_result = "Checking....Please remain still for a few second."
        #latest_result = "                確認中..."
        color_result = COLOR_Pending
        cv2.putText(output, reminder, (50, 630), FONT, 0.8, COLOR_REMINDER, 2, cv2.LINE_AA) 
        #cv2.rectangle(output, (zone_xyxy[0], zone_xyxy[1]), (zone_xyxy[2], zone_xyxy[3]), color_result, 7)
        #output = draw_jp_text(output, reminder, 50, 600, COLOR_REMINDER, size=30)
    else:
        result = agg.ingest(target_person)
        if result is not None:
            if result["state"] == "OK":
                color_result = COLOR_OK 
                latest_result = "PASS"
                #latest_result = "保安具OKです"
            else:
                color_result = COLOR_NG 
                if result["state"] == "NG_helmet":
                    latest_result = "NG! Please put on your helmet!"
                    #latest_result = "NG!ヘルメットを着用してください。"
                elif result["state"] == "NG_harness":
                    latest_result = "NG! Please put on your harness!"
                    #latest_result = "NG!ハーネスを着用してください。"
                else:
                    latest_result = "NG! Please check your safety equipment!"
                    #latest_result = "NG!安全装備を確認してください。"
            #cv2.rectangle(output, (zone_xyxy[0], zone_xyxy[1]), (zone_xyxy[2], zone_xyxy[3]), color_result, 7)
            output=draw_human_shape(output, color_result)

        cv2.putText(output, latest_result, (50, 630), FONT, 0.8, color_result, 2, cv2.LINE_AA)
        #output = draw_jp_text(output, latest_result, 50, 600, color_result, size=30)
        
    cv2.imshow("PPE + Danger Zone", output)
    fps_counter += 1
    if fps_counter == 10:
        elapsed = time.time() - start_time
        fps = fps_counter / elapsed
        print(f"Average FPS (10 frames): {fps:.2f}")
        fps_counter = 0
        start_time = time.time()

    t3 = time.perf_counter()
    print(f"prep={(t2-t1)*1000:.1f}ms "
            f"total={(t3-t0)*1000:.1f}ms")
    frame_id += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

