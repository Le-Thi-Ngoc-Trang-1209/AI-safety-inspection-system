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
import subprocess
import traceback
import os
import json


# ========= Utils =========

# Threshold
IOU_MATCH_THRES_HELMET = 0.10      
IOU_MATCH_THRES_HARNESS   = 0.20      
USE_REGIONS = True 
CONE_ID = 0                
HARNESS_ID = 1
HELMET_ID = 2
PERSON_ID = 3


# Color
COLOR_OK      = (0, 200, 0)        # xanh lá
COLOR_NG      = (0, 0, 255)        # đỏ
COLOR_HELMET  = (0, 200, 255)      # cam
COLOR_HARNESS    = (23, 53, 180)     # xanh dương nhạt/da cam
COLOR_GROUND = (255, 255, 255)
COLOR_REMINDER = (0, 0, 0)
COLOR_Pending = (255, 120, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
overlay = cv2.imread("Picture2.png", cv2.IMREAD_UNCHANGED)



# Draw human shape (Red: ready/NG, Blue: Checking, Green: OK)
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


# Calculate the iou threshold between H, N and P
def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
    
# Calculate head area of person
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


# Exception logs
def write_log(message):
    #print (message)
    os.makedirs("log/exception_log", exist_ok=True)
    log_file = os.path.join(
        "log/exception_log",
        datetime.now().strftime("%Y-%m-%d.log")
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
        f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
    )

# Play recorded sound
def play_audio(path="Test"):
    path = "sound/"+ path + ".wav"
    #print(path)
    try: 
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        subprocess.Popen(["paplay", path])
        message = f"Play sound {path} successfully"
    except Exception:
        message = f"Sound playback failed\n {traceback.format_exc()}"
    write_log(message)


#### Init
fps_counter = 0
start_time = time.time()
frame_id = 0
latest_result = "pending...."
color_result = COLOR_Pending
last_reminder = None
reminder_frames = 0
FPS_EST = 20
VOICE_DELAY_SEC = 4
last_announced_state = None
last_person_id = 0
last_sound = 0
state_change_counter = 0  
frame_w = 640
frame_h = 640
zone_xyxy = [100,50,500,600]


#------------------------------ Connect Post processing logic step -----------------------------
# Initial model
det_model = YOLO("weight/weights.engine")

# Initial RTSP stream
RTSP = "rtsp://10.21.1.213:8554/preview"
try:
    reader = RTSPReader(RTSP)
except Exception:
    write_log(f"[ERROR] Camera loading failed.")

# tracker + smoother
tracker = SimpleTracker()

# filter
filter = InspectionSelector(zone_xyxy)

# second summary  
agg = InspectionAggregator(window_sec=3)


    
# ================== OAK PIPELINE ==================
while True:
    ret, frame = reader.read()
    frame_time = datetime.now().astimezone()  

    #print(frame_time) 
    if not ret:
        write_log(f"[ERROR] Camera read failed.")
        continue
	
    #t0 = time.perf_counter()

    output = frame.copy()
    output = draw_human_shape(output)
    cv2.putText(output, "Please move until the human line turns blue", (30, 40), FONT, 0.8, COLOR_Pending, 2, cv2.LINE_AA)
    
    #t1 = time.perf_counter()
    det_result = det_model(
        frame,
        conf=0.6,
        verbose=False
    )[0]
    #t2 = time.perf_counter()

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
    cv2.rectangle(output, (0, 580), (640, 640), COLOR_GROUND, -1)
    if active_pids == []:
        # person is not in available area
        target_person = None
        ready = False
        reminder = "Please stand within the green frame"
    else:
        # has person in prediction area
        ready, target_person, reminder = filter.select(active_pids)
        
        # Count the number of reminders
        if reminder == last_reminder:
            reminder_frames += 1
        else:
            reminder_frames = 1
            last_reminder = reminder
        #print(target_person)

        # Follow the person in green frame/ human shape
        if target_person is not None: 
            px1, py1, px2, py2 = map(int, target_person["xyxy"])
            new_person_id = int(target_person["personID"])
            cv2.rectangle(output, (px1, py1), (px2, py2), COLOR_HELMET, 3)
    
    # Vote
    if not ready: 
        # no person -> reset
        agg.reset() 
        latest_result = "Checking....Please wait a moment."
        color_result = COLOR_Pending
        cv2.putText(output, reminder, (20, 620), FONT, 1, COLOR_REMINDER, 2, cv2.LINE_AA) 

        # Remind if person stands in an unavailble area too long
        if reminder_frames >= (FPS_EST * VOICE_DELAY_SEC):
            if reminder == "Please move closer to the camera":
                play_audio("close")
            elif reminder == "Please move back.":
                play_audio("back")
            reminder_frames = 0
    else:
        result = agg.ingest(target_person)
        #print(result)
        if result is not None:
            #print(result)
            #message = f"Received PPE result: {result}"
            write_log(f"Received PPE result:")
            write_log(json.dumps(result, default=str))
            current_state = result["state"]

            # Link to text to display
            if current_state == "OK":
                color_result = COLOR_OK 
                latest_result = "PASS"
            else:
                color_result = COLOR_NG 
                if current_state == "NG_helmet":
                    latest_result = "NG! Please put on your helmet!"
                elif current_state == "NG_harness":
                    latest_result = "NG! Please put on your harness!"
                else:
                    latest_result = "Please check your safety equipment!"


            # Link to voice to speak the reminder
            if new_person_id == last_person_id:
                if current_state == last_announced_state:
                    last_sound += 1
                    state_change_counter = 0
                    if last_sound >= 4:
                        ## Repeat
                        play_audio(current_state)
                        last_sound = 1   
                    else:
                        #print("The same state with same person. Skip alarm.") 
                        write_log("The same state with same person. Skip alarm.")               
                else:
                    state_change_counter += 1
                    if state_change_counter >= 2:
                        ## Renew state
                        play_audio(current_state)
                        state_change_counter = 0  
                        last_sound = 1
                        last_announced_state = current_state
                    else:
                        last_sound = 0
                        #print("Skip alarm. Noise because of fast sound")
                        write_log("Skip alarm. Noise because of fast sound")
            else:
                last_sound = 1
                state_change_counter = 0
                play_audio(current_state) 
                last_announced_state = current_state
                ## update the last person id:
                last_person_id = new_person_id

        # change color of human shape
        output=draw_human_shape(output, color_result)

        # show result (not change in 1~2s)
        cv2.rectangle(output, (0, 580), (640, 640), COLOR_GROUND, -1)
        cv2.putText(output, latest_result, (10, 620), FONT, 1, color_result, 2, cv2.LINE_AA)

    #print(output.shape)
    #cv2.namedWindow("PPE inspection", cv2.WINDOW_NORMAL)
    cv2.imshow("PPE inspection", output)
    
    # Calculate the FPS
    """fps_counter += 1
    if fps_counter == 10:
        elapsed = time.time() - start_time
        fps = fps_counter / elapsed
        print(f"Average FPS (10 frames): {fps:.2f}")
        fps_counter = 0
        start_time = time.time()"""

    frame_id += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

