import depthai as dai
import cv2
import numpy as np
from ultralytics import YOLO
import time
from savevid import RTSPReader
from person_track import SimpleTracker
from make_second_log import Second_Aggregator
from datetime import datetime
from upload_to_cloud import DriveAPI
import json
from dotenv import load_dotenv
import threading
import os
from pathlib import Path
import csv

def upload_async(image_path, day_folder_name):
    obj.FileUpload(filepath=image_path, parent_folder_id=day_folder_name)



#------------------------- Initialize ---------------------------------
## Note: check file .env (not env) via: cat .env; vim .env
load_dotenv(".env", override=True)
with open("secret/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
api_url = config["roboflow_pipeline"]["api_url"]
api_key_env_name = config["roboflow_pipeline"]["API_key"]
resolution = tuple(config["camera"]["resolution"])
iou_thresh = config["simple_tracker"]["iou_thresh"]
max_age = config["simple_tracker"]["max_age"]
k_debounce = config["simple_tracker"]["k_debounce"]
log_dir = config["simple_tracker"]["log_dir"]
report_dir = config["minute_aggregator"]["report_dir"]
agg_cfg = config["minute_aggregator"]
eq_threshold = config["minute_aggregator"]["ng_threshold_minutes"]  
eq_env_name = agg_cfg["url_eq"]       
cone_env_name = agg_cfg["url_cone"] 
api_key = os.environ.get(api_key_env_name) 
eq_url = os.environ.get(eq_env_name)
cone_url = os.environ.get(cone_env_name)
drive_cfg = config["google_drive"]
cred_path = drive_cfg["credentials_path"]
token_path = drive_cfg["token_path"]
#print(api_key)

# verification
if not eq_url:
    raise RuntimeError(f"Environment variable {eq_env_name} is not set")
if not api_key:
    raise RuntimeError(f"Environment variable {api_key_env_name} is not set")
if not cone_url:
    raise RuntimeError(f"Environment variable {cone_env_name} is not set")


#------ Initialize model and video stream--------
det_model = YOLO("weight/weights.engine")
seg_model = YOLO("weight/seg.engine")

RTSP = "rtsp://10.21.1.213:8554/preview"
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
COLOR_HARNESS    = (255, 120, 0)      # xanh dương nhạt/da cam
COLOR_CONE = (23, 53, 180)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Save image if Flag = True (NG appears)
def save_snapshot(frame, frame_time, out_dir):
    date_str = frame_time.strftime("%Y%m%d")
    time_str = frame_time.strftime("%H:%M:%S_%f")[:-3]
    name_image = frame_time.strftime("%H-%M-%S_%f")[:-3]
    save_dir = Path(out_dir) / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"Capture_time{name_image}.jpg"
    cv2.putText(
        frame,
        f"Capture time (HH:MM:SS_f): {time_str} ",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    cv2.imwrite(str(save_dir / filename), frame)
    return str(save_dir / filename)

# Create path for csv file (minute_log) => find and upload csv file
def _build_path_for_csv(subdir: str, name: str) -> str:
    base_dir = Path(subdir)
    base_dir.mkdir(parents=True, exist_ok=True)
    name = "minute_log_" + name
    if not name.endswith(".csv"):
        name = f"{name}.csv"
    return str(base_dir / name)

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
tracker = SimpleTracker(iou_thresh=iou_thresh, max_age=max_age, log_dir=log_dir, save_log=False)
# minute summary  
agg = Second_Aggregator(ng_threshold_seconds=10, do_alert=False, url=eq_url, save_log=True)

#------------------------------ Connect GG Drive ----------------------------------------------
# Initialize
"""obj = DriveAPI(credentials_path=cred_path, token_path=token_path)
cloud_name = obj.ensure_folder(folder_name="Inspectionsystemcloud", parent_folder_id=None)
image_folder_name = obj.ensure_folder(folder_name="image", parent_folder_id=cloud_name)
csv_folder_name = obj.ensure_folder(folder_name="csv", parent_folder_id=cloud_name)
day_time = datetime.now().astimezone() 
date_str = day_time.strftime("%Y%m%d")
day_folder_name = obj.ensure_folder(folder_name=date_str, parent_folder_id=image_folder_name)"""

# Init
frame_id = 0
frame_w = 640
frame_h = 640
fps_counter = 0
start_time = time.time()

# Create flag for saving NG case and flag for uploading data
Flag = False
last_uploaded_hour = None
stop_flag = False

    
# ================== OAK PIPELINE ==================
while True:
    #loop_start = time.time()

    ret, frame = reader.read()
    frame_time = datetime.now().astimezone()  
    if not ret:
        continue

    # segment the dangerous areas    
    seg_result = seg_model(
        frame,
        classes=[0, 1],
        conf=0.6,
        verbose=False
    )[0]

    output = frame.copy()
    # display segmentationon a copy of frame
    if seg_result.masks is not None:
        for mask in seg_result.masks.data:
            mask = mask.cpu().numpy().astype(bool)
            color = np.array([0, 0, 255], dtype=np.uint8)
            output[mask] = output[mask] * 0.4 + color * 0.6

    # detect objects
    det_result = det_model(
        frame,
        conf=0.6,
        verbose=False
    )[0]
    
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
            elif cls_id == HELMET_ID and conf > 0.7:
                helmets.append(entry)
            elif cls_id == HARNESS_ID and conf > 0.65:
                harnesses.append(entry)
            elif cls_id == CONE_ID:
                cones.append(entry)
        
    if USE_REGIONS:
        has_helmet, helmet_bbox, helmet_conf = match_eq_to_person(helmets, persons, head_region, IOU_MATCH_THRES_HELMET)
        has_harness, harness_bbox, harness_conf = match_eq_to_person(harnesses, persons, torso_region, IOU_MATCH_THRES_HARNESS)
    else:
        has_helmet, helmet_bbox, helmet_conf = match_eq_to_person(helmets, persons, lambda b: b, IOU_MATCH_THRES_HELMET)
        has_harness, harness_bbox, harness_conf = match_eq_to_person(harnesses, persons, lambda b: b, IOU_MATCH_THRES_HARNESS)

    # Count & Draw
    outlist = []
    for i, p in enumerate(persons):
        px1, py1, px2, py2 = map(int, p["bbox"])
        conf = p["conf"]
        ok_person = bool(has_helmet[i]) and bool(has_harness[i]) # T or F
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

        # Drawing bbox
        color = COLOR_OK if ok_person else COLOR_NG
        #label = f"OK {conf:.2f}" if ok_person else f"NG {conf:.2f}"
        cv2.rectangle(output, (px1, py1), (px2, py2), color, 3)
        cv2.putText(output, status, ((px2 + 6), (py1 + 6)), FONT, 0.8, color, 2, cv2.LINE_AA)


    # Tracking + State Smoothing
    active_pids, lost_pids, new_pids = tracker.update(frame_id, outlist, frame_time=frame_time)
    #print("Pids:", active_pids)
    #print(active_pids)

    # Flush the previous second: Save the image 
    ng_rows = agg.ingest_rows(active_pids)
    if ng_rows:
        # "The image is an illustrative snapshot; the logical timestamp is in the CSV file."
        # Accept that the image will be saved later than expected, by about 1 minute.
        img = output.copy()
        #image_path = save_snapshot(img, frame_time, out_dir="summary/Images")
        #obj.FileUpload(filepath=image_path, parent_folder_id=day_folder_name)
        #threading.Thread(target=upload_async, args=(image_path,day_folder_name), daemon=True).start()


    # Flush the overdue track: 
    for pid in lost_pids:
        second = agg.last_second_of_person.get(pid)
        if second is not None:
            # NOTE: Force flush overdue track
            is_ng = agg._flush(pid, second)


    # Upload the log summary to cloud
    current_hour = frame_time.strftime("%Y-%m-%d_%H")
    current_day = frame_time.strftime("%Y-%m-%d")
    csv_file_path = _build_path_for_csv(subdir=report_dir, name=current_day)
    if last_uploaded_hour is None:
        last_uploaded_hour = current_hour
    elif current_hour != last_uploaded_hour:
        """obj.upload_or_overwrite(
            filepath=csv_file_path,
            parent_folder_id=csv_folder_name,
            mime_type="text/csv"
        )"""
        last_uploaded_hour = current_hour
    
    ## calculate frame rate score
    fps_counter += 1
    if fps_counter == 10:
        elapsed = time.time() - start_time
        fps = fps_counter / elapsed
        print(f"Average FPS (10 frames): {fps:.2f}")
        fps_counter = 0
        start_time = time.time()

    cv2.imshow("PPE + Danger Zone", output)
    #pipeline_latency = time.time() - loop_start
    #print(f"Pipeline={pipeline_latency*1000:.1f}ms")
    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

