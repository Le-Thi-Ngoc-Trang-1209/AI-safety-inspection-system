from datetime import datetime, time
import os
import cv2
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata
from person_tracking import SimpleTracker
from make_minute_log import MinuteAggregator
from pathlib import Path
import csv
from upload_to_cloud import DriveAPI
import json
from dotenv import load_dotenv
import threading

def upload_async(image_path, day_folder_name):
    obj.FileUpload(filepath=image_path, parent_folder_id=day_folder_name)



#------------------------- Initialize ---------------------------------
load_dotenv()
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

# verification
if not eq_url:
    raise RuntimeError(f"Environment variable {eq_env_name} is not set")
if not api_key:
    raise RuntimeError(f"Environment variable {api_key_env_name} is not set")
if not cone_url:
    raise RuntimeError(f"Environment variable {cone_env_name} is not set")


# Create file csv to save output (NG people) if saving image
CSV_HEADER = ["start_time", "person_id", "frames_total", "state", "xyxy"]
def save_row_csv(row, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  
    ts = row["start_time"]
    date_str = str(ts)[:10]
    csv_path = out_dir / f"{date_str}.csv"
    file_empty = (not csv_path.exists()) or (csv_path.stat().st_size == 0)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if file_empty:
            writer.writeheader()
        row_to_write = {
            "start_time": ts,
            "person_id": row["person_id"],
            "frames_total": row["frames_total"],
            "state": row["state"],
            "xyxy": ",".join(map(str, row["xyxy"])),
        }
        writer.writerow(row_to_write)


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


#------------------------------- Connect Inference Pipeline ----------------------------------
# Initialize client
client = InferenceHTTPClient.init(
    api_url=api_url,
    api_key=api_key
)

# Configure video source (webcam)
source = WebcamSource(resolution=resolution)

# Configure streaming options
config = StreamConfig(
    stream_output=["output"],
    data_output=["data"]      
    )

# Create streaming session
session = client.webrtc.stream(
    source=source,
    workflow="eq",
    workspace="kawadard",
    image_input="image",
    config=config
)

#------------------------------ Connect GG Drive ----------------------------------------------
# Initialize
obj = DriveAPI(credentials_path=cred_path, token_path=token_path)
cloud_name = obj.ensure_folder(folder_name="Inspectionsystemcloud", parent_folder_id=None)
image_folder_name = obj.ensure_folder(folder_name="image", parent_folder_id=cloud_name)
csv_folder_name = obj.ensure_folder(folder_name="csv", parent_folder_id=cloud_name)
day_time = datetime.now().astimezone() 
date_str = day_time.strftime("%Y%m%d")
day_folder_name = obj.ensure_folder(folder_name=date_str, parent_folder_id=image_folder_name)



#------------------------------ Connect Post processing logic step -----------------------------
# tracker + smoother
tracker = SimpleTracker(iou_thresh=iou_thresh, max_age=max_age, log_dir=log_dir, save_log=False)
# minute summary  
agg = MinuteAggregator(report_dir=report_dir, ng_threshold_minutes=eq_threshold, alert="TEAMS", url=eq_url, save_log=True)
# Create flag for saving NG case and flag for uploading data
Flag = False
last_uploaded_hour = None
stop_flag = False


#----------------------------- Main loop -------------------------------------------------------
# Handle prediction data via datachannel
@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    frame_time = datetime.now().astimezone()   
    frame_id = int(metadata.frame_id)
    global Flag, last_uploaded_hour
    
    # Tracking + State Smoothing
    active_pids, lost_pids, new_pids = tracker.update(frame_id, data, frame_time=frame_time)

    # Change flag if having New track: Save the first image.
    if new_pids:
        Flag = True

    # Flush the previous mininute: Save the image 
    ng_rows = agg.ingest_rows(active_pids)
    if ng_rows:
        Flag = True
        # Save the log if there are NG person
        #for row in ng_rows:
            #save_row_csv(row.copy(), out_dir="summary/Images")

    # Flush the overdue track: 
    for pid in lost_pids:
        minute = agg.last_minute_of_person.get(pid)
        if minute is not None:
            # NOTE: Force flush overdue track
            is_ng = agg._flush(pid, minute)

    # Upload the log summary to cloud
    current_hour = frame_time.strftime("%Y-%m-%d_%H")
    current_day = frame_time.strftime("%Y-%m-%d")
    csv_file_path = _build_path_for_csv(subdir=report_dir, name=current_day)
    if last_uploaded_hour is None:
        last_uploaded_hour = current_hour
    elif current_hour != last_uploaded_hour:
        obj.upload_or_overwrite(
            filepath=csv_file_path,
            parent_folder_id=csv_folder_name,
            mime_type="text/csv"
        )
        last_uploaded_hour = current_hour


# Handle incoming video frames
@session.on_frame
def show_frame(frame, metadata):
    global stop_flag, Flag
    frame_time = datetime.now().astimezone()   
    # "The image is an illustrative snapshot; the logical timestamp is in the CSV file."
    # Accept that the image will be saved later than expected, by about 1 minute.
    if Flag:
        image_path = save_snapshot(frame, frame_time, out_dir="summary/Images")
        Flag = False
        #obj.FileUpload(filepath=image_path, parent_folder_id=day_folder_name)
        threading.Thread(target=upload_async, args=(image_path,day_folder_name), daemon=True).start()
    
    cv2.imshow("Workflow Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_flag = True

try:
    session.run()
    while not stop_flag:
        time.sleep(0.05)

finally:
    session.close()
    cv2.destroyAllWindows()
