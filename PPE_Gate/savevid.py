import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
from datetime import datetime
import time
Gst.init(None)
class RTSPReader:
    def __init__(self, rtsp):
        pipeline = f"""
        rtspsrc location={rtsp} latency=100 protocols=tcp !
        rtph265depay !
        h265parse !
        nvv4l2decoder !
        nvvidconv left=280 right=1000 top=0 bottom=720 !
        video/x-raw,width=640,height=640,format=RGBA !
        videoconvert !
        video/x-raw,format=BGR !
        appsink name=sink drop=1 max-buffers=1
        """
        self.pipeline = Gst.parse_launch(pipeline)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.set_property("emit-signals", True)
        self.pipeline.set_state(Gst.State.PLAYING)
    def read(self):
        sample = self.appsink.emit("pull-sample")
        if sample is None:
            return False, None
        buf = sample.get_buffer()
        caps = sample.get_caps()
        height = caps.get_structure(0).get_value("height")
        width = caps.get_structure(0).get_value("width")
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return False, None
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        buf.unmap(mapinfo)
        return True, frame

# ========================
# MAIN: read + save video
# ========================
"""rtsp_url = "rtsp://169.254.150.5:8554/preview"
#rtsp_url = "rtsp://localhost:8554/mystream"
reader = RTSPReader(rtsp_url)
# ✅ VideoWriter
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('ng_ld.mp4', fourcc, 25, (640, 640))
fps_counter = 0
start_time = time.time()
while True:
    ret, frame = reader.read()
    if not ret:
        continue
    #out.write(frame)
    cv2.imshow("RTSP", frame)
    fps_counter += 1
    if fps_counter == 10:
        elapsed = time.time() - start_time
        fps = fps_counter / elapsed
        print(f"Average FPS (10 frames): {fps:.2f}")
        fps_counter = 0
        start_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#out.release()
cv2.destroyAllWindows()
"""