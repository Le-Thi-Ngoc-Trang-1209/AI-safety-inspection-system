import cv2
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import RTSPSource, StreamConfig, VideoMetadata
import os

os.makedirs("image", exist_ok=True)

# Initialize client
client = InferenceHTTPClient.init(
    api_url="http://localhost:9001",
    api_key="API_key"
)
 
# Configure video source (RTSP stream)
source = RTSPSource("rtsp://localhost:8554/mystream")
 
# Configure streaming options
config = StreamConfig(
    stream_output=["image"],
    data_output=["infor"]      # Get prediction data via datachannel
)
 
# Create streaming session
session = client.webrtc.stream(
    source=source,
    workflow="check",
    workspace="kawadard",
    image_input="image",
    config=config
)
 
# Handle incoming video frames
@session.on_frame
def show_frame(frame, metadata):
    print(frame.shape)
    if metadata.frame_id % 10 == 0:
        cv2.imwrite(f"image/frame_{metadata.frame_id}.jpg", frame)
       
    cv2.imshow("Workflow Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()
    
 
# Handle prediction data via datachannel
@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    print(f"Frame {metadata.frame_id}: {data}")
 
# Run the session (blocks until closed)
session.run()
 
