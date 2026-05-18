# 1. Import the InferencePipeline library
from inference import InferencePipeline
#import cv2

def my_sink(result, video_frame):
    print(result["image"])
    #if result.get("image"): # Display an image from the workflow response
        #cv2.imshow("Workflow Image", result["image"].numpy_image)
        #cv2.waitKey(1)
    # Do something with the predictions of each frame

# 2. Initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="API_key",
    workspace_name="kawadard",
    workflow_id="check",
    video_reference="rtsp://localhost:8554/mystream", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)

# 3. Start the pipeline and wait for it to finish
pipeline.start()
pipeline.join()

