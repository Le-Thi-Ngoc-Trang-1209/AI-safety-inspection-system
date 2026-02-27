# Guidance

I am currently using this code from the Desktop directory (Jetson Orin Nano 8GB with Jetpack 6.2.2, OpenCV 4.10.0 with CUDA). Please make sure to download it to the correct location to avoid any errors.

## Jetpack 6.2.2
Link Youtube: [Tutorial](https://www.youtube.com/watch?v=Ucg5Zqm9ZMk&list=PLXYLzZ3XzIbhh73dPHczlCzDwtO1N42UG&index=3)

## The system OpenCV with CUDA

JetPack 6.2 requires OpenCV 4.10+ for CUDA 12.6 compatibility. You can install it by following these steps:
[opencv/v4.12](https://zenn.dev/ryuya0124/articles/766cbe737eb281)
[opencv/v4.10](https://www.cytron.io/tutorial/build-opencv-with-cuda-support-for-jetson?r=1)

## Inference (Roboflow)

Original instructions are at: [Roboflow/inference](https://inference.roboflow.com/install/jetson/#manually-starting-the-container)

```bash
sudo docker run -d \
  --name inference-server \
  --runtime nvidia \
  --network host \
  --mount type=bind,source=$HOME/Desktop/AI-safety-inspection-system,target=/workspace/code \
  --volume ~/.inference/cache:/tmp:rw \
  --security-opt="no-new-privileges" \
  --cap-drop="ALL" \
  -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]" \
  -e ORT_TENSORRT_ENGINE_CACHE_ENABLE=1 \
  -e ORT_TENSORRT_ENGINE_CACHE_PATH=/tmp/ort_trt_cache \
  -e ORT_TENSORRT_FP16_ENABLE=1 \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e YOLO_CONFIG_DIR=/tmp/ultralytics \
  roboflow/roboflow-inference-server-jetson-6.2.0:latest
```
I installed inference v0.64.8.

## OAK camera

You will need to install `depthai` library using [Depthai](https://pypi.org/project/depthai/2.30.0.0/). We are using this version:

```bash
 pip install depthai==2.30.0.0
```

## RTSP Streaming

Original instructions are at: [OAK example](https://github.com/luxonis/oak-examples/tree/master/gen2-rtsp-streaming).
This example allows you to stream frames using [MediaMTX](https://github.com/bluenviron/mediamtx).

### Installation

You will need to install `ffmpeg` library, as Python script uses it to forward encoded frames to the MediaMTX server.

```bash
# On Ubuntu 24.04 it should install 6.1.1:
sudo apt install ffmpeg
```
Install [MediaMTX](https://github.com/bluenviron/mediamtx).

### Ubuntu challenges

On Ubuntu 22.04 we encountered that `ffmpeg` is available up to only `4.2.2` with the default apt repo, which doesn't recognize the H264 stream correctly. After upgrading to **Ubuntu 24.04**, we were able to install `ffmpeg==6.1.1` and the code works as expected.

### Usage

First, run the MediaMTX server:

```bash
$ ./mediamtx
2024/08/21 15:26:08 INF MediaMTX v1.8.5
2024/08/21 15:26:08 INF configuration loaded from /Users/erik/Downloads/mediamtx_v1.8.5_darwin_arm64/mediamtx.yml
2024/08/21 15:26:08 INF [RTSP] listener opened on :8554 (TCP), :8000 (UDP/RTP), :8001 (UDP/RTCP)
2024/08/21 15:26:08 INF [RTMP] listener opened on :1935
2024/08/21 15:26:08 INF [HLS] listener opened on :8888
2024/08/21 15:26:08 INF [WebRTC] listener opened on :8889 (HTTP), :8189 (ICE/UDP)
2024/08/21 15:26:08 INF [SRT] listener opened on :8890 (UDP)
```

Now let's run the rtsp.py script, which will start publishing H264-encoded stream to the MediaMTX server.

```
python3 rtsp.py
```

### View stream

To see the streamed frames, use a RTSP Client (e.g. VLC Network Stream) with the following link

```
rtsp://localhost:8554/mystream
```

On Ubuntu or Mac OS, you can use `ffplay` (part of the `ffmpeg` library) to preview the stream, which will provide better performance than VLC (400ms latency vs >1sec latency).

```
ffplay -fflags nobuffer -fflags discardcorrupt -flags low_delay -framedrop rtsp://localhost:8554/mystream
```

## Error

Summary: Can not run? No images are displayed? The system killed without showing any errors? It's incompatibility issue.

Environment: Jetson device, JetPack 6.2.2 (system OpenCV built with CUDA v4.10 or v4.12).
Inference v0.64.0 ~ 1.0.1.

Solution:
Uninstall the bundled opencv-python wheel.
Pin NumPy to 1.26.4.

```
pip install numpy==1.26.4
pip uninstall opencv-python
```

Finding
This points to an incompatibility between the pip opencv-python wheel bundled with the Inference package and the system OpenCV with CUDA (JetPack 6.2.2).

# AI-safety-inspection-system
