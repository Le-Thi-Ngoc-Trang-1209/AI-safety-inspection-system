import depthai as dai

# 1) Tạo pipeline
pipeline = dai.Pipeline()

# 2) Nguồn RGB và node UVC
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)

# ✨ 2) Ép video-output của ISP ra 1280x720 (NV12)
cam.setVideoSize(640, 480)

# angle view
uvc = pipeline.create(dai.node.UVC)
cam.video.link(uvc.input)  # ColorCamera.video là NV12 -> hợp UVC

# 3) Khai báo UVC từ sớm trong BoardConfig
cfg = dai.Device.Config()
cfg.board.uvc = dai.BoardConfig.UVC(640, 480)
cfg.board.uvc.frameType = dai.ImgFrame.Type.NV12
# cfg.board.uvc.cameraName = "OAK-1"  # tùy chọn
 
# 4) Gắn BoardConfig vào pipeline (pre-boot)
pipeline.setBoardConfig(cfg.board)

# 5) Boot thiết bị với pipeline đã có BoardConfig
with dai.Device(pipeline) as device:
    print("UVC đã bật. Mở Zoom/OBS/Camera app và chọn 'DepthAI UVC'...")
    while True:
        pass