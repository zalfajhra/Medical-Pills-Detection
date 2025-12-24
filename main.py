import cv2
import numpy as np
from ultralytics import YOLO

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack
)
from av import VideoFrame

# =========================================================
# FASTAPI INIT
# =========================================================
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================================================
# LOAD YOLO MODEL
# =========================================================
model = YOLO("models/best.pt")

# =========================================================
# WEBRTC VIDEO TRACK WITH YOLO
# =========================================================
class YOLOVideoTrack(VideoStreamTrack):
    """
    Receive webcam frames from browser,
    run YOLO inference,
    send annotated frames back.
    """
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # YOLO inference
        results = model.predict(
            source=img,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        annotated = results[0].plot()

        new_frame = VideoFrame.from_ndarray(
            annotated, format="bgr24"
        )
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

# =========================================================
# WEBRTC SIGNALING
# =========================================================
pcs = set()

@app.post("/offer")
async def offer(sdp: dict):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(YOLOVideoTrack(track))

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp["sdp"], sdp["type"]
        )
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

# =========================================================
# SERVE FRONTEND
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
