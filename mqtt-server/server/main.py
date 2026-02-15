"""FastAPI application for the Alpamayo inference server.

Lifespan: loads model, connects MQTT, starts periodic inference loop.
Endpoints:
    /ws/thor          — Thor data ingestion (msgpack WebSocket)
    /ws/mobile/video  — Relay JPEG frames to mobile app
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from . import config
from .frame_buffer import FrameBuffer
from .inference_engine import InferenceEngine
from .mqtt_client import MQTTClient
from .websocket_handler import mobile_video_websocket, thor_websocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Shared state
frame_buffer = FrameBuffer()
inference_engine = InferenceEngine()
mqtt_client = MQTTClient()


async def periodic_inference():
    """Run Alpamayo inference every INFERENCE_INTERVAL_S seconds."""
    while True:
        await asyncio.sleep(config.INFERENCE_INTERVAL_S)
        try:
            snapshot = frame_buffer.get_inference_snapshot()
            if snapshot is None:
                logger.debug("Not enough data for inference yet.")
                continue

            logger.info("Running inference...")
            frames = snapshot["frames"]
            ego_data = {
                "ego_history_xyz": snapshot["ego_history_xyz"],
                "ego_history_rot": snapshot["ego_history_rot"],
            }

            # Run inference in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, inference_engine.run_inference, frames, ego_data
            )

            logger.info("Inference complete. CoC: %s...", result["coc"][:100])

            # Publish results via MQTT
            mqtt_client.publish_coc(result["coc"], result["trajectory_summary"])
            mqtt_client.publish_trajectory(result["pred_xyz"])

        except Exception:
            logger.exception("Inference loop error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model, connect MQTT, start inference loop."""
    inference_engine.load()
    mqtt_client.connect()
    task = asyncio.create_task(periodic_inference())
    logger.info("Server ready. Inference loop started (every %.0fs).", config.INFERENCE_INTERVAL_S)
    yield
    task.cancel()
    mqtt_client.disconnect()


app = FastAPI(title="Alpamayo Inference Server", lifespan=lifespan)


@app.websocket("/ws/thor")
async def ws_thor(ws: WebSocket):
    await thor_websocket(ws, frame_buffer)


@app.websocket("/ws/mobile/video")
async def ws_mobile_video(ws: WebSocket):
    await mobile_video_websocket(ws, frame_buffer)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": inference_engine.model is not None,
        "frames_buffered": len(frame_buffer._frames),
        "ego_samples": frame_buffer.ego.count,
    }
