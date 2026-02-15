"""WebSocket endpoint for Thor data ingestion.

Handles the /ws/thor endpoint. Receives msgpack binary messages containing
frame_jpeg (bytes) + ego (dict) and pushes them into the frame buffer.
"""

import logging

import msgpack
from fastapi import WebSocket, WebSocketDisconnect

from .frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)


async def thor_websocket(ws: WebSocket, frame_buffer: FrameBuffer):
    """Handle Thor WebSocket connection at /ws/thor.

    Expects msgpack binary messages with:
        frame_jpeg: bytes — JPEG-encoded camera frame
        ego: dict — {timestamp, orientation_ned, velocity_device}
    """
    await ws.accept()
    logger.info("Thor WebSocket connected from %s", ws.client)

    try:
        while True:
            data = await ws.receive_bytes()
            msg = msgpack.unpackb(data, raw=False)

            frame_jpeg = msg.get("frame_jpeg")
            ego = msg.get("ego")

            if frame_jpeg is None:
                continue

            frame_buffer.push(frame_jpeg, ego)

    except WebSocketDisconnect:
        logger.info("Thor WebSocket disconnected.")
    except Exception:
        logger.exception("Error in Thor WebSocket handler")


async def mobile_video_websocket(ws: WebSocket, frame_buffer: FrameBuffer):
    """Relay JPEG frames to mobile clients at /ws/mobile/video.

    Sends the latest JPEG frame whenever available, at roughly MOBILE_VIDEO_FPS.
    """
    import asyncio
    from . import config

    await ws.accept()
    logger.info("Mobile video WebSocket connected from %s", ws.client)
    interval = 1.0 / config.MOBILE_VIDEO_FPS

    try:
        while True:
            jpeg = frame_buffer.latest_jpeg
            if jpeg is not None:
                await ws.send_bytes(jpeg)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        logger.info("Mobile video WebSocket disconnected.")
    except Exception:
        logger.exception("Error in mobile video WebSocket handler")
