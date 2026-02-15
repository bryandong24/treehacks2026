"""Lock-free shared memory ring buffer for camera frames.

Used by holoscan_frame_publisher.py (writer, inside Docker) and
jetson_camerad.py (reader, on host) to pass NV12 frames via /dev/shm.

Layout per buffer:
  Header (64 bytes):
    [0:4]   write_index  (uint32) - slot last written (0..NUM_SLOTS-1)
    [4:8]   sequence     (uint32) - incremented before+after write (odd=writing)
    [8:16]  timestamp_ns (uint64) - monotonic timestamp of last frame
    [16:20] width        (uint32)
    [20:24] height       (uint32)
    [24:28] frame_count  (uint32) - total frames written
    [28:64] reserved

  Frame slots (NUM_SLOTS per buffer):
    Each slot is nv12_size = width * height * 3 // 2 bytes
"""

import mmap
import os
import struct
import time

import numpy as np

HEADER_SIZE = 64
NUM_SLOTS = 4
HEADER_FMT = '<IIQIIi'  # write_index, sequence, timestamp_ns, width, height, frame_count
HEADER_STRUCT_SIZE = struct.calcsize(HEADER_FMT)  # 26 bytes, padded to 64

SHM_PATH_ROAD = '/dev/shm/sunnypilot_cam_road'
SHM_PATH_WIDE = '/dev/shm/sunnypilot_cam_wide'


def nv12_size(width, height):
    return width * height * 3 // 2


def total_buffer_size(width, height):
    return HEADER_SIZE + NUM_SLOTS * nv12_size(width, height)


class ShmRingBufferWriter:
    """Writes NV12 frames to a /dev/shm ring buffer."""

    def __init__(self, path, width, height):
        self.path = path
        self.width = width
        self.height = height
        self.frame_size = nv12_size(width, height)
        self.buf_size = total_buffer_size(width, height)
        self.frame_count = 0
        self.write_index = 0
        self.sequence = 0

        # Create or open the file
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        os.ftruncate(fd, self.buf_size)
        self.mm = mmap.mmap(fd, self.buf_size)
        os.close(fd)

        # Write initial header
        self._write_header()

    def _write_header(self):
        hdr = struct.pack(HEADER_FMT,
                          self.write_index,
                          self.sequence,
                          0,  # timestamp
                          self.width,
                          self.height,
                          self.frame_count)
        self.mm[:HEADER_STRUCT_SIZE] = hdr

    def write(self, nv12_data, timestamp_ns=None):
        """Write an NV12 frame to the next slot.

        Args:
            nv12_data: numpy uint8 array or bytes of length frame_size
            timestamp_ns: monotonic timestamp in nanoseconds (auto if None)
        """
        if timestamp_ns is None:
            timestamp_ns = int(time.monotonic_ns())

        if isinstance(nv12_data, np.ndarray):
            nv12_bytes = nv12_data.tobytes()
        else:
            nv12_bytes = nv12_data

        assert len(nv12_bytes) == self.frame_size, \
            f"Expected {self.frame_size} bytes, got {len(nv12_bytes)}"

        # Advance to next slot
        slot = (self.write_index + 1) % NUM_SLOTS

        # Increment sequence to odd (signals "writing in progress")
        self.sequence += 1
        self._write_header_fast(slot, self.sequence, timestamp_ns)

        # Write frame data
        offset = HEADER_SIZE + slot * self.frame_size
        self.mm[offset:offset + self.frame_size] = nv12_bytes

        # Increment sequence to even (signals "write complete")
        self.sequence += 1
        self.write_index = slot
        self.frame_count += 1
        self._write_header_fast(slot, self.sequence, timestamp_ns)

    def _write_header_fast(self, write_index, sequence, timestamp_ns):
        hdr = struct.pack(HEADER_FMT,
                          write_index,
                          sequence,
                          timestamp_ns,
                          self.width,
                          self.height,
                          self.frame_count)
        self.mm[:HEADER_STRUCT_SIZE] = hdr

    def close(self):
        self.mm.close()


class ShmRingBufferReader:
    """Reads latest NV12 frame from a /dev/shm ring buffer."""

    def __init__(self, path):
        self.path = path
        self.mm = None
        self.width = 0
        self.height = 0
        self.frame_size = 0
        self.last_frame_count = -1

    def connect(self, timeout=5.0):
        """Wait for the shm file to appear and read header.

        Returns True if connected, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(self.path):
                try:
                    fd = os.open(self.path, os.O_RDONLY)
                    stat = os.fstat(fd)
                    if stat.st_size >= HEADER_SIZE:
                        self.mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                        os.close(fd)
                        self._read_header()
                        if self.width > 0 and self.height > 0:
                            return True
                        self.mm.close()
                        self.mm = None
                    else:
                        os.close(fd)
                except OSError:
                    pass
            time.sleep(0.05)
        return False

    def _read_header(self):
        raw = self.mm[:HEADER_STRUCT_SIZE]
        vals = struct.unpack(HEADER_FMT, raw)
        self.width = vals[3]
        self.height = vals[4]
        self.frame_size = nv12_size(self.width, self.height)

    def read_latest(self):
        """Read the most recent complete frame.

        Returns (nv12_data, timestamp_ns, frame_count) or (None, 0, 0) if
        no new frame or a torn read is detected.
        """
        if self.mm is None:
            return None, 0, 0

        # Read header
        raw = self.mm[:HEADER_STRUCT_SIZE]
        vals = struct.unpack(HEADER_FMT, raw)
        write_index, seq_before, timestamp_ns, w, h, frame_count = vals

        # No new frame
        if frame_count == self.last_frame_count:
            return None, 0, 0

        # Odd sequence means writer is mid-write — skip
        if seq_before % 2 != 0:
            return None, 0, 0

        # Read frame data
        offset = HEADER_SIZE + write_index * self.frame_size
        frame_data = self.mm[offset:offset + self.frame_size]

        # Re-read sequence to detect torn read
        raw2 = self.mm[:HEADER_STRUCT_SIZE]
        vals2 = struct.unpack(HEADER_FMT, raw2)
        seq_after = vals2[1]

        if seq_after != seq_before:
            # Writer overwrote during our read — discard
            return None, 0, 0

        self.last_frame_count = frame_count
        return np.frombuffer(frame_data, dtype=np.uint8).copy(), timestamp_ns, frame_count

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
