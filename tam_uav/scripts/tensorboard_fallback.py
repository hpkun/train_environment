"""Dependency-free TensorBoard scalar event writer (TFRecord + Event protobuf)."""

from __future__ import annotations

import os, struct, time
from pathlib import Path


def _varint(value):
    out = bytearray()
    value = int(value)
    while value > 0x7f:
        out.append((value & 0x7f) | 0x80); value >>= 7
    out.append(value)
    return bytes(out)


def _field_bytes(number, value):
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _crc32c(data):
    crc = 0xffffffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    return (~crc) & 0xffffffff


def _masked_crc(data):
    value = _crc32c(data)
    return ((value >> 15) | (value << 17)) + 0xa282ead8 & 0xffffffff


class FallbackSummaryWriter:
    def __init__(self, log_dir):
        directory = Path(log_dir); directory.mkdir(parents=True, exist_ok=True)
        name = f"events.out.tfevents.{int(time.time())}.{os.getpid()}.fallback"
        self.handle = (directory / name).open("wb")
        self._write(struct.pack("<d", time.time()) + _field_bytes(3, b"brain.Event:2"))

    def _write(self, event):
        length = struct.pack("<Q", len(event))
        self.handle.write(length)
        self.handle.write(struct.pack("<I", _masked_crc(length)))
        self.handle.write(event)
        self.handle.write(struct.pack("<I", _masked_crc(event)))

    def add_scalar(self, tag, value, step):
        summary_value = _field_bytes(1, str(tag).encode()) + b"\x15" + struct.pack("<f", float(value))
        summary = _field_bytes(1, summary_value)
        event = (b"\x09" + struct.pack("<d", time.time()) + b"\x10" + _varint(step)
                 + _field_bytes(5, summary))
        self._write(event)

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()
