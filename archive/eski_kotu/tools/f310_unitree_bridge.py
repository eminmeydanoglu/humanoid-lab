#!/usr/bin/env python3
"""Read a Logitech F310 and send Unitree-compatible remote frames safely.

This program never talks DDS and never starts SONIC.  Its TCP peer must be a
loopback-only receiver on the G1 reached through an SSH local-forward.  Frames
are unarmed by default; the receiver may observe them, but must not let them
drive a robot until ``--arm-forwarding`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import glob
import os
import select
import socket
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

JS_EVENT = struct.Struct("<IhBB")
FRAME = struct.Struct("!4sBBHQ40s")
MAGIC = b"F31B"
VERSION = 1
FLAG_ARMED = 0x01
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
BUMPER_CHORD_DELAY_SECONDS = 0.12


def clamp_unit(value: int) -> float:
    return max(-1.0, min(1.0, value / 32767.0))


def find_f310() -> str:
    for node in sorted(glob.glob("/sys/class/input/js*")):
        name_file = os.path.join(node, "device", "name")
        try:
            with open(name_file, encoding="utf-8") as handle:
                if "Gamepad F310" in handle.read():
                    return f"/dev/input/{os.path.basename(node)}"
        except OSError:
            continue
    raise FileNotFoundError("Logitech Gamepad F310 was not found under /sys/class/input")


@dataclass
class F310State:
    """Linux joystick ABI state with the F310 XInput mapping used by this host."""

    axes: dict[int, int] = field(default_factory=dict)
    buttons: dict[int, int] = field(default_factory=dict)
    _bumper_pending_since: float | None = field(default=None, init=False, repr=False)
    _bumper_chord_consumed: bool = field(default=False, init=False, repr=False)

    def update(self, event_type: int, number: int, value: int) -> None:
        event_type &= ~JS_EVENT_INIT
        if event_type == JS_EVENT_AXIS:
            self.axes[number] = value
        elif event_type == JS_EVENT_BUTTON:
            self.buttons[number] = int(bool(value))

    def remote_payload(self, now: float | None = None, *, advance: bool = True) -> bytes:
        """Return the 40-byte Unitree REMOTE_DATA_RX layout used by SONIC."""
        if now is None:
            now = time.monotonic()
        raw = bytearray(40)
        raw[0:2] = b"\xfe\xef"  # Informational packet header; SONIC ignores it.

        # Unitree bit ordering: R1, L1, Start, Select, R2, L2, F1, F2, A..Y, D-pad.
        lb, rb, f1_chord = self._bumper_mapping(now, advance=advance)
        bit = 0
        bit |= rb << 0  # RB -> R1
        bit |= lb << 1  # LB -> L1
        bit |= self.buttons.get(7, 0) << 2  # Start
        bit |= self.buttons.get(6, 0) << 3  # Back -> Select / emergency exit
        bit |= int(self._trigger(5) > 0.5) << 4  # RT -> R2
        bit |= int(self._trigger(2) > 0.5) << 5  # LT -> L2
        # This F310 exposes no Guide/Mode event to Linux.  LB+RB is therefore
        # a deliberate F1 chord; _bumper_mapping suppresses its L1/R1 output.
        bit |= int(bool(self.buttons.get(8, 0)) or f1_chord) << 6
        bit |= self.buttons.get(0, 0) << 8  # A
        bit |= self.buttons.get(1, 0) << 9  # B
        bit |= self.buttons.get(2, 0) << 10  # X
        bit |= self.buttons.get(3, 0) << 11  # Y
        hat_x, hat_y = self.axes.get(6, 0), self.axes.get(7, 0)
        bit |= int(hat_y < -16000) << 12  # up
        bit |= int(hat_x > 16000) << 13  # right
        bit |= int(hat_y > 16000) << 14  # down
        bit |= int(hat_x < -16000) << 15  # left
        struct.pack_into("<H", raw, 2, bit)

        # F310 up is negative on Linux; Unitree's forward convention is positive.
        struct.pack_into("<f", raw, 4, clamp_unit(self.axes.get(0, 0)))
        struct.pack_into("<f", raw, 8, clamp_unit(self.axes.get(3, 0)))
        struct.pack_into("<f", raw, 12, -clamp_unit(self.axes.get(4, 0)))
        struct.pack_into("<f", raw, 16, self._trigger(2))
        struct.pack_into("<f", raw, 20, -clamp_unit(self.axes.get(1, 0)))
        return bytes(raw)

    def _trigger(self, axis: int) -> float:
        # F310 XInput trigger axes rest at -32767 and reach +32767 when held.
        return max(0.0, min(1.0, (self.axes.get(axis, -32767) + 32767) / 65534.0))

    def _bumper_mapping(self, now: float, *, advance: bool) -> tuple[int, int, int]:
        """Map bumpers while reserving a safe, event-free F1 chord.

        A single bumper is withheld briefly before becoming L1/R1.  If both
        arrive in that window they produce only F1.  Once the chord is used,
        neither bumper is forwarded again until both have been released; this
        prevents a chord release from accidentally changing a SONIC mode.
        """
        lb, rb = int(bool(self.buttons.get(4, 0))), int(bool(self.buttons.get(5, 0)))
        if not lb and not rb:
            if advance:
                self._bumper_pending_since = None
                self._bumper_chord_consumed = False
            return 0, 0, 0
        if self._bumper_chord_consumed:
            return 0, 0, 0
        if lb and rb:
            if advance:
                self._bumper_chord_consumed = True
                self._bumper_pending_since = None
            return 0, 0, 1
        if self._bumper_pending_since is None:
            if advance:
                self._bumper_pending_since = now
            return 0, 0, 0
        if now - self._bumper_pending_since < BUMPER_CHORD_DELAY_SECONDS:
            return 0, 0, 0
        return lb, rb, 0

    def summary(self) -> str:
        # Rendering diagnostics must not consume the one-shot LB+RB -> F1 chord.
        raw = self.remote_payload(advance=False)
        buttons = struct.unpack_from("<H", raw, 2)[0]
        lx, rx, ry, l2, ly = struct.unpack_from("<5f", raw, 4)
        return (f"buttons=0x{buttons:04x} lx={lx:+.3f} ly={ly:+.3f} "
                f"rx={rx:+.3f} ry={ry:+.3f} l2={l2:.3f} mapping_ready")


def iter_events(fd: int) -> Iterable[tuple[int, int, int]]:
    while True:
        try:
            data = os.read(fd, JS_EVENT.size)
        except BlockingIOError:
            return
        if not data:
            raise OSError("F310 device was disconnected")
        if len(data) != JS_EVENT.size:
            continue
        _, value, event_type, number = JS_EVENT.unpack(data)
        yield event_type, number, value


class Forwarder:
    def __init__(self, host: str, port: int) -> None:
        self.address = (host, port)
        self.sock: socket.socket | None = None
        self.last_attempt = 0.0

    def send(self, frame: bytes) -> bool:
        if self.sock is None and time.monotonic() - self.last_attempt >= 1.0:
            self.last_attempt = time.monotonic()
            try:
                candidate = socket.create_connection(self.address, timeout=0.4)
                candidate.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock = candidate
                print(f"connected to SSH-forwarded receiver at {self.address[0]}:{self.address[1]}")
            except OSError:
                return False
        if self.sock is None:
            return False
        try:
            self.sock.sendall(frame)
            return True
        except OSError:
            self.sock.close()
            self.sock = None
            return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="F310 joystick node; default: auto-detect")
    parser.add_argument("--inspect", action="store_true", help="print raw Linux events only")
    parser.add_argument("--send", action="store_true", help="send frames to a loopback SSH forward")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=49051)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--arm-forwarding", action="store_true",
                        help="mark frames armed; reserve for an explicitly approved motor test")
    args = parser.parse_args()
    if args.hz <= 0 or args.hz > 200:
        parser.error("--hz must be in (0, 200]")
    if args.arm_forwarding and not args.send:
        parser.error("--arm-forwarding requires --send")

    device = find_f310() if args.device == "auto" else args.device
    fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
    print(f"using {device}; {'RAW INSPECT' if args.inspect else 'bridge'} mode")
    state, forwarder, sequence = F310State(), Forwarder(args.host, args.port), 0
    period = 1.0 / args.hz
    try:
        while True:
            readable, _, _ = select.select([fd], [], [], period)
            if readable:
                for event_type, number, value in iter_events(fd):
                    if args.inspect:
                        print(f"event type=0x{event_type:02x} number={number} value={value}")
                    state.update(event_type, number, value)
                if not args.inspect:
                    print(state.summary())
            if args.send:
                flags = FLAG_ARMED if args.arm_forwarding else 0
                frame = FRAME.pack(MAGIC, VERSION, flags, 0, sequence, state.remote_payload())
                forwarder.send(frame)
                sequence += 1
    except KeyboardInterrupt:
        return 0
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
