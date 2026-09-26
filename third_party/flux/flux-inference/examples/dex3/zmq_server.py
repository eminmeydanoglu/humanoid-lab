"""GPU-side Dex3 ROUTER service; never publishes robot commands."""

import argparse
import hashlib
import ipaddress
import json
import queue
import stat
import threading
import time
from pathlib import Path

import numpy as np
import zmq
from flux_dex3 import protocol
from zmq.auth import load_certificate
from zmq.auth.thread import ThreadAuthenticator

_REQUIRED = ("adapter_config.json", "adapter_model.safetensors", "policy_preprocessor.json",
             "policy_postprocessor.json")


def checkpoint_identity(checkpoint):
    """Require a read-only local adapter and hash all checkpoint-owned processor artifacts."""
    path = Path(checkpoint)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint must be an absolute, existing directory without symlinks")
    path = path.resolve(strict=True)
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("checkpoint directory must be read-only; copy it locally before serving")
    files = [path / name for name in _REQUIRED]
    for processor in ("policy_preprocessor", "policy_postprocessor"):
        config = path / (processor + ".json")
        if config.is_symlink() or not config.is_file():
            raise ValueError("missing processor config: " + config.name)
        manifest = json.loads(config.read_text(encoding="utf-8"))
        steps = manifest.get("steps") if isinstance(manifest, dict) else None
        if not isinstance(steps, list):
            raise ValueError("invalid processor config: " + config.name)
        weights = []
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("invalid processor step: " + config.name)
            filename = step.get("state_file")
            if filename is not None:
                if (not isinstance(filename, str) or Path(filename).name != filename
                        or not filename.startswith(processor + "_step_") or not filename.endswith(".safetensors")):
                    raise ValueError("invalid processor state_file")
                weights.append(path / filename)
        if not weights:
            raise ValueError("missing checkpoint-owned processor weights: " + config.name)
        files.extend(weights)
    files = sorted(set(files))
    digest = hashlib.sha256()
    for file in files:
        if file.is_symlink() or not file.is_file() or file.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise ValueError("missing, writable, or linked checkpoint artifact: " + file.name)
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\x00")
        with file.open("rb") as stream:
            while True:
                data = stream.read(1024 * 1024)
                if not data:
                    break
                digest.update(data)
    return "sha256:" + digest.hexdigest()


def _safe_error(exc):
    return str(exc).encode("ascii", "backslashreplace")[:512].decode("ascii").replace("\x00", "") or "model error"


class Dex3Server:
    """The ROUTER stays on the serving thread; the worker owns the only policy instance."""

    def __init__(self, checkpoint, *, bind_ip="127.0.0.1", port=5557, device="cuda",
                 server_secret_key=None, client_keys_dir=None, model_loader=None, context=None):
        address = ipaddress.ip_address(bind_ip)
        if address.is_unspecified:
            raise ValueError("wildcard bind is forbidden; select an explicit interface IP")
        if not 1 <= port <= 65535:
            raise ValueError("invalid port")
        if not address.is_loopback and (not server_secret_key or not client_keys_dir):
            raise ValueError("non-loopback bind requires CURVE server key and client allowlist")
        self.bind_ip = bind_ip
        self.port = port
        self.checkpoint = checkpoint
        self.device = device
        self.server_secret_key = server_secret_key
        self.client_keys_dir = client_keys_dir
        self.model_loader = model_loader
        self.context = context
        self.status = "LOADING"
        self.identity = ""
        self.error = ""
        self.pending = None
        self.session_id = None
        self.last_seq = -1
        self.seen_sessions = set()
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

    def _worker(self):
        try:
            identity = checkpoint_identity(self.checkpoint)
            if self.model_loader is None:
                from examples.dex3.g1_inference import G1Inference

                model = G1Inference.load(self.checkpoint, device=self.device)
            else:
                model = self.model_loader(self.checkpoint, device=self.device)
            image = np.zeros((192, 256, 3), dtype=np.uint8)
            state = np.zeros(28, dtype=np.float32)
            protocol.encode_predict_reply("warmup", 0, model.predict(image, state, "stack three block"), 0)
            model.reset()
            if checkpoint_identity(self.checkpoint) != identity:
                raise ValueError("checkpoint artifacts changed during load")
            self.results.put(("loaded", identity))
        except Exception as exc:
            self.results.put(("fatal", _safe_error(exc)))
            return
        while not self.stop_event.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                return
            token, request, new_session = job
            try:
                if new_session:
                    model.reset()
                start = time.monotonic()
                actions = model.predict(request["image"], request["state"], request["task"])
                elapsed_ms = (time.monotonic() - start) * 1000
                reply = protocol.encode_predict_reply(request["session_id"], request["seq"], actions, elapsed_ms)
                self.results.put(("prediction", token, reply))
            except Exception as exc:
                self.results.put(("prediction_error", token, _safe_error(exc)))

    def _configure_curve(self, context, socket):
        if ipaddress.ip_address(self.bind_ip).is_loopback:
            return None
        keys = Path(self.client_keys_dir)
        if not keys.is_dir() or not list(keys.glob("*.key")):
            raise ValueError("client certificate allowlist is empty")
        public, secret = load_certificate(self.server_secret_key)
        if secret is None:
            raise ValueError("server certificate must contain a secret key")
        auth = ThreadAuthenticator(context)
        auth.start()
        try:
            auth.configure_curve(domain="*", location=str(keys))
            socket.curve_publickey = public
            socket.curve_secretkey = secret
            socket.curve_server = True
        except Exception:
            auth.stop()
            raise
        return auth

    def _drain_results(self, socket):
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                return
            kind = result[0]
            if kind == "loaded":
                self.identity, self.status = result[1], "READY"
            elif kind == "fatal":
                self.status, self.error = "ERROR", result[1][:512] or "model load failed"
            elif kind in ("prediction", "prediction_error"):
                token = result[1]
                if self.pending is None or self.pending[0] != token:
                    continue
                _, address, session, seq = self.pending
                self.pending = None
                if kind == "prediction":
                    reply = result[2]
                else:
                    self.status, self.error = "ERROR", result[2][:512] or "prediction failed"
                    reply = protocol.encode_error_reply(self.error, session, seq)
                self._send(socket, address, reply)

    @staticmethod
    def _send(socket, address, frames):
        try:
            socket.send_multipart([address] + frames, flags=zmq.DONTWAIT)
        except zmq.ZMQError:
            pass

    def _handle(self, socket, address, frames):
        try:
            request = protocol.decode_request(frames)
        except protocol.ProtocolError as exc:
            self._send(socket, address, protocol.encode_error_reply(str(exc)))
            return
        if request["type"] == "STATUS":
            self._send(socket, address, protocol.encode_status_reply(self.status, self.identity, self.error))
            return
        session, seq = request["session_id"], request["seq"]
        if self.status != "READY":
            self._send(socket, address, protocol.encode_error_reply("model " + self.status, session, seq))
        elif self.pending is not None:
            self._send(socket, address, protocol.encode_error_reply("prediction already pending", session, seq))
        elif session != self.session_id and session in self.seen_sessions:
            self._send(socket, address, protocol.encode_error_reply("stale session", session, seq))
        elif session == self.session_id and seq != self.last_seq + 1:
            self._send(socket, address, protocol.encode_error_reply("duplicate or out-of-order seq", session, seq))
        elif session != self.session_id and seq != 0:
            self._send(socket, address, protocol.encode_error_reply("new session must start at seq 0", session, seq))
        else:
            new_session = session != self.session_id
            self.session_id = session
            self.seen_sessions.add(session)
            self.last_seq = seq
            token = object()
            self.pending = (token, address, session, seq)
            self.jobs.put_nowait((token, request, new_session))

    def serve_forever(self, *, ready_event=None):
        context = self.context or zmq.Context.instance()
        socket = context.socket(zmq.ROUTER)
        auth = None
        try:
            socket.setsockopt(zmq.SNDHWM, 4)
            socket.setsockopt(zmq.RCVHWM, 4)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, 0)
            socket.setsockopt(zmq.RCVTIMEO, 50)
            socket.setsockopt(zmq.MAXMSGSIZE, protocol.MAX_FRAME)
            socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            auth = self._configure_curve(context, socket)
            host = "[{}]".format(self.bind_ip) if ":" in self.bind_ip else self.bind_ip
            socket.bind("tcp://{}:{}".format(host, self.port))
            self.worker = threading.Thread(target=self._worker, name="dex3-policy", daemon=True)
            self.worker.start()
            if ready_event is not None:
                ready_event.set()
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            while not self.stop_event.is_set():
                self._drain_results(socket)
                if socket in dict(poller.poll(50)):
                    try:
                        parts = socket.recv_multipart()
                    except zmq.ZMQError:
                        continue
                    if len(parts) < 2:
                        continue
                    self._handle(socket, parts[0], parts[1:])
        finally:
            self.stop_event.set()
            if self.worker is not None:
                self.worker.join(timeout=2)
            socket.close(linger=0)
            if auth is not None:
                auth.stop()

    def stop(self):
        self.stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="GPU Dex3 ZMQ inference server (no robot commands)")
    parser.add_argument("--checkpoint", required=True, help="absolute read-only local checkpoint directory")
    parser.add_argument("--bind-ip", default="127.0.0.1", help="explicit IP; non-loopback requires CURVE")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-secret-key", help="CURVE server .key_secret certificate")
    parser.add_argument("--client-keys-dir", help="directory of allowlisted client .key certificates")
    args = parser.parse_args()
    Dex3Server(args.checkpoint, bind_ip=args.bind_ip, port=args.port, device=args.device,
               server_secret_key=args.server_secret_key, client_keys_dir=args.client_keys_dir).serve_forever()


if __name__ == "__main__":
    main()
