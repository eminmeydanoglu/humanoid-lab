"""ZMQ client worker; it never shares a socket with ROS callbacks."""

import queue
import threading
import time

from flux_dex3 import protocol


class NetworkWorker(threading.Thread):
    """Owns the only DEALER socket. ROS callbacks only communicate through queues."""

    def __init__(self, endpoint, events, timeout_s, client_public, client_secret, server_public):
        super().__init__(name="dex3-zmq", daemon=True)
        self.endpoint, self.events, self.timeout_s = endpoint, events, timeout_s
        self.client_public, self.client_secret, self.server_public = client_public, client_secret, server_public
        self.commands = queue.Queue(maxsize=2)
        self.closing = threading.Event()

    def submit(self, session, seq, stamp, prompt, image, state):
        self.commands.put_nowait((session, seq, stamp, prompt, image, state))

    def close(self):
        self.closing.set()

    def run(self):
        import zmq

        context = zmq.Context()

        def make_socket():
            sock = context.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, 2)
            sock.setsockopt(zmq.RCVHWM, 2)
            sock.setsockopt(zmq.SNDTIMEO, 0)
            if self.server_public:
                sock.curve_publickey = self.client_public.encode("ascii")
                sock.curve_secretkey = self.client_secret.encode("ascii")
                sock.curve_serverkey = self.server_public.encode("ascii")
            sock.connect(self.endpoint)
            return sock

        socket = make_socket()
        pending = None
        deadline = 0.0
        status_at = 0.0
        try:
            while not self.closing.is_set():
                now = time.monotonic()
                if pending is None:
                    try:
                        request = self.commands.get_nowait()
                    except queue.Empty:
                        request = None
                    if request is not None:
                        session, seq, stamp, prompt, image, state = request
                        try:
                            frames = protocol.encode_predict_request(session, seq, stamp, prompt, image, state)
                            socket.send_multipart(frames)
                            pending = ("PREDICT", session, seq, stamp, now)
                            deadline = now + self.timeout_s
                        except (ValueError, zmq.ZMQError) as exc:
                            self.events.put(("error", session, seq, str(exc)))
                    elif now >= status_at:
                        status_at = now + 1.0
                        try:
                            socket.send_multipart(protocol.encode_status_request())
                        except zmq.ZMQError:
                            self.events.put(("status", {"status": "ERROR", "error": "server unavailable", "checkpoint": ""}))
                        else:
                            pending = ("STATUS", None, None, None, now)
                            deadline = now + self.timeout_s
                try:
                    if socket.poll(50, zmq.POLLIN):
                        reply = protocol.decode_reply(socket.recv_multipart())
                        if pending is None:
                            continue
                        kind, session, seq, stamp, sent = pending
                        pending = None
                        if reply["type"] == "ERROR":
                            self.events.put(("error", session, seq, reply["error"]))
                        elif kind == "STATUS" and reply["type"] == "STATUS":
                            self.events.put(("status", reply))
                        elif (kind == "PREDICT" and reply["type"] == "PREDICT" and
                              reply["session_id"] == session and reply["seq"] == seq):
                            self.events.put(("actions", session, seq, stamp, reply["actions"], sent))
                        else:
                            self.events.put(("error", session, seq, "unexpected reply"))
                except (protocol.ProtocolError, zmq.ZMQError) as exc:
                    self.events.put(("error", pending[1] if pending else None,
                                     pending[2] if pending else None, str(exc)))
                    pending = None
                if pending is not None and time.monotonic() >= deadline:
                    kind, session, seq, _, _ = pending
                    if kind == "PREDICT":
                        self.events.put(("error", session, seq, "prediction timed out"))
                    else:
                        self.events.put(("status", {"status": "ERROR", "error": "server unavailable", "checkpoint": ""}))
                    socket.close()
                    socket = make_socket()
                    pending = None
        finally:
            socket.close()
            context.term()

