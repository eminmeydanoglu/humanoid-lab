#!/usr/bin/env python3
"""Point the Isaac Sim WebRTC client at a streaming host, over its own DevTools port.

NVIDIA's client accepts no server argument: the renderer keeps the address in
``localStorage["server"]`` (default ``127.0.0.1``) and only ever fills the field
from there.  Running the client with ``--remote-debugging-port`` makes that
value reachable, so this script writes it, reloads the page and, if the form is
still waiting, presses Connect.

Needs ``python3-websocket`` (the Ubuntu ``python3-websocket`` package) and a
client started with ``--remote-debugging-port``.
"""

import argparse
import itertools
import json
import time
import urllib.error
import urllib.request

import websocket

#: DevTools request ids have to be unique per connection.
_REQUEST_IDS = itertools.count(1)


def page_socket(port: int, wait_seconds: float) -> str:
    """The DevTools socket of the client's renderer page."""
    deadline = time.time() + wait_seconds
    last_error = "no target"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
                targets = json.load(response)
            pages = [
                target
                for target in targets
                if target.get("type") == "page" and target.get("url", "").endswith("index.html")
            ]
            if pages:
                return pages[0]["webSocketDebuggerUrl"]
            last_error = f"targets without the renderer page: {[t.get('url') for t in targets]}"
        except (urllib.error.URLError, OSError) as error:
            last_error = str(error)
        time.sleep(1)
    raise SystemExit(f"the client's DevTools port {port} never answered ({last_error})")


def open_page(port: int, wait_seconds: float):
    # Chromium refuses DevTools sockets that carry an Origin header.
    return websocket.create_connection(
        page_socket(port, wait_seconds), timeout=20, suppress_origin=True
    )


def devtools(socket, method: str, params: dict | None = None) -> dict:
    """Send one DevTools command and return its ``result`` payload."""
    request_id = next(_REQUEST_IDS)
    socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
    while True:
        message = json.loads(socket.recv())
        if message.get("id") == request_id:
            if "error" in message:
                raise SystemExit(f"{method} failed: {message['error']}")
            return message.get("result", {})


def evaluate(socket, expression: str):
    """Evaluate an expression in the client's page and return its value."""
    result = devtools(
        socket,
        "Runtime.evaluate",
        {"expression": expression, "returnByValue": True, "awaitPromise": True},
    )
    if "exceptionDetails" in result:
        raise SystemExit(f"the client's page raised: {result['exceptionDetails']}")
    return result.get("result", {}).get("value")


def page_state(socket) -> tuple[str | None, str]:
    """(server field value or None, the page's own text)."""
    field = evaluate(socket, "document.querySelector('input')?.value ?? null")
    text = " / ".join(filter(None, (evaluate(socket, "document.body.innerText") or "").splitlines()))
    return field, text


def wait_for_render(socket, timeout: float = 40) -> str:
    """Wait until the renderer has drawn something.

    A freshly started client answers on the DevTools port well before React has
    rendered the form, and a click sent at that moment is simply lost.
    """
    deadline = time.time() + timeout
    text = ""
    while time.time() < deadline:
        text = " / ".join(
            filter(None, (evaluate(socket, "document.body.innerText") or "").splitlines())
        )
        if text:
            return text
        time.sleep(1)
    return text


#: The client renders the incoming stream into a <video> element; its
#: videoWidth stays zero until the first decoded frame arrives.
VIDEO_STATE = """(() => {
    const video = document.querySelector('video');
    if (!video) return {present: false};
    return {present: true, width: video.videoWidth, height: video.videoHeight,
            time: video.currentTime, paused: video.paused};
})()"""


def wait_for_video(socket, timeout: float, label: str = "stream", *, after_time: float = 0.0) -> dict:
    """Wait until the client is decoding, and report what it sees.

    ``after_time`` makes this a liveness check rather than a presence check:
    the decoded frame clock has to move past that ``currentTime``, so a stalled
    stream is not mistaken for a running one just because its dimensions are
    already known.
    """
    deadline = time.time() + timeout
    state: dict = {"present": False}
    while time.time() < deadline:
        state = evaluate(socket, VIDEO_STATE) or {"present": False}
        if state.get("present") and state.get("width"):
            if not after_time or float(state.get("time") or 0.0) > after_time:
                break
        time.sleep(1)
    if not state.get("present") or not state.get("width"):
        raise SystemExit(f"{label}: the client never decoded a frame ({state})")
    if after_time and float(state.get("time") or 0.0) <= after_time:
        raise SystemExit(
            f"{label}: the decoded frame clock did not move past {after_time:.2f}s ({state})"
        )
    return state


def capture(socket, path):
    """Save one decoded frame of the streamed UI.

    This is the frame the viewer is looking at, taken through the client's own
    compositor rather than from the simulator's camera, so it covers the whole
    path: rendered scene, encoded, streamed, decoded and displayed.
    """
    import base64
    import pathlib

    result = devtools(socket, "Page.captureScreenshot", {"format": "png"})
    data = result.get("data")
    if not data:
        raise SystemExit(f"the client returned no screenshot: {result}")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data))
    print(f"captured {path}")
    return path


def point_at_client(
    port: int,
    server: str | None,
    wait_seconds: float = 30,
    *,
    capture_path: str | None = None,
    video_timeout: float = 60.0,
) -> None:
    socket = open_page(port, wait_seconds)
    if server:
        field, _ = page_state(socket)
        if field != server:
            evaluate(socket, f"localStorage.setItem('server', {json.dumps(server)}); 'stored'")
            evaluate(socket, "location.reload(); 'reloading'")
            socket.close()
            time.sleep(4)
            socket = open_page(port, wait_seconds)
        else:
            print(f"the client already points at {server}")

    text = wait_for_render(socket)
    field, _ = page_state(socket)
    print(f"server field: {field!r}")

    if not text:
        raise SystemExit("the client's page never rendered")

    if "Connect" in text:
        clicked = evaluate(
            socket,
            """(() => {
                const buttons = [...document.querySelectorAll('button')];
                const connect = buttons.find(b => /connect/i.test(b.textContent || ''));
                if (!connect) return 'no connect button; buttons: ' + buttons.map(b => b.textContent).join('|');
                connect.click();
                return 'connect pressed';
            })()""",
        )
        print(f"form: {clicked}")
        time.sleep(3)
        text = " / ".join(
            filter(None, (evaluate(socket, "document.body.innerText") or "").splitlines())
        )
    print(f"client now shows: {text[:200]}")
    if capture_path:
        state = wait_for_video(socket, video_timeout)
        print(
            f"client is decoding {state['width']}x{state['height']} at t={state['time']:.2f}s"
        )
        capture(socket, capture_path)
        # A decoded picture is not proof the stream is live: ask for the frame
        # clock to advance past the captured one before calling it running.
        state = wait_for_video(
            socket, 15.0, "after capture", after_time=float(state.get("time") or 0.0)
        )
        print(f"stream is live: clock advanced to t={state['time']:.2f}s, paused={state['paused']}")
    socket.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9223, help="the client's DevTools port")
    parser.add_argument("--server", required=True, help="streaming host to connect to")
    parser.add_argument("--capture", default=None, help="save one decoded frame of the streamed UI to this PNG")
    parser.add_argument("--video-timeout", type=float, default=60.0, help="seconds to wait for the first decoded frame")
    args = parser.parse_args()
    point_at_client(
        args.port, args.server, capture_path=args.capture, video_timeout=args.video_timeout
    )


if __name__ == "__main__":
    main()
