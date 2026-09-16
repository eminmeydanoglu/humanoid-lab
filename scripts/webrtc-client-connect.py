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
import json
import time
import urllib.error
import urllib.request

import websocket


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


def evaluate(socket, expression: str, counter=[0]):
    counter[0] += 1
    socket.send(
        json.dumps(
            {
                "id": counter[0],
                "method": "Runtime.evaluate",
                "params": {"expression": expression, "returnByValue": True, "awaitPromise": True},
            }
        )
    )
    while True:
        message = json.loads(socket.recv())
        if message.get("id") == counter[0]:
            result = message.get("result", {})
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


def point_at_client(port: int, server: str | None, wait_seconds: float = 30) -> None:
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
    socket.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9223, help="the client's DevTools port")
    parser.add_argument("--server", required=True, help="streaming host to connect to")
    args = parser.parse_args()
    point_at_client(args.port, args.server)


if __name__ == "__main__":
    main()
