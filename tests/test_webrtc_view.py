"""Contract tests for watching a streamed run from another machine.

A livestreamed Isaac run has no local window, so the viewer lives on whatever
machine the operator is sitting at.  These tests run without a network: they
cover the parts that broke or would break silently -- the target aliases, the
"is the viewer connected" check (which must read the server side's own socket
table), and the fact that NVIDIA's client only takes its server address from its
renderer's local storage, never from an argument.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "dev.sh"
VIEW = REPO / "scripts" / "webrtc-view.sh"
CONNECT = REPO / "scripts" / "webrtc-client-connect.py"


class DevEntryTests(unittest.TestCase):
    def setUp(self):
        self.dev = DEV.read_text()

    def test_the_command_exists_and_forwards_its_argument(self):
        self.assertIn("  webrtc-view)\n", self.dev)
        arm = self.dev.split("  webrtc-view)\n", 1)[1].split(";;", 1)[0]
        self.assertIn('exec ./scripts/webrtc-view.sh "${@:2}"', arm)

    def test_the_usage_line_names_it(self):
        usage = [line for line in self.dev.splitlines() if line.strip().startswith("echo \"usage:")]
        self.assertTrue(any("webrtc-view" in line for line in usage))


class ViewScriptTests(unittest.TestCase):
    def setUp(self):
        self.view = VIEW.read_text()

    def test_the_default_viewer_is_emin_1_and_raider_is_one_argument_away(self):
        self.assertIn('target="${1:-${ISAAC_VIEW_TARGET:-emin-1}}"', self.view)
        self.assertIn("emin@emin-1", self.view)
        raider = self.view.split("  raider|raider16)", 1)[1].split(";;", 1)[0]
        self.assertIn("ISAAC_VIEW_RAIDER", raider)
        self.assertIn("aksoy-msi-raider-16-max-hx-b2wj", raider)

    def test_the_address_is_this_hosts_livestream_endpoint(self):
        self.assertIn('ISAAC_LIVESTREAM_ENDPOINT', self.view)
        self.assertIn('tailscale ip -4', self.view)
        self.assertIn('VIEW_PORT="${ISAAC_LIVESTREAM_PORT:-49100}"', self.view)

    def test_it_notices_when_nothing_is_streaming_yet(self):
        self.assertIn("nothing is listening on port ${VIEW_PORT} yet", self.view)

    def test_a_connected_viewer_is_read_from_our_own_socket_table(self):
        # `ss` prints the accepted socket as local:PORT peer:PORT, so the viewer
        # is the *peer* of our listening port.  Matching the other order (as the
        # first version did) never matches and reports every run as a failure.
        check = self.view.split("wait_for_connection() {", 1)[1].split("\n}\n", 1)[0]
        self.assertIn('ss -Htn state established', check)
        self.assertIn('grep -F ":${VIEW_PORT} "', check)
        self.assertIn('grep -qF "${remote_address}:"', check)

    def test_an_already_watching_viewer_is_left_alone(self):
        self.assertIn("is already watching", self.view)
        self.assertIn("ISAAC_VIEW_RESTART", self.view)

    def test_the_viewer_side_is_checked_before_launching(self):
        self.assertIn('python3 -c "import websocket"', self.view)
        self.assertIn("install it with: scripts/install-isaac-webrtc-client.sh", self.view)
        self.assertIn("--remote-debugging-port=", self.view)

    def test_it_explains_itself_when_the_viewer_is_unreachable(self):
        self.assertIn("could not reach", self.view)
        self.assertIn("ISAAC_VIEW_RAIDER=user@host", self.view)


class ConnectHelperTests(unittest.TestCase):
    def setUp(self):
        self.helper = CONNECT.read_text()

    def test_the_address_is_written_where_the_client_reads_it(self):
        self.assertIn("localStorage.setItem('server'", self.helper)
        self.assertIn("location.reload()", self.helper)

    def test_it_only_talks_to_the_renderer_page(self):
        self.assertIn('target.get("type") == "page"', self.helper)
        self.assertIn('endswith("index.html")', self.helper)

    def test_chromium_origin_check_is_waived_for_the_devtools_socket(self):
        self.assertIn("suppress_origin=True", self.helper)

    def test_it_waits_for_the_page_before_pressing_connect(self):
        # A click sent while React has not rendered yet is simply lost.
        self.assertIn("def wait_for_render", self.helper)
        self.assertIn("text = wait_for_render(socket)", self.helper)
        self.assertLess(
            self.helper.index("text = wait_for_render(socket)"),
            self.helper.index('if "Connect" in text'),
        )


if __name__ == "__main__":
    unittest.main()
