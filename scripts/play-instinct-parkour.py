#!/usr/bin/env python3
"""Play the shipped Project-Instinct G1 parkour checkpoint in Isaac Sim.

Why this script exists
----------------------
The downloadable "Data & Model" archive ships **ONNX exports only**
(``actor.onnx`` + ``0-depth_encoder.onnx``); there is no ``model_*.pt``.  The
stock task script therefore cannot play it::

    python source/.../tasks/parkour/scripts/play.py \
        --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=/path/to/run

resolves ``--load_run`` through ``get_checkpoint_path``, which lists the run
directory looking for ``model_.*\\.pt`` and raises
``ValueError: No checkpoints in the directory``.  Nothing is wrong with the
checkpoint; the stock script simply has no ONNX playback path
(``--useonnx`` only compares an already-loaded torch policy against ONNX).

This script keeps the stock control flow and adds:
  * the missing ONNX inference path (depth encoder -> 128-D latent -> actor),
  * a scene that is actually watchable (walls off, camera under your control),
  * a second window showing the head camera's depth image,
  * wall-clock pacing plus FPS/RTF reporting.

Run it through ``./dev.sh instinct-parkour`` (see docs/instinct-parkour.md).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# The stock play.py pulls `cli_args` off the repo's scripts/instinct_rl directory.
# Do the same so this script does not depend on the working directory.
_INSTINCTLAB_ROOT = os.environ.get("INSTINCTLAB_ROOT", "/workspace/humanoid-lab/data/src/InstinctLab")
if not os.path.isdir(os.path.join(_INSTINCTLAB_ROOT, "scripts", "instinct_rl")):
    raise SystemExit(
        f"[ERROR] InstinctLab checkout not found at {_INSTINCTLAB_ROOT!r}. "
        "Set INSTINCTLAB_ROOT to the directory containing scripts/instinct_rl."
    )
sys.path.append(os.path.join(_INSTINCTLAB_ROOT, "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser(description="Play an Instinct-RL agent from exported ONNX policies.")
parser.add_argument("--task", type=str, default="Instinct-Parkour-Target-Amp-G1-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--keyboard_control",
    action="store_true",
    default=True,
    help="Drive the robot from the keyboard (default). Use --no-keyboard to let the environment's own "
    "velocity command term drive it instead.",
)
parser.add_argument(
    "--no-keyboard",
    dest="keyboard_control",
    action="store_false",
    help="Let the environment sample its own velocity commands instead of using the keyboard.",
)
parser.add_argument("--keyboard_linvel_step", type=float, default=0.25, help="Linear velocity change per keypress.")
parser.add_argument("--keyboard_angvel", type=float, default=0.5, help="Angular velocity set by F/G.")
parser.add_argument(
    "--keyboard_linvel_max",
    type=float,
    default=1.0,
    help="Clamp for the forward command. The checkpoint's training config samples lin_vel_x in "
    "(0.45, 1.0) m/s with only_positive_lin_vel_x=True, so this is the policy's trained envelope.",
)
parser.add_argument(
    "--keyboard_linvel_min",
    type=float,
    default=0.0,
    help="Clamp for the reverse command. 0.0 by default because the training config never sampled a "
    "negative lin_vel_x; set it (e.g. -0.3) to try reverse at your own risk.",
)
parser.add_argument(
    "--keyboard_latvel_max",
    type=float,
    default=0.0,
    help="Clamp for A/D lateral commands. 0.0 by default because every terrain in the training config "
    "sampled lin_vel_y = (0.0, 0.0); raise it to experiment.",
)
parser.add_argument(
    "--keyboard_angvel_max",
    type=float,
    default=1.0,
    help="Clamp for F/G. The training config samples ang_vel_z in (-1.0, 1.0) rad/s.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--debug", action="store_true", default=False, help="Wait for a debugpy client on :6789.")

parser.add_argument(
    "--scene",
    choices=("play", "train"),
    default="play",
    help="'play' (default) keeps the checkpoint's stairs/obstacles but drops the 5 m boundary walls "
    "and shrinks the grid, so the run is watchable. 'train' uses the raw task scene as-is.",
)
parser.add_argument(
    "--no_depth_window",
    dest="depth_window",
    action="store_false",
    help="Do not open the extra window that shows the head camera's depth image.",
)
parser.add_argument(
    "--camera",
    choices=("free", "follow"),
    default="free",
    help="'free' (default) frames the robot at start-up and then leaves the viewport camera to its own "
    "navigation, so you can orbit, pan and zoom as usual. 'follow' tracks the robot every frame -- which "
    "also overwrites any navigation every frame. Press C in the window to switch between the two.",
)
# AppLauncher declares its own --device; --sim_device is this script's flag and
# wins, because the physics/sensor device is what decides whether the run is
# smooth (see make_fast).  Rendering always stays on the GPU.
parser.add_argument(
    "--sim_device",
    default="cuda:0",
    choices=("cpu", "cuda:0"),
    help="Device for physics and sensors. 'cuda:0' (default) is ~4x faster per step here: the depth "
    "camera is a warp ray-cast kernel, and on the CPU its ~200k rays are the single most expensive "
    "thing in the loop. Use 'cpu' only to compare or when the GPU is needed elsewhere.",
)
parser.add_argument(
    "--realtime",
    action="store_true",
    default=True,
    help="Pace the loop to wall clock so keyboard commands feel right (default).",
)
parser.add_argument("--no_realtime", dest="realtime", action="store_false", help="Run as fast as the machine allows.")
parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds of simulated time (0 = until closed).")
parser.add_argument("--report_period", type=float, default=2.0, help="Seconds between FPS/RTF reports.")
parser.add_argument(
    "--no_tuning",
    action="store_true",
    default=False,
    help="Skip the playback-only scene trimming (mesh_boxes terrain, debug markers, rewards, monitors).",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed.")

cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The policy's only exteroception is a ray-cast depth camera.  Without this the
# sensor is never rendered and the policy would be fed a dead depth buffer.
args_cli.enable_cameras = True

# Whether this run draws the application's UI at all -- locally or streamed to a
# WebRTC client.  AppLauncher forces `headless` on a livestreaming run, so the
# answer has to be taken before the launcher rewrites the flag.
ui_displayed = not args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402
from carb.input import KeyboardEventType  # noqa: E402

from instinct_rl.utils.utils import get_obs_slice  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper  # noqa: E402
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg  # noqa: E402


def make_fast(env_cfg, interactive: bool) -> None:
    """Cut per-step work that does not affect what the policy sees or does.

    ``mesh_boxes`` tiles are layered triangle-mesh obstacles and cost roughly four
    times the raycast time of any other sub-terrain, which matters because the
    depth camera re-raycasts the whole scene every policy step.  The tile is
    swapped for the cheaper discrete-obstacle geometry, but the dict entry keeps
    its name: the velocity command term looks sub-terrains up by name
    (``velocity_ranges`` in parkour_env_cfg.py) and raises if one is missing.

    Debug visualization markers are also disabled -- they are redrawn every frame
    and say nothing about the policy.

    Three managers are then switched off, none of which this script reads:

    * **rewards** (~16 ms/step).  The step's reward return value is discarded
      here, and with keyboard control the episode never times out, so every one
      of the ~40 reward terms was recomputed purely to be thrown away.
    * **unused observation groups** (~1 ms/step).  The task builds ``critic``,
      ``amp_policy`` and ``amp_reference`` groups for AMP training.  ONNX
      playback only consumes ``policy``, and Isaac Lab skips a group whose
      config is ``None``.
    * terminations are deliberately **kept**: they are what resets the robot
      when it falls over, which is behaviour worth watching.

    ``--no_tuning`` turns all of this off and restores the raw training config.
    """
    if not interactive:
        return

    gen = env_cfg.scene.terrain.terrain_generator
    if gen is not None and "mesh_boxes" in gen.sub_terrains and "boxes" in gen.sub_terrains:
        old = gen.sub_terrains["mesh_boxes"]
        gen.sub_terrains["mesh_boxes"] = gen.sub_terrains["boxes"].replace(proportion=old.proportion)
        print("[INFO] perf: 'mesh_boxes' tiles use the cheaper discrete-box geometry")

    # Debug markers: foot contact spheres, the leg volume point cloud and the
    # commanded-velocity arrow.
    for attr in dir(env_cfg.scene):
        entity = getattr(env_cfg.scene, attr, None)
        if hasattr(entity, "debug_vis"):
            entity.debug_vis = False
    if env_cfg.commands is not None:
        for name in list(vars(env_cfg.commands)):
            term = getattr(env_cfg.commands, name, None)
            if hasattr(term, "debug_vis"):
                term.debug_vis = False
    print("[INFO] perf: debug visualization markers disabled")

    # Rewards are computed by the env every step and then discarded by this
    # script; with the episode timeout disabled they have no gameplay effect.
    if env_cfg.rewards is not None:
        env_cfg.rewards = None
        print("[INFO] perf: reward terms disabled (playback discards the reward)")

    # AMP training observation groups, unused by the exported actor.
    dropped = []
    for group in ("critic", "amp_policy", "amp_reference"):
        if getattr(env_cfg.observations, group, None) is not None:
            setattr(env_cfg.observations, group, None)
            dropped.append(group)
    if dropped:
        print(f"[INFO] perf: observation groups disabled: {', '.join(dropped)} (playback reads 'policy')")



def make_watchable(env_cfg, num_envs: int) -> None:
    """Apply the same scene tweaks the registered ``-Play-v0`` task makes.

    The training scene wraps every 8x8 m tile in 5 m walls (``wall_prob`` 0.3 on
    eight of the ten sub-terrains) and generates a 10x20 terrain grid.  For an
    interactive run that is both slow to build and impossible to see through, so
    the walls come off and the grid shrinks -- the stairs, gaps and box fields
    themselves are untouched.
    """
    if env_cfg.scene.terrain is not None and env_cfg.scene.terrain.terrain_generator is not None:
        gen = env_cfg.scene.terrain.terrain_generator
        for sub in gen.sub_terrains.values():
            if getattr(sub, "wall_prob", None) is not None:
                sub.wall_prob = [0.0, 0.0, 0.0, 0.0]
        # Enough columns that the stair/gap/box variety is preserved, but far fewer
        # tiles to generate than the 10x20 training grid.
        gen.num_rows = min(gen.num_rows, 4)
        gen.num_cols = min(gen.num_cols, 10)
        print(f"[INFO] scene: terrain grid -> {gen.num_rows}x{gen.num_cols}, boundary walls disabled")
    if num_envs > 1:
        # With the camera glued to env 0, extra envs only cost frame time.
        print(f"[INFO] scene: {num_envs} envs requested; the viewer shows env 0 only")


def configure_viewer(env_cfg, mode: str) -> None:
    """Frame the robot at start-up, then either track it or hand the camera back.

    Isaac Lab's viewport camera controller re-poses the camera every frame while
    it follows an asset root (``_update_tracking_callback``), so in that mode
    any orbit, pan or zoom is overwritten before the next frame is drawn: the
    camera looks locked.  ``follow`` keeps that behaviour, ``free`` starts from
    the same framing and then leaves the camera alone, which is what makes the
    viewport's own navigation work.
    """
    env_cfg.viewer = ViewerCfg(
        eye=[3.0, 0.0, 1.4],
        lookat=[0.6, 0.0, 0.5],
        origin_type="asset_root" if mode == "follow" else "world",
        asset_name="robot",
    )
    print(
        "[INFO] scene: viewer follows the robot"
        if mode == "follow"
        else "[INFO] scene: viewer starts behind the robot and is then yours to move (press C to follow)"
    )


def depth_to_rgba(depth, near_m: float, far_m: float):
    """One depth frame as an RGBA byte image, near bright.

    The sensor ships depth in metres and the policy sees it clipped to a fixed
    range, so the same clip is applied here: everything past ``far_m`` is the
    darkest value on screen, exactly like it is the largest value the policy
    receives.
    """
    frame = np.asarray(depth, dtype=np.float32)
    while frame.ndim > 2:  # the sensor output carries a channel axis
        frame = frame[..., 0]
    clipped = np.clip((frame - near_m) / (far_m - near_m), 0.0, 1.0)
    gray = ((1.0 - clipped) * 255.0).astype(np.uint8)
    image = np.empty((*gray.shape, 4), dtype=np.uint8)
    image[..., :3] = gray[..., None]
    image[..., 3] = 255
    return image


class DepthWindow:
    """The head camera's depth image, in a window of its own.

    The depth comes from a ray-cast sensor rather than from a render product, so
    there is no USD camera for a viewport window to attach to: the image shown
    here is the sensor's own tensor, pushed through a byte provider.  The widget
    scales the 64x36 frame up instead of resampling it, which keeps the update
    cheap enough to sit in the control loop.
    """

    TITLE = "G1 Head Camera Depth"
    #: The sensor refreshes at 50 Hz, the window does not need to.
    EVERY_STEPS = 4
    #: Display size of the widget; the image itself stays 64x36.
    WIDGET_SCALE = 6
    #: The range the policy's normalization uses: what is dark here is what the
    #: policy treats as far.
    RANGE_M = (0.1, 2.5)

    def __init__(self, env):
        self._env = env
        self._provider = None
        self._window = None
        self._since_update = 0
        self._failed = False

    def update(self) -> None:
        if self._failed:
            return
        self._since_update += 1
        if self._since_update % self.EVERY_STEPS:
            return
        try:
            self._push_frame()
        except Exception as error:  # a small display must not end a driving run
            self._failed = True
            print(f"[INFO] depth window: switched off after {type(error).__name__}: {error}")

    def _push_frame(self) -> None:
        camera = self._env.unwrapped.scene["camera"]
        if "distance_to_image_plane" not in camera.data.output:
            return
        frame = camera.data.output["distance_to_image_plane"][0]
        if hasattr(frame, "detach"):  # the ray-cast camera hands back device tensors
            frame = frame.detach().cpu().numpy()
        image = depth_to_rgba(frame, *self.RANGE_M)
        if self._provider is None:
            self._open(image.shape[1], image.shape[0])
        # A list of bytes is what this provider's API takes (see Isaac Sim's own
        # heightmap importer); 64x36 keeps that conversion negligible.
        self._provider.set_bytes_data(list(image.tobytes()), [image.shape[1], image.shape[0]])

    def _open(self, width: int, height: int) -> None:
        import omni.ui as ui

        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(
            f"{self.TITLE} ({self.RANGE_M[0]}-{self.RANGE_M[1]} m, near bright)",
            width=width * self.WIDGET_SCALE,
            height=height * self.WIDGET_SCALE,
        )
        with self._window.frame:
            ui.ImageWithProvider(
                self._provider,
                width=width * self.WIDGET_SCALE,
                height=height * self.WIDGET_SCALE,
                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
            )
        print(f"[INFO] depth window: {self.TITLE}, {width}x{height} sensor image, nearest pixel upscale")


class OnnxParkourPolicy:
    """Run the exported parkour policy exactly as the real robot does.

    ``instinct_onboard`` feeds the depth encoder the cropped frame stack and
    concatenates its 128-D latent *after* the proprioception block, so the actor
    sees 768 + 128 = 896 floats.  The proprioception block is taken from the
    environment's own observation segments, in the environment's own order, so it
    cannot silently drift from what the policy was trained on.

    One deliberate difference from deployment: onnxruntime is pinned to the CPU
    provider.  Both graphs are tiny (a 3-layer MLP and a small conv net), and
    keeping them off the GPU leaves the whole 24 GB and the copy queues to PhysX
    and the ray-caster camera.
    """

    def __init__(self, run_dir: str, env):
        import onnxruntime as ort

        exported = os.path.join(run_dir, "exported")
        actor_path = os.path.join(exported, "actor.onnx")
        encoder_path = os.path.join(exported, "0-depth_encoder.onnx")
        for path in (actor_path, encoder_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"[ERROR] Missing ONNX export: {path}")
        self.paths = (actor_path, encoder_path)

        self._encoder = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self._actor = ort.InferenceSession(actor_path, providers=["CPUExecutionProvider"])
        self._encoder_input = self._encoder.get_inputs()[0].name
        self._actor_input = self._actor.get_inputs()[0].name

        segments = env.get_obs_segments()
        depth_name = next((n for n in segments if "depth" in n), None)
        if depth_name is None:
            raise RuntimeError("[ERROR] No depth observation term found; this policy requires one.")

        # The env flattens every term into one vector in this order, so these
        # offsets are the real observation layout rather than an assumption.
        print("[INFO] observation segments (environment order):")
        offset = 0
        offsets = {}
        for name, shape in segments.items():
            size = int(np.prod(shape))
            offsets[name] = (offset, offset + size)
            print(f"         [{offset:>5d}:{offset + size:>5d}] {name:<20} {tuple(shape)}")
            offset += size
        print(f"         {'total':<28} {offset}")

        self._depth_shape = tuple(segments[depth_name])
        depth_start, depth_stop = offsets[depth_name]
        proprio_names = [n for n in segments if n != depth_name]
        proprio_size = sum(int(np.prod(segments[n])) for n in proprio_names)

        # The ONNX layout is proprio-then-latent, which is only valid if the depth
        # term is last in the vector.
        if proprio_names and offsets[proprio_names[-1]][1] != proprio_size:
            raise RuntimeError(
                f"[ERROR] Depth term '{depth_name}' is not last in the observation vector, so the ONNX "
                "layout (proprioception first, depth latent last) does not apply."
            )
        self._proprio_slice = slice(0, proprio_size)
        self._depth_slice = slice(depth_start, depth_stop)

        expected = int(self._actor.get_inputs()[0].shape[-1])
        encoder_frames = tuple(int(d) for d in self._encoder.get_inputs()[0].shape[1:])
        latent_size = int(self._encoder.get_outputs()[0].shape[-1])
        self.expected_obs_dim = expected

        print(f"[INFO] depth '{depth_name}' {self._depth_shape} -> encoder {encoder_frames} -> latent {latent_size}")
        print(f"[INFO] actor input {expected} = proprio {proprio_size} + latent {latent_size}")
        for path in self.paths:
            print(f"[INFO]   {os.path.basename(path)}  ({os.path.getsize(path) / 1e6:.2f} MB)")

        if self._depth_shape != encoder_frames:
            raise RuntimeError(
                f"[ERROR] Depth encoder expects frames {encoder_frames} but the environment produces "
                f"{self._depth_shape}. This checkpoint does not match this task configuration."
            )
        if proprio_size + latent_size != expected:
            raise RuntimeError(
                f"[ERROR] Actor expects {expected} inputs but {proprio_size} proprio + {latent_size} latent "
                f"= {proprio_size + latent_size}. This checkpoint does not match this task configuration."
            )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        proprio = obs[:, self._proprio_slice].detach().cpu().numpy()
        depth = obs[:, self._depth_slice].detach().cpu().numpy().reshape((-1, *self._depth_shape))
        latent = self._encoder.run(None, {self._encoder_input: depth})[0]
        net_in = np.concatenate([proprio, latent], axis=1).astype(np.float32, copy=False)
        out = self._actor.run(None, {self._actor_input: net_in})[0]
        return torch.from_numpy(out).to(obs.device)


def main():
    # --load_run / --no_resume / --checkpoint come from cli_args (shared with the
    # stock play.py).  ONNX playback needs --load_run: it names the directory that
    # holds the exported policies.
    if not args_cli.load_run:
        raise SystemExit(
            "[ERROR] --load_run is required: it must point at a checkpoint directory that\n"
            "        contains an 'exported' subdirectory, e.g.\n"
            "        .../checkpoints/parkour_onboard_preview_stair"
        )

    run_dir = os.path.abspath(os.path.expanduser(args_cli.load_run))
    if not os.path.isdir(os.path.join(run_dir, "exported")):
        raise SystemExit(
            f"[ERROR] {run_dir}/exported does not exist.\n"
            "        Pass --load_run pointing at a checkpoint directory such as\n"
            "        .../checkpoints/parkour_onboard_preview_stair"
        )

    if args_cli.debug:
        import debugpy

        debugpy.listen(("0.0.0.0", 6789))
        print("[INFO] waiting for debugger on 0.0.0.0:6789", flush=True)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.sim_device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Validates that the task registers an instinct_rl agent config; the ONNX path
    # does not otherwise consume it.
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    if args_cli.keyboard_control:
        # You are driving it, not measuring it: one robot and no episode timeout.
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1e10
    elif args_cli.num_envs is None:
        # --no-keyboard without an explicit --num_envs would otherwise inherit the
        # task's *training* default of 4096 envs.  That renders as a slideshow
        # (~1 fps) and is never what someone opening a viewer wants, so cap the
        # automatic case.  Passing --num_envs explicitly overrides this.
        env_cfg.scene.num_envs = 8
        print(f"[INFO] scene: --num_envs not given, defaulting to {env_cfg.scene.num_envs} for a watchable run")

    if args_cli.scene == "play":
        make_watchable(env_cfg, env_cfg.scene.num_envs)
    # The camera is the operator's, not the scene's: it applies to both scenes.
    configure_viewer(env_cfg, args_cli.camera)
    make_fast(env_cfg, interactive=args_cli.scene == "play" and not args_cli.no_tuning)

    step_dt = float(env_cfg.decimation * env_cfg.sim.dt)
    print(f"[INFO] task           : {args_cli.task}")
    print(f"[INFO] checkpoint     : {run_dir}")
    print(f"[INFO] num_envs       : {env_cfg.scene.num_envs}")
    print(f"[INFO] episode length : {env_cfg.episode_length_s}s")
    print(f"[INFO] control rate   : sim dt {env_cfg.sim.dt}s x decimation {env_cfg.decimation} -> {1.0 / step_dt:.1f} Hz")
    print(f"[INFO] render interval: {env_cfg.sim.render_interval} physics steps")
    print(f"[INFO] physics/sensors: {args_cli.sim_device}")
    print(f"[INFO] render         : {args_cli.device} (Isaac Kit viewport)")
    print(f"[INFO] pacing         : {'wall-clock (realtime)' if args_cli.realtime else 'unpaced (max speed)'}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = InstinctRlVecEnvWrapper(env)

    # The camera's depth image, on screen while you drive.  It is the sensor's
    # own frame, uncropped: the policy reads a cropped, noised and normalized
    # slice of it, and this window is for seeing where that slice comes from.
    depth_window = DepthWindow(env) if ui_displayed and args_cli.depth_window else None

    policy = OnnxParkourPolicy(run_dir=run_dir, env=env)

    # Keyboard command override: same bindings as the stock play.py, but the
    # repeat count is derived from the real observation shape instead of the
    # stock `shape[0] // 3` arithmetic, which only happens to work for one
    # particular shape convention.
    override_command = torch.zeros(env.num_envs, 3, device=env.device)
    command_slice, command_shape = get_obs_slice(env.get_obs_segments(), "velocity_commands")
    command_frames = int(np.prod(command_shape)) // 3
    print(f"[INFO] velocity_commands obs: slice {command_slice} shape {tuple(command_shape)} -> {command_frames} frames")

    def toggle_camera() -> None:
        """Give the viewport camera to the operator, or back to the tracker.

        The camera controller holds its own copy of the viewer config, and its
        per-frame callback re-poses the camera whenever that copy says
        ``asset_root``.  Flipping the mode on that copy is therefore what stops
        -- and resumes -- the tracking, and it leaves the camera exactly where it
        is at the moment of the flip, so nothing jumps.
        """
        controller = getattr(env.unwrapped, "viewport_camera_controller", None)
        if controller is None:
            print("[INFO] camera: this run has no viewport camera to move")
            return
        follow = controller.cfg.origin_type != "asset_root"
        controller.cfg.origin_type = "asset_root" if follow else "world"
        print(
            "[INFO] camera: following the robot (press C for free navigation)"
            if follow
            else "[INFO] camera: free -- the viewport's own navigation is yours (press C to follow)"
        )

    def on_keyboard_input(event):
        pressed = event.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT)
        if not pressed:
            return
        if event.input == carb.input.KeyboardInput.C:
            # A camera toggle is not a command: it must not touch the clamps.
            toggle_camera()
            return
        if event.input == carb.input.KeyboardInput.W:
            override_command[:, 0] += args_cli.keyboard_linvel_step
        elif event.input == carb.input.KeyboardInput.S:
            override_command[:, 0] -= args_cli.keyboard_linvel_step
        elif event.input == carb.input.KeyboardInput.A:
            override_command[:, 1] += args_cli.keyboard_linvel_step
        elif event.input == carb.input.KeyboardInput.D:
            override_command[:, 1] -= args_cli.keyboard_linvel_step
        elif event.input == carb.input.KeyboardInput.F:
            override_command[:, 2] = args_cli.keyboard_angvel
        elif event.input == carb.input.KeyboardInput.G:
            override_command[:, 2] = -args_cli.keyboard_angvel
        elif event.input == carb.input.KeyboardInput.X:
            override_command[:] = 0.0
            return

        # Hold the command inside the envelope the checkpoint was trained on.  The
        # stock play.py accumulates without a bound, and KEY_REPEAT fires while a
        # key is held, so a second of held W walks the command far outside the
        # training range (39 m/s has been observed) and the policy degrades.
        override_command[:, 0].clamp_(args_cli.keyboard_linvel_min, args_cli.keyboard_linvel_max)
        override_command[:, 1].clamp_(-args_cli.keyboard_latvel_max, args_cli.keyboard_latvel_max)
        override_command[:, 2].clamp_(-args_cli.keyboard_angvel_max, args_cli.keyboard_angvel_max)

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    obs, _ = env.get_observations()
    policy_step_dt = float(env.unwrapped.step_dt)

    timestep = 0
    sim_time = 0.0
    wall_start = time.perf_counter()
    last_report = wall_start
    next_deadline = wall_start

    print("\n[INFO] Controls: W forward | S back | A strafe left | D strafe right")
    print("[INFO]           F turn left | G turn right | X stop")
    print("[INFO]           C toggle the camera between free and robot-follow")
    print(
        "[INFO] Depth   : head camera depth window is open (--no_depth_window turns it off)"
        if depth_window is not None
        else "[INFO] Depth   : depth window off"
    )
    print(
        "[INFO] Camera  : free -- the viewport's own navigation moves it"
        if args_cli.camera == "free"
        else "[INFO] Camera  : following the robot (C hands it to you)"
    )
    print(
        f"[INFO] Command envelope: vx [{args_cli.keyboard_linvel_min:+.2f}, {args_cli.keyboard_linvel_max:+.2f}] "
        f"| vy [{-args_cli.keyboard_latvel_max:+.2f}, {args_cli.keyboard_latvel_max:+.2f}] "
        f"| wz [{-args_cli.keyboard_angvel_max:+.2f}, {args_cli.keyboard_angvel_max:+.2f}]"
    )
    print("[INFO] Close the window or press Ctrl+C here to stop.\n")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Writing the command into `obs` must happen inside inference mode:
                # the observation tensor is created by the env under
                # torch.inference_mode(), and torch rejects in-place updates to such
                # tensors from outside it.
                if args_cli.keyboard_control:
                    obs[:, command_slice] = override_command.repeat(1, command_frames)

                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            if depth_window is not None:
                depth_window.update()

            timestep += 1
            sim_time += policy_step_dt

            now = time.perf_counter()
            if args_cli.report_period > 0 and now - last_report >= args_cli.report_period:
                elapsed = now - wall_start
                fps = timestep / elapsed
                rtf = sim_time / elapsed
                # Robot state, so a report tells you whether the policy is actually
                # moving rather than just that the loop is turning.
                pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
                vel = env.unwrapped.scene["robot"].data.root_lin_vel_b[0]
                print(
                    f"[perf] t={sim_time:7.1f}s sim | {fps:6.1f} loops/s | "
                    f"{fps * env.num_envs:7.1f} env-steps/s | RTF {rtf:5.2f}x | cmd "
                    f"[{override_command[0, 0]:+.2f}, {override_command[0, 1]:+.2f}, {override_command[0, 2]:+.2f}] | "
                    f"pos ({pos[0]:+7.2f},{pos[1]:+7.2f},{pos[2]:+5.2f}) | vx {vel[0]:+5.2f}",
                    flush=True,
                )
                last_report = now

            if args_cli.realtime:
                # Pace to wall clock. Rendering already blocks in the kit loop,
                # so this only trims genuine bursts and keeps key response sane.
                next_deadline += policy_step_dt
                slack = next_deadline - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_deadline = time.perf_counter()

            if args_cli.duration > 0 and sim_time >= args_cli.duration:
                print(f"[INFO] reached --duration={args_cli.duration}s of simulated time")
                break
    except KeyboardInterrupt:
        print("\n[INFO] interrupted")

    total_wall = time.perf_counter() - wall_start
    if timestep and total_wall > 0:
        print(
            f"[perf] TOTAL {timestep} policy steps | {total_wall:.1f}s wall | "
            f"{timestep / total_wall:.1f} loops/s | {sim_time / total_wall:.2f}x RTF | {sim_time:.1f}s sim"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
