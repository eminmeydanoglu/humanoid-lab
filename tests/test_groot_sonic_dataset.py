"""The produced pack: writer metadata, then a real convert -> validate round trip.

The metadata tests are pure. The round trip builds a synthetic raw collection and
a synthetic 50 Hz SONIC corpus on disk, runs the two command-line entry points
against them, and then tampers with the produced pack to prove the validator
fails closed. It skips when parquet, video decoding or ffmpeg are unavailable --
the same condition the container does not have.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from humanoid_lab.datasets.groot_sonic.contract import (  # noqa: E402
    CANONICAL_ACTION_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_PATH,
    GRAVITY_FIELD,
    LEFT_HAND_FIELD,
    MOTION_TOKEN_FIELD,
    RIGHT_HAND_FIELD,
    STATE_FIELD,
    load_config,
    official_stats_command,
)
from humanoid_lab.datasets.groot_sonic.convert import ConvertedEpisode, HandRepair, RawEpisode, SonicEpisode  # noqa: E402
from humanoid_lab.datasets.groot_sonic.modality import modality_payload  # noqa: E402
from humanoid_lab.datasets.groot_sonic.validate import (  # noqa: E402
    EXPECTED_MODALITY,
    REGISTRY_KEY,
    REGISTRY_RELATIVE_PATH,
    ValidationError,
    assert_registry_matches_contract,
    invalid_mask,
    parse_registry_entry,
    repaired_measured,
    validate_dataset,
)
from humanoid_lab.datasets.groot_sonic.writer import (  # noqa: E402
    EpisodeRecord,
    feature_specs,
    info_payload,
    repair_totals,
    tasks_rows,
    video_command,
)
from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import UNITREE_ACTION_NAMES  # noqa: E402

CONVERTER = REPO_ROOT / "scripts/groot-sonic-convert.py"
VALIDATOR = REPO_ROOT / "scripts/groot-sonic-validate.py"
STATS_PREP = REPO_ROOT / "scripts/groot-sonic-stats-prep.py"
COLLECTION = "G1_Dex3_PickApple_Dataset"
RAW_FRAMES = 100
CORPUS_ROWS = 167
ROWS = CORPUS_ROWS - 45


def has_video_stack() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    for module in ("pyarrow", "av"):
        try:
            __import__(module)
        except ImportError:
            return False
    return True


class WriterMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_feature_specs_declare_the_schema(self) -> None:
        features = feature_specs(self.config)
        self.assertEqual(features[STATE_FIELD]["shape"], [43])
        self.assertEqual(tuple(features[STATE_FIELD]["names"]), tuple(CANONICAL_STATE_NAMES))
        self.assertEqual(features[GRAVITY_FIELD]["shape"], [3])
        self.assertEqual(features[MOTION_TOKEN_FIELD]["shape"], [64])
        self.assertEqual(features[LEFT_HAND_FIELD]["shape"], [7])
        self.assertEqual(tuple(features[LEFT_HAND_FIELD]["names"]), CANONICAL_ACTION_HAND_NAMES[:7])
        self.assertEqual(tuple(features[RIGHT_HAND_FIELD]["names"]), CANONICAL_ACTION_HAND_NAMES[7:])
        ego = features[self.config.camera_target_key]
        self.assertEqual(ego["dtype"], "video")
        self.assertEqual(ego["shape"], [3, 480, 640])
        self.assertEqual(ego["info"]["video.codec"], "h264")
        self.assertEqual(ego["info"]["video.pix_fmt"], "yuv420p")
        self.assertEqual(ego["info"]["video.fps"], 50)

    def test_only_one_camera_is_declared(self) -> None:
        features = feature_specs(self.config)
        cameras = [key for key, spec in features.items() if spec["dtype"] in ("video", "image")]
        self.assertEqual(cameras, [self.config.camera_target_key])

    def test_info_payload_is_v21(self) -> None:
        info = info_payload(self.config, "train", episodes=3, frames=300)
        self.assertEqual(info["codebase_version"], "v2.1")
        self.assertEqual(info["fps"], 50)
        self.assertEqual(info["total_episodes"], 3)
        self.assertEqual(info["total_frames"], 300)
        self.assertEqual(info["splits"], {"train": "0:3"})
        self.assertEqual(info["data_path"], "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    def test_tasks_rows_follow_the_collection_order(self) -> None:
        rows = tasks_rows(self.config)
        self.assertEqual(len(rows), len(self.config.collections))
        self.assertEqual(rows[0], {"task_index": 0, "task": self.config.tasks[self.config.collection_names[0]]})

    def test_video_command_pins_the_hold_policy_and_the_row_count(self) -> None:
        raw = RawEpisode(
            collection=COLLECTION, episode_index=1, state=np.zeros((2, 28)), action=np.zeros((2, 28)),
            timestamp=np.zeros(2), state_names=UNITREE_ACTION_NAMES, action_names=UNITREE_ACTION_NAMES,
            data_file="data/x.parquet", video_file=Path("/tmp/source.mp4"), video_start_s=1.5, video_stop_s=6.0,
        )
        command = video_command(self.config, raw, ROWS, 1.5, Path("/tmp/out.mp4"))
        self.assertIn("fps=50:round=down", command)
        self.assertEqual(command[command.index("-ss") + 1], "1.500000")
        self.assertEqual(command[command.index("-frames:v") + 1], str(ROWS))
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
        self.assertEqual(command[-1], "/tmp/out.mp4")

    def test_repair_totals_aggregate_per_channel(self) -> None:
        def record(index: int, channels: dict[str, int]) -> EpisodeRecord:
            return EpisodeRecord(
                episode_index=index, length=10, task_index=0, task="t", source_collection=COLLECTION,
                source_episode_index=index, source_data_file="d", source_frames=20, source_fps=30.0,
                corpus_frames=55, trimmed_rows=45, corpus_hand_max_error=0.0, video_start_s=0.0,
                video_stop_s=1.0, parquet=Path("p"), video=Path("v"),
                repair={"channels": channels, "invalid_source_samples": sum(channels.values())},
            )

        totals = repair_totals([record(0, {"left_hand_index_0_joint": 2}), record(1, {"left_hand_index_0_joint": 3})])
        self.assertEqual(totals["invalid_source_samples"], 5)
        self.assertEqual(totals["channels"], {"left_hand_index_0_joint": 5})
        self.assertEqual(totals["episodes_with_repair"], 2)

    def test_writer_and_validator_agree_on_the_modality_box(self) -> None:
        self.assertEqual(modality_payload(self.config), EXPECTED_MODALITY)

    def test_stats_are_not_the_official_file(self) -> None:
        self.assertEqual(self.config.stats_own_file, "stats_groot_sonic.json")
        self.assertEqual(self.config.stats_official_file, "stats.json")
        self.assertNotEqual(self.config.stats_own_file, self.config.stats_official_file)


def registry_source(
    *,
    key: str = REGISTRY_KEY,
    state_keys: tuple[str, ...] = (
        "left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand", "projected_gravity",
    ),
    action_keys: tuple[str, ...] = ("motion_token", "left_hand_joints", "right_hand_joints"),
    action_delta: str = "list(range(40))",
    state_delta: str = "[0]",
    representations: tuple[str, ...] = ("ABSOLUTE", "ABSOLUTE", "ABSOLUTE"),
) -> str:
    """A registry file in the pinned shape, so the check needs no gr00t import."""
    quoted = lambda keys: ", ".join(f'"{value}"' for value in keys)  # noqa: E731
    configs = ", ".join(
        f"ActionConfig(rep=ActionRepresentation.{rep}, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)"
        for rep in representations
    )
    return (
        "from gr00t.data.types import ActionConfig, ActionRepresentation, ActionType, ActionFormat, ModalityConfig\n\n"
        "MODALITY_CONFIGS = {\n"
        '    "another_embodiment": {\n'
        '        "video": ModalityConfig(delta_indices=[0], modality_keys=["webcam"]),\n'
        "    },\n"
        f'    "{key}": {{\n'
        '        "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),\n'
        f'        "state": ModalityConfig(delta_indices={state_delta}, modality_keys=[{quoted(state_keys)}]),\n'
        '        "action": ModalityConfig(\n'
        f"            delta_indices={action_delta},\n"
        f"            modality_keys=[{quoted(action_keys)}],\n"
        f"            action_configs=[{configs}],\n"
        "        ),\n"
        '        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),\n'
        "    },\n"
        "}\n"
    )


class GrootRegistryTest(unittest.TestCase):
    """The pre-registered embodiment config is read from source, not imported."""

    def test_reads_the_pinned_entry_shape(self) -> None:
        entry = parse_registry_entry(registry_source())
        self.assertEqual(entry["video"]["keys"], ("ego_view",))
        self.assertEqual(entry["state"]["keys"][0], "left_leg")
        self.assertEqual(entry["action"]["keys"], ("motion_token", "left_hand_joints", "right_hand_joints"))
        self.assertEqual(entry["action"]["delta_indices"], list(range(40)))
        self.assertEqual(entry["action"]["action_representations"], ("ABSOLUTE", "ABSOLUTE", "ABSOLUTE"))
        self.assertEqual(entry["language"]["keys"], ("annotation.human.task_description",))

    def test_accepts_the_registry_that_describes_this_pack(self) -> None:
        detail = assert_registry_matches_contract(parse_registry_entry(registry_source()), 40)
        self.assertIn("8 state / 3 action keys", detail)
        self.assertIn("horizon 40", detail)

    def test_the_registry_key_is_the_tag_value_not_the_member_name(self) -> None:
        self.assertEqual(REGISTRY_KEY, "unitree_g1_sonic")
        self.assertEqual(REGISTRY_KEY, "UNITREE_G1_SONIC".lower())
        with self.assertRaisesRegex(ValidationError, "no 'unitree_g1_sonic' entry"):
            parse_registry_entry(registry_source(key="UNITREE_G1_SONIC"))

    def test_a_missing_entry_lists_what_is_registered(self) -> None:
        with self.assertRaisesRegex(ValidationError, "another_embodiment"):
            parse_registry_entry(registry_source(key="not_registered"))

    def test_rejects_a_different_state_key_order(self) -> None:
        source = registry_source(
            state_keys=(
                "left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand",
                "projected_gravity",
            )
        )
        with self.assertRaisesRegex(ValidationError, "state keys"):
            assert_registry_matches_contract(parse_registry_entry(source), 40)

    def test_rejects_a_different_action_key(self) -> None:
        source = registry_source(action_keys=("motion_token", "left_hand_joints"))
        with self.assertRaisesRegex(ValidationError, "action keys"):
            assert_registry_matches_contract(parse_registry_entry(source), 40)

    def test_rejects_a_shorter_action_horizon(self) -> None:
        source = registry_source(action_delta="list(range(39))")
        with self.assertRaisesRegex(ValidationError, "horizon"):
            assert_registry_matches_contract(parse_registry_entry(source), 40)

    def test_accepts_a_bare_range_for_the_horizon(self) -> None:
        entry = parse_registry_entry(registry_source(action_delta="range(40)"))
        self.assertEqual(entry["action"]["delta_indices"], list(range(40)))
        assert_registry_matches_contract(entry, 40)

    def test_rejects_a_relative_action_representation(self) -> None:
        source = registry_source(representations=("ABSOLUTE", "RELATIVE", "ABSOLUTE"))
        with self.assertRaisesRegex(ValidationError, "ABSOLUTE"):
            assert_registry_matches_contract(parse_registry_entry(source), 40)

    def test_rejects_a_multi_frame_observation(self) -> None:
        source = registry_source(state_delta="[0, 1]")
        with self.assertRaisesRegex(ValidationError, "current frame only"):
            assert_registry_matches_contract(parse_registry_entry(source), 40)

    def test_refuses_an_expression_it_cannot_read(self) -> None:
        source = registry_source(action_delta="HORIZON")
        with self.assertRaisesRegex(ValidationError, "does not read"):
            parse_registry_entry(source)

    def test_rejects_a_registry_that_does_not_parse(self) -> None:
        with self.assertRaisesRegex(ValidationError, "not parsable"):
            parse_registry_entry("MODALITY_CONFIGS = {  # oops")

    def test_rejects_a_module_without_the_configs_dict(self) -> None:
        with self.assertRaisesRegex(ValidationError, "MODALITY_CONFIGS"):
            parse_registry_entry("CONFIGS = {}\n")


class ValidatorHelperTest(unittest.TestCase):
    def test_invalid_mask_matches_the_declared_threshold(self) -> None:
        samples = np.array([0.5, 3.0, -3.0, 3.0001, np.nan])
        np.testing.assert_array_equal(invalid_mask(samples), [False, False, False, True, True])

    def test_repaired_measured_interpolates_and_reports(self) -> None:
        state = np.zeros((5, 28))
        state[:, 14:28] = np.arange(28, 42) / 100.0
        state[2, 14] = 8.0
        repaired, ledger = repaired_measured(state, UNITREE_ACTION_NAMES, "label")
        self.assertEqual(ledger, {"left_hand_thumb_0_joint": 1})
        self.assertAlmostEqual(repaired[2, 14], 0.28, places=12)
        self.assertEqual(repaired[2, 15], 0.29)

    def test_repaired_measured_refuses_a_boundary_corruption(self) -> None:
        state = np.zeros((5, 28))
        state[0, 27] = np.inf
        with self.assertRaisesRegex(ValidationError, "cannot be repaired"):
            repaired_measured(state, UNITREE_ACTION_NAMES, "label")


def write_parquet(path: Path, columns: dict[str, list]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="snappy")


def encode_source_video(path: Path, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30",
            "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(path),
        ],
        check=True,
    )


def reject_video_frames(path: Path, frames: int) -> None:
    """Re-encode a clip one frame short: the pack must then fail to validate."""
    temporary = path.with_suffix(".partial.mp4")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(temporary),
        ],
        check=True,
    )
    temporary.replace(path)


@unittest.skipUnless(has_video_stack(), "parquet, video decoding and ffmpeg are not available here")
class MiniPackRoundTripTest(unittest.TestCase):
    """Convert a synthetic collection with the real entry points, then validate it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.temporary = tempfile.TemporaryDirectory(prefix="groot-sonic-mini-")
        cls.root = Path(cls.temporary.name)
        cls.raw_root = cls.root / "raw"
        cls.sonic_root = cls.root / "sonic"
        cls.output_root = cls.root / "pack"
        cls.manifest_path = cls.root / "split_manifest.json"
        cls.config_path = cls.root / "contract.yaml"
        cls._write_config()
        cls._write_manifest()
        cls._write_source_fixture()
        cls.convert = subprocess.run(
            [
                sys.executable, str(CONVERTER),
                "--config", str(cls.config_path),
                "--split-manifest", str(cls.manifest_path),
                "--output-root", str(cls.output_root),
                "--collection", COLLECTION,
                "--train-per-task", "1",
                "--val-per-task", "1",
            ],
            capture_output=True, text=True,
        )
        cls.validate = subprocess.run(
            [
                sys.executable, str(VALIDATOR),
                "--config", str(cls.config_path),
                "--split-manifest", str(cls.manifest_path),
                "--root", str(cls.output_root),
                "--quiet",
                "--no-lerobot",
            ],
            capture_output=True, text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    # -- fixture -----------------------------------------------------------
    @classmethod
    def _write_config(cls) -> None:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["source"]["raw_root"] = str(cls.raw_root)
        raw["source"]["sonic_root"] = str(cls.sonic_root)
        raw["split"]["manifest"] = str(cls.manifest_path)
        raw["output"]["root"] = str(cls.output_root)
        cls.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    @classmethod
    def _write_manifest(cls) -> None:
        manifest = {
            "schema_version": 1,
            "name": cls.config.split_name,
            "strategy": "episode_level_stratified_by_collection",
            "seed": 20260917,
            "collections": {
                collection.name: {
                    "total": 3,
                    "usable": 3 - len(collection.excluded),
                    "excluded": list(collection.excluded),
                    "instruction": collection.instruction,
                    "train": [1],
                    "val": [2],
                }
                for collection in cls.config.collections
            },
        }
        cls.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def _episode_block(cls, episode: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
        ramp = np.linspace(0.2, 1.1, frames)
        block = np.outer(ramp, np.arange(1, 29) / 28.0)
        desired = block.copy()
        # A gross corruption in the measured left thumb, absent from the action.
        if episode == 1:
            block[10, 14] = 7.5
            block[11, 14] = np.nan
        return block, desired

    @classmethod
    def _write_source_fixture(cls) -> None:
        dataset = cls.raw_root / COLLECTION
        data_columns: dict[str, list] = {
            "episode_index": [], "timestamp": [],
            "observation.state": [], "action": [],
        }
        meta_columns: dict[str, list] = {
            "episode_index": [], "length": [],
            "data/chunk_index": [], "data/file_index": [],
            "dataset_from_index": [], "dataset_to_index": [],
            "videos/observation.images.cam_left_high/chunk_index": [],
            "videos/observation.images.cam_left_high/file_index": [],
            "videos/observation.images.cam_left_high/from_timestamp": [],
            "videos/observation.images.cam_left_high/to_timestamp": [],
        }
        cursor = 0
        for file_index, episode in enumerate((1, 2)):
            state, action = cls._episode_block(episode, RAW_FRAMES)
            data_columns["episode_index"].extend([episode] * RAW_FRAMES)
            data_columns["timestamp"].extend((np.arange(RAW_FRAMES) / 30.0).tolist())
            data_columns["observation.state"].extend(state.tolist())
            data_columns["action"].extend(action.tolist())
            encode_source_video(
                dataset / f"videos/observation.images.cam_left_high/chunk-000/file-{file_index:03d}.mp4",
                RAW_FRAMES,
            )
            meta_columns["episode_index"].append(episode)
            meta_columns["length"].append(RAW_FRAMES)
            meta_columns["data/chunk_index"].append(0)
            meta_columns["data/file_index"].append(0)
            meta_columns["dataset_from_index"].append(cursor)
            meta_columns["dataset_to_index"].append(cursor + RAW_FRAMES)
            meta_columns["videos/observation.images.cam_left_high/chunk_index"].append(0)
            meta_columns["videos/observation.images.cam_left_high/file_index"].append(file_index)
            meta_columns["videos/observation.images.cam_left_high/from_timestamp"].append(0.0)
            meta_columns["videos/observation.images.cam_left_high/to_timestamp"].append(RAW_FRAMES / 30.0)
            cursor += RAW_FRAMES
        write_parquet(dataset / "data/chunk-000/file-000.parquet", data_columns)
        write_parquet(dataset / "meta/episodes/chunk-000/file-000.parquet", meta_columns)
        info = {
            "codec_version": "v3.0",
            "fps": 30,
            "total_episodes": 2,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [28], "names": [list(UNITREE_ACTION_NAMES)]},
                "action": {"dtype": "float32", "shape": [28], "names": [list(UNITREE_ACTION_NAMES)]},
            },
        }
        (dataset / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")

        for episode in (1, 2):
            state, action = cls._episode_block(episode, RAW_FRAMES)
            timestamps = np.arange(CORPUS_ROWS) / 50.0
            token = np.arange(CORPUS_ROWS * 64, dtype=np.float32).reshape(CORPUS_ROWS, 64) / 1013.0
            # The corpus stores the left hand in the source motor order, both hands
            # interpolated from the same-row desired action.
            hands = np.stack(
                [np.interp(timestamps, np.arange(RAW_FRAMES) / 30.0, action[:, column]) for column in range(14, 28)],
                axis=1,
            ).astype(np.float32)
            mask = np.ones(CORPUS_ROWS, dtype=bool)
            mask[-45:] = False
            directory = cls.sonic_root / COLLECTION / "episodes" / f"episode_{episode:06d}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez(
                directory / "action.npz",
                action=np.concatenate([token, hands], axis=1),
                timestamp=timestamps,
                training_valid_mask=mask,
            )

    # -- assertions --------------------------------------------------------
    def test_conversion_passes(self) -> None:
        self.assertEqual(self.convert.returncode, 0, self.convert.stderr[-2000:])
        self.assertIn("PASS", self.convert.stdout)
        self.assertIn("--embodiment-tag UNITREE_G1_SONIC", self.convert.stdout)

    def test_validation_passes(self) -> None:
        self.assertEqual(self.validate.returncode, 0, self.validate.stdout[-4000:] + self.validate.stderr[-2000:])

    def test_pack_layout(self) -> None:
        for split in ("train", "val"):
            root = self.output_root / split
            for relative in (
                "meta/info.json", "meta/modality.json", "meta/tasks.jsonl", "meta/episodes.jsonl",
                "meta/episodes_stats.jsonl", "meta/stats_groot_sonic.json", "meta/provenance.json",
                "data/chunk-000/episode_000000.parquet",
                "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
            ):
                self.assertTrue((root / relative).is_file(), f"{split}/{relative} is missing")
            self.assertFalse((root / "meta/stats.json").exists(), "the official stats file is not ours to write")

    def test_stored_state_layout_is_the_official_exporter_order(self) -> None:
        """Every stored block must hold the source channel its name claims.

        The fixture's source columns are ``0.2 * (column + 1) / 28`` at frame 0,
        so the stored value names the source column that landed there: the check
        fails if the arms and hands are stored in any other order.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(self.output_root / "val/data/chunk-000/episode_000000.parquet")
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
        self.assertEqual(state.shape[1], 43)
        # Source column 0 is kLeftShoulderPitch, 7 kRightShoulderPitch, 19 kLeftHandIndex0, 24 kRightHandIndex0.
        for index, source_column in ((15, 0), (22, 19), (29, 7), (36, 24)):
            self.assertAlmostEqual(state[0, index], 0.2 * (source_column + 1) / 28.0, places=5, msg=f"index {index}")

        names = json.loads((self.output_root / "val/meta/info.json").read_text(encoding="utf-8"))["features"][
            "observation.state"
        ]["names"]
        self.assertEqual(names[15], "left_shoulder_pitch_joint")
        self.assertEqual(names[22], "left_hand_index_0_joint")
        self.assertEqual(names[29], "right_shoulder_pitch_joint")
        self.assertEqual(names[36], "right_hand_index_0_joint")
        modality = json.loads((self.output_root / "val/meta/modality.json").read_text(encoding="utf-8"))["state"]
        self.assertEqual(modality["right_arm"], {"start": 29, "end": 36})
        self.assertEqual(modality["left_hand"], {"start": 22, "end": 29})

    def test_provenance_records_the_repair_and_the_official_command(self) -> None:
        provenance = json.loads((self.output_root / "train/meta/provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["hand_repair"]["invalid_source_samples"], 2)
        self.assertEqual(provenance["hand_repair"]["channels"], {"left_hand_thumb_0_joint": 2})
        self.assertEqual(
            provenance["statistics"]["official_command"],
            official_stats_command(self.config, self.output_root / "train"),
        )
        self.assertFalse(provenance["statistics"]["official_produced"])
        self.assertEqual(provenance["state_sources"]["projected_gravity"], "synthetic upright [0, 0, -1]")

    def test_episode_metadata_carries_the_repair_ledger(self) -> None:
        rows = [
            json.loads(line)
            for line in (self.output_root / "train/meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["length"], ROWS)
        self.assertEqual(rows[0]["corpus_frames"], CORPUS_ROWS)
        self.assertEqual(rows[0]["trimmed_rows"], 45)
        self.assertEqual(rows[0]["hand_repair"]["channels"], {"left_hand_thumb_0_joint": 2})
        self.assertEqual(rows[0]["source_episode_index"], 1)
        self.assertEqual(rows[0]["source_collection"], COLLECTION)
        val_rows = [
            json.loads(line)
            for line in (self.output_root / "val/meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(val_rows[0]["hand_repair"]["invalid_source_samples"], 0)

    def test_conversion_manifest_aggregates_the_repair(self) -> None:
        manifest = json.loads((self.output_root / "conversion_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["hand_repair"]["invalid_source_samples"], 2)
        self.assertEqual(manifest["hand_repair"]["channels"], {"left_hand_thumb_0_joint": 2})
        self.assertEqual(manifest["selection"]["train"][COLLECTION], 1)
        self.assertEqual(manifest["trim"]["trailing_invalid_rows"], 45)

    def test_stats_prep_recomputes_our_own_file_and_prints_the_official_command(self) -> None:
        result = subprocess.run(
            [
                sys.executable, str(STATS_PREP),
                "--config", str(self.config_path),
                "--root", str(self.output_root),
                "--own", "--check",
            ],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("matches a fresh computation", result.stdout)
        self.assertIn("/opt/src/isaac-groot/gr00t/data/stats.py", result.stdout)
        self.assertIn("--dataset-path", result.stdout)
        # The pinned tool requires the tag, spelled as the enum member name.
        self.assertIn("--embodiment-tag UNITREE_G1_SONIC", result.stdout)

    def test_stats_prep_refuses_a_tampered_statistics_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="groot-sonic-stats-", dir=self.root) as directory:
            root = Path(directory) / "pack"
            shutil.copytree(self.output_root, root)
            path = root / "train/meta/stats_groot_sonic.json"
            stats = json.loads(path.read_text(encoding="utf-8"))
            stats["observation.state"]["max"][0] = stats["observation.state"]["max"][0] + 1.0
            path.write_text(json.dumps(stats, indent=4), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, str(STATS_PREP),
                    "--config", str(self.config_path),
                    "--root", str(root),
                    "--check",
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("do not match", result.stderr)

    def test_the_pack_carries_the_corpus_timeline_and_tokens(self) -> None:
        import pyarrow.parquet as pq

        table = pq.read_table(self.output_root / "train/data/chunk-000/episode_000000.parquet")
        timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
        np.testing.assert_allclose(timestamps, np.arange(ROWS) / 50.0, rtol=0, atol=1e-6)
        tokens = np.asarray(table.column("action.motion_token").to_pylist(), dtype=np.float32)
        with np.load(self.sonic_root / COLLECTION / "episodes/episode_000001/action.npz") as payload:
            np.testing.assert_array_equal(tokens, np.asarray(payload["action"])[:ROWS, :64])

    def test_the_installed_state_uses_independent_helpers(self) -> None:
        """A converted episode written by hand must satisfy the same helpers."""
        episode = ConvertedEpisode(
            collection=COLLECTION,
            episode_index=1,
            state=np.tile(np.asarray([0.0] * 43), (ROWS, 1)),
            gravity=np.tile(np.array([0.0, 0.0, -1.0]), (ROWS, 1)),
            motion_token=np.zeros((ROWS, 64), dtype=np.float32),
            left_hand=np.zeros((ROWS, 7), dtype=np.float32),
            right_hand=np.zeros((ROWS, 7), dtype=np.float32),
            timestamp=np.arange(ROWS) / 50.0,
            repair=HandRepair(threshold_rad=3.0),
            corpus_rows=CORPUS_ROWS,
            trimmed_rows=45,
            corpus_hand_max_error=0.0,
            raw=RawEpisode(
                collection=COLLECTION, episode_index=1, state=np.zeros((2, 28)), action=np.zeros((2, 28)),
                timestamp=np.zeros(2), state_names=UNITREE_ACTION_NAMES, action_names=UNITREE_ACTION_NAMES,
                data_file="d", video_file=Path("v"), video_start_s=0.0, video_stop_s=1.0,
            ),
            sonic=SonicEpisode(
                path=Path("a"), action=np.zeros((CORPUS_ROWS, 78), dtype=np.float32),
                timestamp=np.arange(CORPUS_ROWS) / 50.0, training_valid_mask=np.ones(CORPUS_ROWS, dtype=bool),
            ),
        )
        self.assertEqual(episode.frames, ROWS)
        self.assertEqual(episode.hand_action.shape, (ROWS, 14))

    def test_tampered_episode_is_refused(self) -> None:
        report = self._validate_copy("tampered", self._tamper_state)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("train.G1_Dex3_PickApple_Dataset.ep000001.state_matches_raw_by_name", self._failed(report))

    def test_tampered_timeline_is_refused(self) -> None:
        report = self._validate_copy("timeline", self._tamper_timeline)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("train.G1_Dex3_PickApple_Dataset.ep000001.timestamps_are_corpus_timestamps", self._failed(report))

    def test_truncated_video_is_refused(self) -> None:
        report = self._validate_copy("video", self._tamper_video)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("train.G1_Dex3_PickApple_Dataset.ep000001.video_frames", self._failed(report))

    def test_video_holding_another_segment_is_refused(self) -> None:
        """The right row count is not enough: the pictures must be the episode's."""
        report = self._validate_copy("video-offset", self._tamper_video_offset)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("train.G1_Dex3_PickApple_Dataset.ep000001.video_alignment", self._failed(report))

    def test_swapped_modality_slice_is_refused(self) -> None:
        report = self._validate_copy("modality", self._tamper_modality)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("train.modality_exact", self._failed(report))

    def test_converter_refuses_a_corpus_with_the_wrong_tail(self) -> None:
        temporary = Path(tempfile.mkdtemp(prefix="groot-sonic-tail-", dir=self.root))
        try:
            corpus = np.load(self.sonic_root / COLLECTION / "episodes/episode_000001/action.npz")
            mask = np.asarray(corpus["training_valid_mask"]).copy()
            mask[-44:] = True
            mask[-45] = False
            directory = temporary / COLLECTION / "episodes" / "episode_000001"
            directory.mkdir(parents=True)
            np.savez(directory / "action.npz", action=corpus["action"], timestamp=corpus["timestamp"],
                     training_valid_mask=mask)
            contract = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            contract["source"]["sonic_root"] = str(temporary)
            path = temporary / "contract.yaml"
            path.write_text(yaml.safe_dump(contract), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, str(CONVERTER), "--config", str(path),
                    "--split-manifest", str(self.manifest_path),
                    "--output-root", str(temporary / "pack"),
                    "--collection", COLLECTION, "--train-per-task", "1", "--val-per-task", "0",
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid", result.stderr)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def test_registry_check_passes_in_the_training_image(self) -> None:
        """With /opt/src mounted, the check runs against the real pinned registry."""
        groot_source = Path("/opt/src/isaac-groot")
        if not (groot_source / REGISTRY_RELATIVE_PATH).is_file():
            self.skipTest("the pinned Isaac-GR00T checkout is not mounted in this environment")
        report = self._validate_copy("registry-real", lambda root: None, groot_source=groot_source)
        self.assertEqual(report["status"], "PASS", self._failed(report))
        names = {entry["check"] for entry in report["checks"]}
        self.assertIn("train.groot_registry", names)
        self.assertNotIn("train.groot_registry", {entry["check"] for entry in report["skipped"]})

    def test_registry_check_fails_on_a_drifted_registry(self) -> None:
        report = self._validate_copy(
            "registry-drift",
            lambda root: None,
            groot_source=self._write_groot_source("drift", registry_source(action_keys=("motion_token",))),
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("train.groot_registry", self._failed(report))

    def test_registry_check_fails_on_a_missing_entry(self) -> None:
        report = self._validate_copy(
            "registry-missing",
            lambda root: None,
            groot_source=self._write_groot_source("missing", registry_source(key="some_other_embodiment")),
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("train.groot_registry", self._failed(report))

    def test_registry_check_is_skipped_without_the_checkout(self) -> None:
        temporary = Path(tempfile.mkdtemp(prefix="groot-sonic-nogroot-", dir=self.root))
        try:
            report = self._validate_copy("registry-absent", lambda root: None, groot_source=temporary / "empty")
            self.assertEqual(report["status"], "PASS", self._failed(report))
            self.assertIn("train.groot_registry", {entry["check"] for entry in report["skipped"]})
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def test_float32_timestamp_grid_does_not_shift_the_re_derivation(self) -> None:
        """A numerically perfect pack must not fail on its float32 timestamp column.

        LeRobot v2.1 stores `timestamp` as float32, which rounds the 50 Hz grid by
        up to ~2e-6 s; re-deriving the state and the hands on that rounded grid
        instead of the frozen corpus grid turns a fast hand's motion into more than
        the micro-radian tolerance. The shift below stays far inside the tolerance
        the stored column is checked against, and row offsets (not timestamps)
        drive the action horizon, so the pack must still validate.
        """
        report = self._validate_copy("timestamp-grid", self._tamper_timestamps)
        self.assertEqual(report["status"], "PASS", self._failed(report))

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _failed(report: dict) -> set[str]:
        return {entry["check"] for entry in report["failed"]}

    def _write_groot_source(self, name: str, source: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix=f"groot-sonic-registry-{name}-", dir=self.root))
        path = root / REGISTRY_RELATIVE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return root

    def _validate_copy(self, name: str, tamper, *, groot_source: Path | None = None) -> dict:
        with tempfile.TemporaryDirectory(prefix=f"groot-sonic-{name}-", dir=self.root) as directory:
            root = Path(directory) / "pack"
            shutil.copytree(self.output_root, root)
            tamper(root)
            config = load_config(self.config_path)
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return validate_dataset(
                config,
                root,
                manifest,
                check_lerobot=False,
                check_groot=True,
                groot_source=groot_source,
            )

    @staticmethod
    def _tamper_state(root: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = root / "train/data/chunk-000/episode_000000.parquet"
        table = pq.read_table(path)
        rows = table.column("observation.state").to_pylist()
        rows[3][10] = 0.5
        table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", pa.array(rows))
        pq.write_table(table, path)

    @staticmethod
    def _tamper_timeline(root: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = root / "train/data/chunk-000/episode_000000.parquet"
        table = pq.read_table(path)
        stamps = table.column("timestamp").to_pylist()
        stamps[5] = stamps[5] + 0.5
        table = table.set_column(table.schema.get_field_index("timestamp"), "timestamp", pa.array(stamps))
        pq.write_table(table, path)

    @staticmethod
    def _tamper_timestamps(root: Path) -> None:
        """Shift the stored float32 grid, well inside the tolerance it is checked with."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = root / "train/data/chunk-000/episode_000000.parquet"
        table = pq.read_table(path)
        stamps = [value + 2e-5 for value in table.column("timestamp").to_pylist()]
        table = table.set_column(table.schema.get_field_index("timestamp"), "timestamp", pa.array(stamps))
        pq.write_table(table, path)

    @staticmethod
    def _tamper_video(root: Path) -> None:
        clip = root / "train/videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
        reject_video_frames(clip, ROWS - 3)

    def _tamper_video_offset(self, root: Path) -> None:
        """Re-encode the clip from one second later: same length, wrong segment."""
        clip = root / "train/videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
        source = self.raw_root / COLLECTION / "videos/observation.images.cam_left_high/chunk-000/file-000.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "1.0", "-i", str(source),
                "-vf", "fps=50:round=down", "-frames:v", str(ROWS), "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-y", str(clip),
            ],
            check=True,
        )

    @staticmethod
    def _tamper_modality(root: Path) -> None:
        path = root / "train/meta/modality.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["state"]["left_hand"] = {"start": 29, "end": 35}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
