"""Isaac-side measurement of the target object's head-camera visibility.

The replay scene gets one extra camera next to the head camera: same pose,
same optics and same pixel scale, but a field of view scaled by
``ANALYSIS_VIEW_SCALE``. The head image is exactly the central crop of that
analysis image, so the analysis render measures the target's full projection
(pixels outside the head image included) in head-camera pixel units.

Every mesh leaf of the target object is labelled with the semantic type
``grail_target`` and a distinct value, so the annotation pipeline reports one
bounding box and one occlusion ratio per leaf. Per frame:

* ``visible_pixels`` is the number of ``grail_target`` pixels in the actual
  head-camera image.
* ``unoccluded_projected_pixels`` is ``sum(visible_leaf / (1 - occlusion_leaf))``
  over the analysis render, which is the object's unoccluded projection.

The metric arithmetic, report schema and window summaries live in
``visibility.py``; this module only talks to Isaac.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .visibility import (
    ANALYSIS_VIEW_SCALE,
    STATE_FULLY_OCCLUDED,
    STATE_OUTSIDE_IMAGE,
    STATE_VISIBLE,
    FrameVisibility,
    LeafVisibility,
    VisibilityError,
    aggregate_leaf_visibility,
    fraction,
    unmeasured_frame,
)

LABEL_TYPE = "grail_target"
LABEL_FILTER = f"{LABEL_TYPE}:*"


class VisibilitySetupError(ValueError):
    """The scene could not be prepared for visibility measurement."""


def leaf_name(index: int) -> str:
    return f"leaf_{index}"


def label_leaf_semantics(prim: Any, index: int) -> None:
    """Apply the ``grail_target`` semantic to one mesh leaf."""
    from pxr import Semantics

    api = Semantics.SemanticsAPI.Get(prim, "Semantics")
    if not api:
        api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        api.CreateSemanticTypeAttr()
        api.CreateSemanticDataAttr()
    api.GetSemanticTypeAttr().Set(LABEL_TYPE)
    api.GetSemanticDataAttr().Set(leaf_name(index))


def label_target_object(object_prim: Any) -> list[str]:
    """Label every mesh leaf of the target and return their paths.

    Instanceable subtrees are de-instanced first: semantics cannot be authored
    onto instance proxies, and a labelled proxy would defeat the per-leaf
    occlusion ratios anyway.
    """
    from pxr import Usd, UsdGeom

    for prim in Usd.PrimRange(object_prim):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
    leaves = [prim for prim in Usd.PrimRange(object_prim) if prim.IsA(UsdGeom.Gprim)]
    if not leaves:
        raise VisibilitySetupError(
            f"target object {object_prim.GetPath()} has no mesh prim to label for visibility analysis"
        )
    for index, prim in enumerate(leaves):
        label_leaf_semantics(prim, index)
    return [prim.GetPath().pathString for prim in leaves]


@dataclass(frozen=True)
class LeafMeasurement:
    """Raw annotator values of one mesh leaf in one frame."""

    name: str
    path: str
    visible_pixels: int
    occlusion_ratio: float | None
    bbox: tuple[int, int, int, int] | None


class TargetVisibility:
    """Labels the target, attaches the annotators and reads one row per frame."""

    def __init__(
        self,
        profile: Any,
        env_path: str,
        *,
        visual_scale: int = ANALYSIS_VIEW_SCALE,
    ) -> None:
        from isaaclab.sensors import Camera, CameraCfg
        import isaaclab.sim as sim_utils

        if visual_scale < 1:
            raise VisibilitySetupError(f"visibility view scale must be >= 1, got {visual_scale}")
        camera = profile.camera
        self.scale = int(visual_scale)
        self.resolution = (camera.width * self.scale, camera.height * self.scale)
        self.head_resolution = (camera.width, camera.height)
        self.field_of_view_deg = analysis_camera_angles(profile, self.scale)
        self.camera_path = f"{env_path}/Robot/{camera.parent_link}/{camera.name}_visibility"
        self._camera = Camera(
            CameraCfg(
                prim_path=self.camera_path,
                update_period=0.0,
                width=self.resolution[0],
                height=self.resolution[1],
                data_types=["semantic_segmentation"],
                colorize_semantic_segmentation=False,
                semantic_filter=LABEL_FILTER,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=camera.focal_length_mm,
                    horizontal_aperture=camera.horizontal_aperture_mm * self.scale,
                    vertical_aperture=camera.vertical_aperture_mm * self.scale,
                    clipping_range=(0.05, 20.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=camera.position_m, rot=camera.rotation_wxyz, convention="world"
                ),
            )
        )
        self._head_camera: Any = None
        self._bbox: Any = None
        self._leaves: list[tuple[str, str]] = []
        self._object_prim_path = ""
        self._saw_target = False

    @property
    def camera(self) -> Any:
        return self._camera

    @property
    def object_prim_path(self) -> str:
        return self._object_prim_path

    @property
    def leaf_paths(self) -> list[str]:
        return [path for _name, path in self._leaves]

    @property
    def saw_target(self) -> bool:
        """True once any frame showed a target pixel or bounding box."""
        return self._saw_target

    def attach(self, object_prim: Any, head_camera: Any) -> None:
        """Label the target leaves and attach the per-leaf bounding box annotator."""
        import omni.replicator.core as rep
        from omni.syntheticdata.scripts.SyntheticData import SyntheticData

        self._head_camera = head_camera
        self._object_prim_path = object_prim.GetPath().pathString
        paths = label_target_object(object_prim)
        self._leaves = [(leaf_name(index), path) for index, path in enumerate(paths)]
        # The filter is global state, so re-assert it after every camera exists.
        SyntheticData.Get().set_instance_mapping_semantic_filter(LABEL_FILTER)
        self._bbox = rep.AnnotatorRegistry.get_annotator(
            "bounding_box_2d_loose_fast", init_params={"semanticFilter": LABEL_FILTER}
        )
        render_products = self._camera.render_product_paths
        if not render_products:
            raise VisibilitySetupError("the visibility camera has no render product")
        self._bbox.attach(render_products[0])

    def summary(self) -> dict[str, Any]:
        """Configuration of the measurement, for the manifest and the report."""
        return {
            "label_type": LABEL_TYPE,
            "leaves": self.leaf_paths,
            "view_scale": self.scale,
            "resolution": list(self.resolution),
            "camera_path": self.camera_path,
            "field_of_view_deg": list(self.field_of_view_deg),
        }

    # ------------------------------------------------------------------ reading

    @staticmethod
    def _pixel_counts(camera: Any) -> tuple[dict[int, int], Mapping[Any, Any]]:
        """Per-semantic-id pixel counts and the frame's idToLabels mapping."""
        import numpy as np

        output = camera.data.output.get("semantic_segmentation")
        info = camera.data.info[0].get("semantic_segmentation") or {}
        if output is None or not info:
            return {}, {}
        values = output.reshape(-1).to("cpu").numpy()
        counts = np.bincount(values.astype(np.int64, copy=False))
        return {int(index): int(count) for index, count in enumerate(counts) if count}, info

    @staticmethod
    def _label_value(label_info: Any) -> str | None:
        return label_info.get(LABEL_TYPE) if isinstance(label_info, Mapping) else None

    @classmethod
    def _target_pixels(cls, counts: Mapping[int, int], info: Mapping[Any, Any]) -> int:
        labels = info.get("idToLabels", {})
        total = 0
        for semantic_id, label_info in labels.items():
            if cls._label_value(label_info) is not None:
                total += counts.get(int(semantic_id), 0)
        return total

    @classmethod
    def _leaf_pixels(cls, counts: Mapping[int, int], info: Mapping[Any, Any]) -> dict[str, int]:
        labels = info.get("idToLabels", {})
        by_leaf: dict[str, int] = {}
        for semantic_id, label_info in labels.items():
            name = cls._label_value(label_info)
            if name is not None:
                by_leaf[name] = by_leaf.get(name, 0) + counts.get(int(semantic_id), 0)
        return by_leaf

    def _analysis_crop_pixels(self, info: Mapping[Any, Any]) -> int:
        """Target pixels inside the analysis image's head-camera crop.

        The head image is the central crop of the analysis image, so this must
        match the head camera's own count. A mismatch means the two cameras are
        no longer aligned and the denominator would not be in head-camera
        pixels, which the caller turns into an unmeasured frame.
        """
        import numpy as np

        output = self._camera.data.output.get("semantic_segmentation")
        labels = info.get("idToLabels", {})
        target_ids = [
            int(semantic_id)
            for semantic_id, label_info in labels.items()
            if self._label_value(label_info) is not None
        ]
        if output is None or not target_ids:
            return 0
        width, height = self.head_resolution
        left = (self.resolution[0] - width) // 2
        top = (self.resolution[1] - height) // 2
        image = output.reshape(self.resolution[1], self.resolution[0])
        values = image[top : top + height, left : left + width].to("cpu").numpy()
        return int(np.isin(values, np.asarray(target_ids, dtype=values.dtype)).sum())

    def _bounding_boxes(self) -> tuple[dict[str, tuple[float, tuple[int, int, int, int]]], bool]:
        """Loose bounding box and occlusion ratio per labelled prim path."""
        data = self._bbox.get_data()
        if not isinstance(data, Mapping):
            return {}, False
        rows = data.get("data")
        info = data.get("info") or {}
        paths = list(info.get("primPaths", []))
        by_path: dict[str, tuple[float, tuple[int, int, int, int]]] = {}
        for index, row in enumerate(rows):
            path = str(paths[index]) if index < len(paths) else ""
            bbox = (int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"]))
            by_path[path] = (float(row["occlusionRatio"]), bbox)
        return by_path, True

    def _measure(self) -> tuple[list[LeafMeasurement] | None, int | None, str]:
        """Read the current frame; returns ``(leaves, head_visible, reason)``.

        ``reason`` is non-empty when the annotator output was unusable, in
        which case the frame is reported as unmeasured rather than as invisible.
        """
        if not self._leaves or self._head_camera is None:
            return None, None, "target leaves were never labelled"
        boxes, boxes_ok = self._bounding_boxes()
        if not boxes_ok:
            return None, None, "bounding box annotator returned no data"
        head_counts, head_info = self._pixel_counts(self._head_camera)
        if not head_info:
            return None, None, "head camera semantic segmentation is unavailable"
        analysis_counts, analysis_info = self._pixel_counts(self._camera)
        by_leaf = self._leaf_pixels(analysis_counts, analysis_info)
        head_target = self._target_pixels(head_counts, head_info)
        crop_target = self._analysis_crop_pixels(analysis_info)
        tolerance = max(2, int(0.01 * max(head_target, 1)))
        if abs(crop_target - head_target) > tolerance:
            return (
                None,
                None,
                f"cameras are misaligned: the head image shows {head_target} target pixels but its crop of "
                f"the analysis image shows {crop_target}",
            )
        leaves = []
        for name, path in self._leaves:
            row = boxes.get(path)
            leaves.append(
                LeafMeasurement(
                    name=name,
                    path=path,
                    visible_pixels=by_leaf.get(name, 0),
                    occlusion_ratio=row[0] if row else None,
                    bbox=row[1] if row else None,
                )
            )
        if boxes or any(measurement.visible_pixels for measurement in leaves):
            self._saw_target = True
        return leaves, head_target, ""

    def read(
        self,
        render_frame: int,
        *,
        render_fps: float,
        grasp_time_seconds: float,
    ) -> FrameVisibility:
        """One frame row for the visibility report."""
        self._camera.update(dt=0.0)
        leaves, head_visible, reason = self._measure()
        if leaves is None or head_visible is None:
            return unmeasured_frame(
                render_frame,
                render_fps=render_fps,
                grasp_time_seconds=grasp_time_seconds,
                reason=reason,
            )
        projected, exact, classification, reason = self._denominator(leaves)
        if classification == "unmeasured":
            return unmeasured_frame(
                render_frame,
                render_fps=render_fps,
                grasp_time_seconds=grasp_time_seconds,
                reason=reason,
            )
        try:
            value = fraction(head_visible, projected)
        except VisibilityError as exc:
            return unmeasured_frame(
                render_frame,
                render_fps=render_fps,
                grasp_time_seconds=grasp_time_seconds,
                reason=str(exc),
            )
        if head_visible > 0:
            state = STATE_VISIBLE
        elif classification == "fully_occluded":
            state = STATE_FULLY_OCCLUDED
        else:
            state = STATE_OUTSIDE_IMAGE
        return FrameVisibility(
            render_frame=render_frame,
            source_time_seconds=render_frame / render_fps,
            relative_to_grasp_seconds=render_frame / render_fps - grasp_time_seconds,
            visible_fraction=value,
            visible_pixels=head_visible,
            unoccluded_projected_pixels=projected,
            valid=True,
            exact=exact,
            state=state,
            reason=reason,
        )

    def _denominator(self, leaves: Sequence[LeafMeasurement]) -> tuple[float, bool, str, str]:
        """Unoccluded projected pixels plus validity of the aggregation."""
        aggregate = aggregate_leaf_visibility(
            [
                LeafVisibility(
                    name=leaf.name,
                    path=leaf.path,
                    visible_pixels=leaf.visible_pixels,
                    occlusion_ratio=leaf.occlusion_ratio,
                    bbox=leaf.bbox,
                )
                for leaf in leaves
            ],
            self.resolution,
        )
        return aggregate.projected_pixels, aggregate.exact, aggregate.classification, aggregate.reason


def analysis_camera_angles(profile: Any, scale: int = ANALYSIS_VIEW_SCALE) -> tuple[float, float]:
    """Field-of-view angles rendered by the visibility camera, in degrees."""
    camera = profile.camera
    return (
        math.degrees(2.0 * math.atan(scale * math.tan(math.radians(camera.horizontal_fov_deg / 2.0)))),
        math.degrees(2.0 * math.atan(scale * math.tan(math.radians(camera.vertical_fov_deg / 2.0)))),
    )
