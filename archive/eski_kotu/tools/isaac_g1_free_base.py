"""Free-base USD preparation for the G1 assets used in simulation.

``g1_29dof_inspire_hand.usd`` carries an extra disabled ``root_joint`` above the
pelvis.  Leaving it in place makes PhysX fail to create the articulation, so the
articulation root is moved onto ``pelvis`` and the world joint is disabled.

The function only touches USD; it is a no-op guard when the asset does not have
the extra root joint.
"""

from __future__ import annotations

__all__ = ["FREE_BASE_ROOT_JOINT_PATH", "configure_free_base_articulation",
           "has_extra_root_joint"]

FREE_BASE_ROOT_JOINT_PATH = "/World/envs/env_0/Robot/root_joint"
ROBOT_PATH = "/World/envs/env_0/Robot"
PELVIS_PATH = "/World/envs/env_0/Robot/pelvis"


def has_extra_root_joint() -> bool:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    return bool(stage.GetPrimAtPath(FREE_BASE_ROOT_JOINT_PATH))


def configure_free_base_articulation() -> bool:
    """Move the articulation root from the disabled world joint to the pelvis.

    Returns True when an override was applied, False when the asset had no
    extra root joint and nothing needed changing.
    """
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_joint_prim = stage.GetPrimAtPath(FREE_BASE_ROOT_JOINT_PATH)
    robot = stage.GetPrimAtPath(ROBOT_PATH)
    pelvis = stage.GetPrimAtPath(PELVIS_PATH)
    root_joint = UsdPhysics.Joint(root_joint_prim)

    if not root_joint_prim or not root_joint_prim.IsValid():
        return False
    if not root_joint or not robot or not pelvis:
        raise RuntimeError(
            "free-base override cannot find Robot, root_joint, and pelvis in the stage"
        )

    collision_apis = [
        UsdPhysics.CollisionAPI(prim)
        for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
        if UsdPhysics.CollisionAPI(prim)
    ]
    if not collision_apis or any(api.GetCollisionEnabledAttr().Get() is False for api in collision_apis):
        raise RuntimeError("robot asset must provide enabled collision prims")

    root_joint.GetJointEnabledAttr().Set(False)
    source_api = PhysxSchema.PhysxArticulationAPI(root_joint_prim)
    source_attributes = {
        name: root_joint_prim.GetAttribute(name).Get()
        for name in source_api.GetSchemaAttributeNames()
        if root_joint_prim.GetAttribute(name)
    }
    root_joint_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    root_joint_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    UsdPhysics.ArticulationRootAPI.Apply(pelvis)
    PhysxSchema.PhysxArticulationAPI.Apply(pelvis)
    for name, value in source_attributes.items():
        pelvis.GetAttribute(name).Set(value)

    if not UsdPhysics.ArticulationRootAPI(pelvis) or PhysxSchema.PhysxArticulationAPI(root_joint_prim):
        raise RuntimeError("free-base override did not move the articulation root to pelvis")
    return True
