"""Controller selection.

The profile decides which controller a run uses; the CLI may override it.  Only
the provider named in the profile is imported, so a run without a controller
never touches the DDS bindings or a policy runtime.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts.commands import CommandError, JointCommand
from .base import ControllerInterface, ControllerSource

PROVIDERS = ("none", "scripted", "sonic_dds")


def build_controller(
    config: Mapping[str, Any] | None,
    *,
    provider_override: str | None,
    physics_dt: float,
    ttl_s: float,
) -> tuple[ControllerSource | None, ControllerInterface | None]:
    settings = dict(config or {})
    provider = str(settings.get("provider", "none"))
    if provider_override is not None:
        provider = provider_override
    if provider == "none":
        return None, None
    if provider == "scripted":
        from . import scripted

        return (
            scripted.ScriptedController(settings, physics_dt=physics_dt, ttl_s=ttl_s),
            scripted.interface(settings),
        )
    if provider == "sonic_dds":
        from . import sonic_dds

        return (
            sonic_dds.SonicDdsController(settings, physics_dt=physics_dt, ttl_s=ttl_s),
            sonic_dds.interface(settings),
        )
    raise CommandError(f"unknown controller provider {provider!r}; expected one of {PROVIDERS}")


__all__ = ["PROVIDERS", "build_controller", "JointCommand"]
