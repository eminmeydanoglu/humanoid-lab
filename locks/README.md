# locks/

Per-environment frozen uv locks (all COMMITTED; `uv sync --frozen` in the
image build fails loudly rather than resolving a new graph):

- `isaac-sonic/` — Python 3.11 (Isaac Sim Kit interpreter) + Torch 2.7.0+cu128
  + Isaac Lab 2.3.2 + SONIC `[training]`. SONIC's floating SMPLSim/smplx VCS
  deps are pinned via `tool.uv.override-dependencies` to immutable archives.
- `sonic-sim/` — Python 3.11 + MuJoCo 3.12.0 + SONIC `[sim]` (G1 MuJoCo env).
- groot-n17 uses the upstream Isaac-GR00T `pyproject.toml` + `uv.lock`
  unchanged (installed from the pinned source tree, not copied here).

Regenerating (rarely, deliberately): edit the `pyproject.toml` of the env,
run `uv lock` against it, review the diff, and commit both files together.
