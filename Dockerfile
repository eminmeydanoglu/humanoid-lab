# syntax=docker/dockerfile:1.7

ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAAC_SIM_TAG=5.1.0
ARG ISAAC_SIM_DIGEST=UNSET_DIGEST
FROM ${ISAAC_SIM_IMAGE}:${ISAAC_SIM_TAG}@sha256:${ISAAC_SIM_DIGEST} AS sources

ARG UV_VERSION=0.12.9
ARG UV_SHA256=ec7a99cd05e0cd7f80243f135ce1361c76835cb0ee60055d14d20eba8eba1460
ARG DEVELOPER_UID=1000
ARG DEVELOPER_GID=1000
ARG ISAAC_LAB_COMMIT=37ddf626871758333d6ed89cf64ad702aef127d0
ARG SONIC_COMMIT=a0732b642c0333077e127a2f56ab0014c196bca4
ARG ISAAC_GROOT_COMMIT=1a1837f20538b7d7e21f977a11a5aee14f99803c
ARG BUILD_DATE=unknown

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_NO_MANAGED_PYTHON=0 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN set -eux; \
    for commit in "${ISAAC_LAB_COMMIT}" "${SONIC_COMMIT}" "${ISAAC_GROOT_COMMIT}"; do \
      printf '%s' "$commit" | grep -Eq '^[0-9a-f]{40}$'; \
    done; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        bash-completion build-essential ca-certificates cmake curl file \
        ffmpeg git git-lfs iproute2 iputils-ping jq libegl1 libgl1 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libvulkan1 net-tools \
        ninja-build pkg-config python3 libpython3.12-dev tcpdump cyclonedds-dev; \
    rm -rf /var/lib/apt/lists/*; \
    # cyclonedds build prefix (unitree_sdk2py binding)
    mkdir -p /opt/cyclonedds; \
    ln -sfn /usr/include /opt/cyclonedds/include; \
    ln -sfn /usr/bin /opt/cyclonedds/bin; \
    ln -sfn /usr/lib/x86_64-linux-gnu /opt/cyclonedds/lib; \
    git lfs install --system --skip-smudge; \
    if ! getent group "${DEVELOPER_GID}" >/dev/null; then groupadd --gid "${DEVELOPER_GID}" developer; fi; \
    if ! getent passwd "${DEVELOPER_UID}" >/dev/null; then useradd --uid "${DEVELOPER_UID}" --gid "${DEVELOPER_GID}" --create-home --shell /bin/bash developer; fi

RUN set -eux; \
    curl --fail --location --show-error --silent \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
      --output /tmp/uv.tar.gz; \
    echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum --check --status; \
    tar --extract --gzip --file /tmp/uv.tar.gz --directory /usr/local/bin --strip-components=1; \
    rm -f /tmp/uv.tar.gz; \
    uv --version | grep -F "${UV_VERSION}"

ENV CYCLONEDDS_HOME=/opt/cyclonedds \
    CMAKE_PREFIX_PATH=/opt/cyclonedds \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python
# Runtime bootstrap overrides this image default with the persistent uv cache.

COPY containers/prepare-g1-assets.py /opt/humanoid-lab/

RUN --network=host set -eux; \
    checkout() { \
      url="$1"; commit="$2"; target="$3"; \
      git init "$target"; \
      git -C "$target" remote add origin "$url"; \
      git -C "$target" fetch --depth=1 origin "$commit"; \
      git -C "$target" checkout --detach FETCH_HEAD; \
      test "$(git -C "$target" rev-parse HEAD)" = "$commit"; \
      test -z "$(git -C "$target" status --porcelain)"; \
    }; \
    mkdir -p /opt/src /opt/venvs /opt/humanoid-lab /opt/assets /workspace/humanoid-lab; \
    checkout https://github.com/isaac-sim/IsaacLab.git "${ISAAC_LAB_COMMIT}" /opt/src/isaaclab; \
    test -d /isaac-sim; \
    ln -s /isaac-sim /opt/src/isaaclab/_isaac_sim; \
    checkout https://github.com/NVlabs/GR00T-WholeBodyControl.git "${SONIC_COMMIT}" /opt/src/sonic; \
    git -C /opt/src/sonic lfs pull \
      --include "gear_sonic/data/robot_model/model_data/g1/meshes/*"; \
    test -f /opt/src/sonic/gear_sonic/data/robot_model/model_data/g1/meshes/head_link.STL; \
    test "$(head -c 5 /opt/src/sonic/gear_sonic/data/robot_model/model_data/g1/meshes/head_link.STL)" != "versi"; \
    python3 /opt/humanoid-lab/prepare-g1-assets.py \
      --source /opt/src/sonic/gear_sonic/data/robot_model/model_data/g1 \
      --output /opt/assets/g1-mujoco-binary; \
    rm -rf /opt/src/sonic/gear_sonic/data/robot_model/model_data/g1; \
    mv /opt/assets/g1-mujoco-binary /opt/src/sonic/gear_sonic/data/robot_model/model_data/g1; \
    checkout https://github.com/NVIDIA/Isaac-GR00T.git "${ISAAC_GROOT_COMMIT}" /opt/src/isaac-groot; \
    git -C /opt/src/isaac-groot submodule update --init --recursive; \
    # Verify the pinned fine-tuning fixture, then restore pointer-only source files. \
    git -C /opt/src/isaac-groot lfs pull --include 'demo_data/cube_to_bowl_5/**'; \
    groot_demo=/opt/src/isaac-groot/demo_data/cube_to_bowl_5; \
    test -f "$groot_demo/meta/modality.json"; \
    for pattern in '*.parquet' '*.mp4'; do \
      asset="$(find "$groot_demo" -type f -name "$pattern" -print -quit)"; \
      test -n "$asset"; \
      test "$(head -c 42 "$asset")" != 'version https://git-lfs.github.com/spec/v1'; \
    done; \
    for pattern in '*.parquet' '*.mp4'; do \
      find "$groot_demo" -type f -name "$pattern" -print; \
    done | while IFS= read -r asset; do \
      relative="${asset#/opt/src/isaac-groot/}"; \
      git -C /opt/src/isaac-groot show "HEAD:$relative" > "$asset"; \
    done; \
    rm -rf /opt/src/isaac-groot/.git/lfs/objects; \
    test "$(head -c 42 "$(find "$groot_demo" -type f -name '*.mp4' -print -quit)")" = 'version https://git-lfs.github.com/spec/v1'; \
    test -z "$(git -C /opt/src/isaac-groot status --porcelain)"; \
    test "$(git -C /opt/src/isaac-groot/external_dependencies/LIBERO rev-parse HEAD)" = 8f1084e3132a39270c3a13ebe37270a43ece2a01; \
    test "$(git -C /opt/src/isaac-groot/external_dependencies/SimplerEnv rev-parse HEAD)" = 8a2d286c926c1371927caa7651a412b4cc331756; \
    test "$(git -C /opt/src/isaac-groot/external_dependencies/robocasa rev-parse HEAD)" = d89d481ce9c76da7f179466981676e268aa842e5; \
    test "$(git -C /opt/src/isaac-groot/external_dependencies/robocasa-gr1-tabletop-tasks rev-parse HEAD)" = 4840e671596f93ca03651524b9f72ffb1aadfeff; \
    chown -R "${DEVELOPER_UID}:${DEVELOPER_GID}" /opt/src /opt/venvs /opt/humanoid-lab /opt/assets /workspace

FROM sources AS dev

# Runtime code is deliberately copied after the expensive source layer.  A
# change to a shell helper therefore makes only this small final layer dirty.
COPY locks/ /opt/locks/
COPY containers/bootstrap-venvs.sh containers/entrypoint.sh containers/shell.sh containers/isaac-sim-env.sh containers/cyclonedds-sim.xml /opt/humanoid-lab/
COPY patches/sonic-sim-dds-isolation.patch /opt/humanoid-lab/
COPY scripts/smoke-test.sh scripts/fetch-groot-demo-data.sh scripts/groot-finetune-smoke.sh /opt/humanoid-lab/

RUN set -eux; \
    git -C /opt/src/sonic apply --check /opt/humanoid-lab/sonic-sim-dds-isolation.patch; \
    git -C /opt/src/sonic apply /opt/humanoid-lab/sonic-sim-dds-isolation.patch; \
    chmod 0755 /opt/humanoid-lab/*.sh; \
    chown -R "${DEVELOPER_UID}:${DEVELOPER_GID}" /opt/humanoid-lab /opt/src

LABEL org.opencontainers.image.source="git@github.com:eminmeydanoglu/humanoid-lab.git" \
      org.opencontainers.image.revision="unknown" \
      org.humanoid-lab.schema="1" \
      org.humanoid-lab.isaac-sim="5.1.0" \
      org.humanoid-lab.isaac-lab="2.3.2" \
      org.humanoid-lab.sonic="a0732b642c0333077e127a2f56ab0014c196bca4" \
      org.humanoid-lab.groot="1a1837f20538b7d7e21f977a11a5aee14f99803c" \
      org.humanoid-lab.uv="${UV_VERSION}" \
      org.humanoid-lab.build-date="${BUILD_DATE}"

# /isaac-sim 750 isaac-sim:isaac-sim -> developer group uyeligi
RUN set -eux; \
    DEV_USER="$(getent passwd "${DEVELOPER_UID}" | cut -d: -f1)"; \
    test -n "${DEV_USER}"; \
    if getent group isaac-sim >/dev/null; then usermod -aG isaac-sim "${DEV_USER}"; fi; \
    id "${DEV_USER}"

USER ${DEVELOPER_UID}:${DEVELOPER_GID}
WORKDIR /workspace/humanoid-lab
