# syntax=docker/dockerfile:1.7

ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAAC_SIM_TAG=5.1.0
ARG ISAAC_SIM_DIGEST=UNSET_DIGEST
FROM ${ISAAC_SIM_IMAGE}:${ISAAC_SIM_TAG}@sha256:${ISAAC_SIM_DIGEST}

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
        ninja-build pkg-config python3 tcpdump cyclonedds-dev; \
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
# UV_PYTHON_INSTALL_DIR world-readable olmali; /root default'i developer icin dead symlink uretir

COPY locks/ /opt/locks/
COPY containers/entrypoint.sh containers/shell.sh containers/isaac-sim-env.sh /opt/humanoid-lab/
COPY containers/prepare-g1-assets.py /opt/humanoid-lab/
COPY scripts/smoke-test.sh /opt/humanoid-lab/smoke-test.sh

RUN set -eux; \
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
    test "$(git -C /opt/src/isaac-groot/external_dependencies/LIBERO rev-parse HEAD)" = 8f1084e3132a39270c3a13ebe37270a43ece2a01; \
    test "$(git -C /opt/src/isaac-groot/external_dependencies/SimplerEnv rev-parse HEAD)" = 8a2d286c926c1371927caa7651a412b4cc331756; \
    test "$(git -C /opt/src/isaac-groot/external_dependencies/robocasa rev-parse HEAD)" = d89d481ce9c76da7f179466981676e268aa842e5; \
    test "$(git -C /opt/src/isaac-groot/external_dependencies/robocasa-gr1-tabletop-tasks rev-parse HEAD)" = 4840e671596f93ca03651524b9f72ffb1aadfeff; \
    chown -R "${DEVELOPER_UID}:${DEVELOPER_GID}" /opt/src /opt/venvs /opt/humanoid-lab /opt/assets /workspace

RUN set -eux; \
    test -f /opt/locks/isaac-sonic/uv.lock; \
    test -f /opt/locks/sonic-sim/uv.lock; \
    test -f /opt/src/isaac-groot/uv.lock; \
    test -x /isaac-sim/kit/python/bin/python3; \
    uv venv --python /isaac-sim/kit/python/bin/python3 /opt/venvs/isaac-sonic; \
    UV_PROJECT_ENVIRONMENT=/opt/venvs/isaac-sonic uv sync --frozen --no-dev --project /opt/locks/isaac-sonic; \
    # archive dists data dosyalari eksik; pinned source tree'den editable install
    uv pip uninstall --python /opt/venvs/isaac-sonic/bin/python \
      isaaclab isaaclab-assets isaaclab-tasks isaaclab-rl gear-sonic || true; \
    uv pip install --python /opt/venvs/isaac-sonic/bin/python --no-deps \
      -e /opt/src/isaaclab/source/isaaclab \
      -e /opt/src/isaaclab/source/isaaclab_assets \
      -e /opt/src/isaaclab/source/isaaclab_tasks \
      -e /opt/src/isaaclab/source/isaaclab_rl \
      -e '/opt/src/sonic/gear_sonic[training]'; \
    /opt/venvs/isaac-sonic/bin/python -c 'import torch; assert torch.__version__.startswith("2.7.0"); import isaaclab, gear_sonic; print("isaaclab", isaaclab.__version__)'; \
    uv venv --python 3.11 /opt/venvs/sonic-sim; \
    UV_PROJECT_ENVIRONMENT=/opt/venvs/sonic-sim uv sync --frozen --no-dev --project /opt/locks/sonic-sim; \
    # cyclonedds cp311 wheel yok -> /opt/cyclonedds prefix'ine source build
    uv pip install --python /opt/venvs/sonic-sim/bin/python cyclonedds==0.10.2; \
    test -d /opt/src/sonic/external_dependencies/unitree_sdk2_python/unitree_sdk2py; \
    cp -a /opt/src/sonic/external_dependencies/unitree_sdk2_python/unitree_sdk2py \
      /opt/venvs/sonic-sim/lib/python3.11/site-packages/; \
    uv pip uninstall --python /opt/venvs/sonic-sim/bin/python gear-sonic || true; \
    uv pip install --python /opt/venvs/sonic-sim/bin/python --no-deps \
      -e '/opt/src/sonic/gear_sonic[sim]'; \
    /opt/venvs/sonic-sim/bin/python -c 'import mujoco, gear_sonic'; \
    /opt/venvs/sonic-sim/bin/python -c 'import unitree_sdk2py, gear_sonic.scripts.run_sim_loop'; \
    uv venv --python 3.12 /opt/venvs/groot-n17; \
    UV_PROJECT_ENVIRONMENT=/opt/venvs/groot-n17 uv sync --frozen --no-dev --project /opt/src/isaac-groot; \
    /opt/venvs/groot-n17/bin/python -c 'import torch, flash_attn, gr00t; assert torch.__version__.startswith("2.9.0")'; \
    chmod 0755 /opt/humanoid-lab/*.sh; \
    chown -R "${DEVELOPER_UID}:${DEVELOPER_GID}" /opt/venvs /opt/humanoid-lab /opt/src

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
