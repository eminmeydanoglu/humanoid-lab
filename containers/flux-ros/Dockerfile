# syntax=docker/dockerfile:1.7
#
# Simulation-only ROS 2 side of the Flux 3 / Dex3 evaluation.
#
# The node source is the repository's own ros2/flux_dex3 package; it is built
# from the bind-mounted checkout into a persistent overlay (scripts/flux-ros-build.sh)
# so the simulation and the robot run the same files.  This image supplies the
# fixed pieces only: a digest-pinned Ubuntu 24.04 / Jazzy base, the pinned
# Unitree HG message package that the SONIC-vendored unitree_sdk2py binds to on
# the same DDS wire, pinned Python dependencies for the camera bridge, and RViz
# for looking at the running simulation (the robot model is the repository's own
# URDF, so the image needs no simulation assets).

ARG ROS_IMAGE=ros
ARG ROS_TAG=jazzy-ros-base-noble
ARG ROS_DIGEST=sha256:c3706ef0a0aa45413c07803cf433602f543b22e45b4855f6fca955c2d8ecc4e8
FROM ${ROS_IMAGE}:${ROS_TAG}@${ROS_DIGEST}

ARG UNITREE_ROS2_COMMIT=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
ARG DEVELOPER_UID=1000
ARG DEVELOPER_GID=1000
ARG BUILD_DATE=unknown

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

# Python deps are pinned to the exact noble archive versions this image was
# verified with: a silent version drift must fail the build, not the run.
# rosidl-generator-dds-idl is a build dependency even though this stack only
# speaks CycloneDDS: the pinned unitree_hg CMakeLists generates Connext IDL
# tuples next to the standard typesupport and requires the generator package.
# Mesa's software rasterizer (libgl1-mesa-dri) keeps RViz usable on a host
# without GPU passthrough; the container normally renders on the NVIDIA driver.
RUN set -eux; \
    printf '%s' "${UNITREE_ROS2_COMMIT}" | grep -Eq '^[0-9a-f]{40}$'; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates git iproute2 \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-sensor-msgs \
        ros-jazzy-std-srvs \
        ros-jazzy-geometry-msgs \
        ros-jazzy-rosidl-default-generators \
        ros-jazzy-rosidl-generator-dds-idl \
        ros-jazzy-rviz2 \
        ros-jazzy-foxglove-bridge \
        libgl1-mesa-dri \
        python3-colcon-common-extensions \
        python3-numpy=1:1.26.4+ds-6ubuntu1 \
        python3-zmq=24.0.1-5build1 \
        python3-pil=10.2.0-1ubuntu1.3 \
        python3-msgpack=1.0.3-3build2 \
        python3-pytest=7.4.4-1; \
    rm -rf /var/lib/apt/lists/*; \
    if ! getent group "${DEVELOPER_GID}" >/dev/null; then groupadd --gid "${DEVELOPER_GID}" developer; fi; \
    # The ROS base already carries a uid-1000 account, so the runtime identity
    # is the numeric uid/gid rather than a name that may not exist.
    mkdir -p /home/developer; \
    chown -R "${DEVELOPER_UID}:${DEVELOPER_GID}" /home/developer

# Unitree HG messages at the pinned commit.  The layouts are byte-identical to
# the SONIC-vendored unitree_sdk2py IDL (motor_state[35] LowState, sequence
# HandState, MotorCmd fields), so both sides describe the same DDS types.
RUN set -eux; \
    git init /tmp/unitree_ros2; \
    git -C /tmp/unitree_ros2 remote add origin https://github.com/unitreerobotics/unitree_ros2.git; \
    git -c http.version=HTTP/1.1 -C /tmp/unitree_ros2 fetch --depth=1 origin "${UNITREE_ROS2_COMMIT}"; \
    git -C /tmp/unitree_ros2 checkout --detach FETCH_HEAD; \
    test "$(git -C /tmp/unitree_ros2 rev-parse HEAD)" = "${UNITREE_ROS2_COMMIT}"; \
    mkdir -p /opt/humanoid-lab/flux-ros/unitree-ws/src; \
    cp -a /tmp/unitree_ros2/cyclonedds_ws/src/unitree/unitree_hg /opt/humanoid-lab/flux-ros/unitree-ws/src/; \
    rm -rf /tmp/unitree_ros2; \
    bash -lc 'source /opt/ros/jazzy/setup.bash \
      && cd /opt/humanoid-lab/flux-ros/unitree-ws \
      && colcon build --packages-select unitree_hg --event-handlers console_cohesion+'; \
    bash -lc 'source /opt/ros/jazzy/setup.bash \
      && source /opt/humanoid-lab/flux-ros/unitree-ws/install/setup.bash \
      && python3 -c "from unitree_hg.msg import HandCmd, HandState, LowCmd, LowState; print(\"unitree_hg messages import\")"'

COPY containers/flux-ros/cyclonedds-sim.xml /opt/humanoid-lab/flux-ros/cyclonedds-sim.xml
COPY containers/flux-ros/entrypoint.sh /opt/humanoid-lab/flux-ros/entrypoint.sh

RUN set -eux; \
    chmod 0755 /opt/humanoid-lab/flux-ros/entrypoint.sh; \
    bash -lc 'source /opt/ros/jazzy/setup.bash \
      && source /opt/humanoid-lab/flux-ros/unitree-ws/install/setup.bash \
      && { \
        python3 -c "import rclpy, numpy, zmq, msgpack, PIL; import rosidl_runtime_py; print(\"python deps import\")"; \
        printf "base_image=%s\nunitree_ros2_commit=%s\n" "ros:jazzy-ros-base-noble" "${UNITREE_ROS2_COMMIT}" > /opt/humanoid-lab/flux-ros/deps.txt; \
        python3 -c "import numpy, zmq, msgpack, PIL; print(\"numpy\", numpy.__version__); print(\"pyzmq\", zmq.__version__); print(\"msgpack\", msgpack.version); print(\"pillow\", PIL.__version__)" >> /opt/humanoid-lab/flux-ros/deps.txt; \
        ros2 pkg executables rmw_cyclonedds_cpp >/dev/null; \
        ros2 pkg executables rviz2 >/dev/null; \
      }'; \
    chown -R "${DEVELOPER_UID}:${DEVELOPER_GID}" /opt/humanoid-lab/flux-ros

USER ${DEVELOPER_UID}:${DEVELOPER_GID}
ENV HOME=/home/developer
WORKDIR /workspace/humanoid-lab

LABEL org.opencontainers.image.source="git@github.com:eminmeydanoglu/humanoid-lab.git" \
      org.humanoid-lab.schema="1" \
      org.humanoid-lab.flux-ros="jazzy" \
      org.humanoid-lab.unitree-ros2="${UNITREE_ROS2_COMMIT}" \
      org.humanoid-lab.build-date="${BUILD_DATE}"
