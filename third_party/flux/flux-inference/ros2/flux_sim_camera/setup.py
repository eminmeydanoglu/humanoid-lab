from setuptools import setup

package_name = "flux_sim_camera"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/flux_sim.launch.py"]),
    ],
    install_requires=["setuptools", "numpy", "pyzmq", "msgpack", "Pillow"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "camera_bridge = flux_sim_camera.camera_bridge:main",
        "camera_jpeg = flux_sim_camera.camera_jpeg:main",
    ]},
)
