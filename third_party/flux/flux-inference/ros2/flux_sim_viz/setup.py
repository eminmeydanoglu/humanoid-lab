from setuptools import setup

package_name = "flux_sim_viz"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "launch/flux_tf.launch.py",
            "launch/flux_rviz.launch.py",
        ]),
        ("share/" + package_name + "/rviz", ["rviz/flux_sim.rviz"]),
    ],
    install_requires=["setuptools", "rclpy"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "joint_state_bridge = flux_sim_viz.joint_state_bridge:main",
        ]
    },
)
