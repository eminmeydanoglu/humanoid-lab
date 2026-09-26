from setuptools import setup

setup(
    name="flux_dex3",
    version="0.1.0",
    packages=["flux_dex3"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/flux_dex3"]),
        ("share/flux_dex3", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pyzmq"],
    zip_safe=True,
    entry_points={"console_scripts": ["dex3_node = flux_dex3.node:main"]},
)
