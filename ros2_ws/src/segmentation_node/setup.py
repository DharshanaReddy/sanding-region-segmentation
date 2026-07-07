from setuptools import find_packages, setup

package_name = "segmentation_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/segmentation.launch.py"]),
        (f"share/{package_name}/rviz", ["rviz/segmentation.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="DharshanaReddy",
    maintainer_email="dharshareddy02@gmail.com",
    description="Real-time defect segmentation inference node (ONNX Runtime / OpenVINO backends).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "segmentation_node = segmentation_node.segmentation_node:main",
        ],
    },
)
