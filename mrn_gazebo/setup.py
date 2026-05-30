from glob import glob

from setuptools import find_packages, setup

package_name = "mrn_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="multirobot-navigation maintainers",
    maintainer_email="maintainers@example.com",
    description="Optional Gazebo (gz sim) adapter for multirobot-navigation.",
    license="Apache-2.0",
    test_suite="test_mrn_gazebo",
    entry_points={
        "console_scripts": [
            "mrn_gz_pose_adapter = mrn_gazebo.gz_pose_adapter_node:main",
        ],
    },
)
