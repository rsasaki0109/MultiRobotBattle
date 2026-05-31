from glob import glob

from setuptools import find_packages, setup

package_name = "mrn_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/scenarios", glob("scenarios/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="multirobot-navigation maintainers",
    maintainer_email="maintainers@example.com",
    description="Deterministic 2D multi-robot world simulator for "
    "multirobot-navigation.",
    license="Apache-2.0",
    test_suite="test_mrn_sim",
    entry_points={
        "console_scripts": [
            "mrn_sim_world = mrn_sim.sim_node:main",
            "mrn_sim_bench = mrn_sim.bench_cli:main",
        ],
    },
)
