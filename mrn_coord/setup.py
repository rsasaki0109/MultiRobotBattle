from glob import glob

from setuptools import find_packages, setup

package_name = "mrn_coord"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/benchmarks", glob("benchmarks/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="multirobot-battle maintainers",
    maintainer_email="maintainers@example.com",
    description="Multi-robot coordination layer (MAPF, formation, coverage) "
    "for multirobot-battle.",
    license="Apache-2.0",
    test_suite="test_mrn_coord",
    entry_points={
        "console_scripts": [
            "mrn_mapf_demo = mrn_coord.mapf.demo:main",
            "mrn_formation_demo = mrn_coord.formation.demo:main",
            "mrn_coverage_demo = mrn_coord.coverage.demo:main",
            "mrn_lifelong_demo = mrn_coord.lifelong.demo:main",
            "mrn_mapf_bench = mrn_coord.mapf.bench_cli:main",
            "mrn_mapf_planner = mrn_coord.mapf.planner_node:main",
            "mrn_path_follower = mrn_coord.mapf.follower_node:main",
            "mrn_formation_controller = "
            "mrn_coord.formation.controller_node:main",
            "mrn_coverage_allocator = mrn_coord.coverage.allocator_node:main",
            "mrn_goal_follower = mrn_coord.coverage.goal_follower_node:main",
            "mrn_agent_sim = mrn_coord.agent_sim_node:main",
            "mrn_pose_bridge = mrn_coord.pose_bridge_node:main",
        ],
    },
)
