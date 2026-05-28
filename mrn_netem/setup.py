from setuptools import find_packages, setup

package_name = "mrn_netem"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", [
            "config/loss20_delay80.yaml",
            "config/burst_loss.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="multirobot-navigation maintainers",
    maintainer_email="maintainers@example.com",
    description="Network fault profiles and experiment helpers.",
    license="Apache-2.0",
    test_suite="test_mrn_netem",
    entry_points={
        "console_scripts": [
            "mrn_netem=mrn_netem.cli:main",
            "mrn_netem_netns=mrn_netem.netns_cli:main",
        ],
    },
)
