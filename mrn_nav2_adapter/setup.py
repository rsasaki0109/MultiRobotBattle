from setuptools import find_packages, setup

package_name = "mrn_nav2_adapter"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="multirobot-navigation maintainers",
    maintainer_email="maintainers@example.com",
    description="Conservative Nav2 adapter for cooperative pose corrections.",
    license="Apache-2.0",
    test_suite="test",
    entry_points={
        "console_scripts": [
            "mrn_nav2_correction_broadcaster="
            "mrn_nav2_adapter.correction_broadcaster_node:main",
        ],
    },
)
