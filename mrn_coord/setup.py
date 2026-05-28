from setuptools import find_packages, setup

package_name = "mrn_coord"

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
    description="Multi-robot coordination layer (MAPF, formation, coverage) "
    "for multirobot-navigation.",
    license="Apache-2.0",
    test_suite="test_mrn_coord",
    entry_points={
        "console_scripts": [
            "mrn_mapf_demo = mrn_coord.mapf.demo:main",
        ],
    },
)
