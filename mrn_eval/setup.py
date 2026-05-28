from setuptools import find_packages, setup

package_name = "mrn_eval"

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
    description="Evaluation metrics and report scaffolding for cooperative localization.",
    license="Apache-2.0",
    test_suite="test_mrn_eval",
    entry_points={
        "console_scripts": [
            "mrn_eval=mrn_eval.cli:main",
            "mrn_eval_bag_to_csv=mrn_eval.bag_to_csv_cli:main",
            "mrn_eval_offline_ate=mrn_eval.offline_ate_cli:main",
            "mrn_eval_rtk_to_csv=mrn_eval.rtk_to_csv_cli:main",
            "mrn_eval_tum_to_csv=mrn_eval.tum_to_csv_cli:main",
            "mrn_experiment=mrn_eval.experiment_cli:main",
            "mrn_online_ate=mrn_eval.online_ate_node:main",
            "mrn_report=mrn_eval.report_cli:main",
        ],
    },
)
