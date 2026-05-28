# Demo Bags

Small MCAP bags will live here only when they are suitable for repository storage.

Larger bags should be published as release assets with a manifest that records ROS distro, scenario config, git SHA, and topic list.

The current demo bag contract is described in:

```bash
tools/validate_bag_manifest.py mrn_demos/bags/mrn_demo_3robots_manifest.yaml
```

Record a bag with matching topics:

```bash
scripts/record_demo_bag.sh --print-topics
scripts/record_demo_bag.sh bags/mrn_demo_3robots
```
