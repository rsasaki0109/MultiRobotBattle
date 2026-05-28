# Demo Storyboard

Title: Cooperative Localization in ROS 2 under GNSS Outage, Packet Loss, and Clock Drift

## Sequence

1. Three robots move in a synthetic map.
2. Robot 2 loses GNSS.
3. Local-only pose drifts.
4. V2V relative constraints appear.
5. Packet loss and latency are injected.
6. Some constraints turn red as rejected.
7. Cooperative pose recovers.
8. Covariance ellipse shrinks.
9. Foxglove shows latency, packet loss, and ATE.
10. The final frame shows the one-command launch.
