# FastWAM RealMan deployment

The server owns the GPU/model. The robot-side client reads three D435 color
streams and both RealMan arms, sends one synchronized observation, receives an
action chunk, and streams the absolute joint targets at 30 Hz.
The left- and right-wrist RGB frames are rotated 180 degrees before upload to
match the orientation used by the RealMan teleoperation training recorder; the
overhead frame is left unchanged.

## 1. Server

```bash
bash scripts/tasks/serve_realman_fastwam.sh
```

The launcher defaults to the requested `step_005000.pt`; its sibling
`config.yaml` and `dataset_stats.json` are discovered automatically. Set
`FASTWAM_API_TOKEN` on both machines when the service is reachable outside a
trusted robot LAN.

## 2. Robot client

Install `pyrealsense2` in the client Python environment, then identify the
three camera serial numbers with `realsense-viewer`:

```bash
export SERVER_URL=http://SERVER_IP:8000
bash scripts/tasks/run_realman_fastwam_client.sh --max-replans 2
```

The launcher contains the deployed camera serials as defaults. Override
`HIGH_CAMERA_ID`, `LEFT_CAMERA_ID`, or `RIGHT_CAMERA_ID` when hardware changes.

### SSH tunnel (current deployment)

Keep the tunnel open in terminal 1. Local port `18000` forwards to the FastWAM
server listening on `127.0.0.1:8000` on the GPU machine:

```bash
ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  -p 3329 root@203.0.113.10
```

Start the dry-run client in terminal 2:

Activate any Python environment containing the client dependencies, then run:

```bash
cd /home/robot2/deltawam
bash scripts/tasks/run_realman_fastwam_client.sh \
  --max-sync-skew-ms 50 \
  --sync-timeout 8
```

The SSH command has no output while the tunnel is healthy. If port `18000` is
already occupied, choose another local port in both the `-L` option and
`SERVER_URL`.

This first run is a dry-run: it captures real observations and validates model
actions but sends no motion command. After checking camera placement, joint
ordering, logs, and an accessible emergency stop, enable motion explicitly:

```bash
EXECUTE=1 bash scripts/tasks/run_realman_fastwam_client.sh
```

Useful overrides include `LEFT_ARM_IP`, `RIGHT_ARM_IP`, `ARM_PORT`,
`--max-joint-step-rad`, `--max-sync-skew-ms`, `--sync-timeout`, `--action-hz`,
and `--instruction`. The default joint-step limit is `0.5` rad. Each camera is captured continuously into a timestamped
buffer. For every observation, the client selects the three frames nearest to
the concurrently sampled dual-arm state; the observation is rejected when the
full camera/state span exceeds `--max-sync-skew-ms` (50 ms by default). Ctrl-C
and request, protocol, SDK, synchronization, or safety failures issue a
slow-stop before disconnecting.

### Replan action alignment

Action alignment is disabled by default. Enable it to re-read the physical arm
joints after inference and skip stale leading actions at a replan boundary:

```bash
EXECUTE=1 bash scripts/tasks/run_realman_fastwam_client.sh \
  --align-action-chunk \
  --action-align-max-skip 8 \
  --action-align-min-improvement-rad 0.01
```

The client searches action indices `0..8` using the maximum absolute error over
all 14 arm joints. It skips to the closest action only when this improves the
error relative to action 0 by at least `0.01` rad. The limit prevents an
accidental match far into the chunk. Each replan logs `selected_index`, the
first/selected errors, and the improvement. Gripper values do not participate
in matching, but the gripper command at the selected action is retained.


### Tracking-gated action execution

Tracking gating is disabled by default. Enable it to time-stretch a model chunk
when the physical arm cannot keep up with 30 Hz absolute targets:

```bash
EXECUTE=1 bash scripts/tasks/run_realman_fastwam_client.sh \
  --tracking-gate \
  --tracking-error-rad 0.05 \
  --tracking-timeout 2.0
```

After sending each model action, the client reads all 14 arm joints and repeats
the same CANFD target at `--action-hz` until the maximum tracking error is at or
below `0.05` rad. Only then does it advance to the next model action. Gripper
commands are sent once and are not repeated by the gate. If the target is not
reached within two seconds, the client raises a safety error and slow-stops.
This preserves model waypoints while stretching their physical execution time.
It can be combined with `--align-action-chunk`.


### Record full-rate camera videos

The default `--record-dir log/realman_demo` records all native frames from all
three camera capture threads for the complete client run:

```bash
bash scripts/tasks/run_realman_fastwam_client.sh \
  --record-fps 30 \
  --record-codec h264_vaapi \
  --max-sync-skew-ms 50 \
  --sync-timeout 8
```

Each run creates `log/realman_demo/client_YYYYmmdd_HHMMSS_microseconds/`:

- `high_rgb.mp4`
- `left_wrist_rgb.mp4`
- `right_wrist_rgb.mp4`
- `camera_frames.jsonl` with every camera frame's sequence and capture timestamp

All three streams use the native `640x480@30` profile used by the training data.
Recording and inference consume the same source frames, with no aspect-ratio
conversion or high-resolution downsampling. Videos use Intel VAAPI H.264 with
quality parameter QP 18. Wrist videos are rotated 180 degrees for correct demo
orientation. Dedicated writer threads keep encoding off the control path; if a
writer queue fills, the client fails instead of silently dropping frames.

Override `--record-dir` to choose another output root.
Pass `--record-dir ""` to disable recording explicitly.

### Binary gripper hysteresis

The default `--gripper-mode binary-hysteresis` turns continuous model values
into one-shot fully-closed/fully-open commands:

```bash
EXECUTE=1 bash scripts/tasks/run_realman_fastwam_client.sh \
  --gripper-close-threshold 300 \
  --gripper-open-threshold 700 \
  --gripper-confirm-actions 3 \
  --gripper-confirm-seconds 0 \
  --max-sync-skew-ms 50 \
  --sync-timeout 8
```

Values at or below 300 request close position `1`; values at or above 700
request open position `1000`. Values between the thresholds keep the current
state. A transition is sent once after three consecutive qualifying actions.
Set `--gripper-confirm-actions 0 --gripper-confirm-seconds 0.2` to confirm by
time instead. If both are positive, whichever condition is reached first wins.
The left and right grippers maintain independent states.

The global close threshold and confirmation count apply to both sides unless a
side-specific override is provided:

```bash
EXECUTE=1 bash scripts/tasks/run_realman_fastwam_client.sh \
  --gripper-close-threshold 300 \
  --gripper-confirm-actions 3 \
  --left-gripper-close-threshold 500 \
  --left-gripper-confirm-actions 1 \
  --right-gripper-close-threshold 200 \
  --right-gripper-confirm-actions 2
```

In this example the left gripper closes at values at or below 500 after one
qualifying action, while the right gripper closes at values at or below 200
after two qualifying actions. `--gripper-open-threshold` and
`--gripper-confirm-seconds` remain shared by both sides.

### Blocking gripper before arm motion

This mode is disabled by default. When enabled, any changed gripper command is
sent in blocking mode and must complete before the arm target is sent:

```bash
EXECUTE=1 bash scripts/tasks/run_realman_fastwam_client.sh \
  --gripper-before-arm \
  --gripper-blocking-timeout 5
```

If neither gripper changes, the arm target is sent immediately. If both change,
the right and left grippers are completed sequentially before either arm moves.
An SDK failure or timeout prevents the arm command and triggers the normal
safety-stop path. This mode is best paired with `binary-hysteresis`; continuous
position mode may block frequently because many actions change gripper position.


## Simulation tests

No robot, camera, checkpoint load, or GPU is used:

```bash
bash scripts/tasks/test_realman_fastwam_deployment.sh
```
