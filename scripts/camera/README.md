# Camera ground truth: filming guide

The robot's own sensors (one IMU in the pelvis, the servo encoders) cannot tell us where
its body actually bends, whether a foot rolls, lifts or slides, or which way a fall
twists. On 10/25 about 75% of the pelvis tilt never showed up in the encoders. A phone
filming printed tags stuck on each part of the robot measures all of that independently.

You do steps 1-3 once. Steps 4-6 happen on every filmed run.

## 1. Print the tags

```bash
python scripts/camera/make_targets.py          # -> data/camera/camera_targets_A4.pdf and _Letter.pdf
```

- Print the PDF for your paper size at **100% / "Actual size"**. Turn off "fit to page".
- Use **matte** paper. Shiny paper reflects light, and a reflection on a tag makes it unreadable.
- **Check the scale.** Every page has a bar at the bottom that must be **exactly 100 mm**.
  Measure it with a ruler. If it is off, the printer scaled the page: fix the setting and print again.
  Then measure the black squares: the floor tag is 150 mm, the torso tags 50 mm, the rest 32 mm
  (the feet are 30 mm). A tag printed 3% small makes everything look 3% farther away.
- Keep the white border around every tag when you cut them out. The tag needs it to be detected.

## 2. Mount the tags

Every tag must be **flat and rigid**: glue it to stiff card first (3 mm foam board, or a
2 mm 3D-printed plate), never bare paper. A tag that flexes measures its own wobble, not
the robot's. Mount each tag **upright**, with the printed label edge toward the floor.
Never put a tag across a joint or on a servo horn. It must move with one part only.

| Tag | Part | Where it goes |
|-----|------|---------------|
| 0 | floor | Flat on the floor right in front of the robot's toes, in line with the foot nearest the camera: its centre about 12 cm ahead of the toes, so it is about as far from the camera as the robot and the near foot does not hide it. The arrow on its page points the way the robot faces. Tape it down at all four corners so it stays flat. |
| 1, 2 | pelvis | A **fin**: a stiff card about 70 x 50 mm, glued to the BACK of the pelvis, centred between the legs, sticking straight back (the card stands vertical and runs front-to-back). Tag 1 goes on the side of the card facing the robot's right, tag 2 on the side facing its left. The hip servos cover the sides of the pelvis, which is why it needs a fin. |
| 3, 4 | torso | A **flag**: a stiff card about 70 x 70 mm, glued along the middle of the TOP of the torso and standing straight up (vertical, running front-to-back). Tag 3 faces right, tag 4 faces left. The arms cover the sides of the torso, which is why it needs a flag. |
| 5 / 6 | right / left thigh | Outer side of the thigh, half way up, clear of both servo horns. |
| 7 / 8 | right / left shin | Outer side of the shin, half way down. |
| 9 / 10 | right / left foot | A small **L-tab** about 45 mm tall, glued to the top of the foot at its outer edge near the toe, standing straight up and facing out to the side. The sole is only 5 mm thick, too thin for a tag. |

The camera sees one side of the robot. If you film from the robot's right you see tags
0, 1, 3, 5, 7, 9. From the left you see 0, 2, 4, 6, 8, 10. The far leg hides behind the near
leg, so film the side you want to measure.

If you had to put a tag somewhere else, or sideways, update its entry in
`config/camera/tags.yaml` (`face`, `spin_deg`, `offset_mm`; the comments there explain each one).

## 3. Calibrate the phone (once per phone and video mode)

The software needs to know your phone camera's exact focal length and lens distortion.
Set the phone up exactly the way you will film the robot, then keep those settings:

- the same resolution and frame rate as the robot videos (4K 60 fps if your phone has it, else 1080p 60 fps)
- the main 1x lens, no zoom
- **video stabilisation OFF**. Stabilisation re-warps every frame, which breaks the measurement.
- focus and exposure locked (on most phones, press and hold on the screen until "AE/AF lock" shows)

Print the calibration board (the last page of the PDF) and glue it flat to stiff board.
Measure one square: it should be 30 mm. Then film it for about **30 seconds**. Hold the phone
still and move the board slowly, or hold the board still and move the phone. Get the board
into the centre, every corner and every edge of the picture, both near and far, and tilted
20-40 degrees in every direction. Move slowly, because blurred frames are thrown away.

```bash
python scripts/camera/calibrate_camera.py board.mp4 --name pixel7_4k60
#   add --square-mm 29.6 if your square measured 29.6 mm
```

It prints the **RMS reprojection error**. Below 1 px is good, and 0.2-0.5 px is typical.
Above 1 px it warns you: redo the video with more light, slower motion, a flatter board,
and stabilisation really off. The result is saved as `config/camera/<name>.yaml`. Commit it,
since it is reused for every video filmed in that mode.

## 4. Set up for a run

- **Tripod.** The camera must not move during the video, so do not touch it once it is recording.
- **Side view, far enough back that everything fits.** From the floor to the top of the torso
  flag is about 0.85 m, and a phone's main lens filming sideways (landscape) only sees about
  38-40 degrees from the top of the picture to the bottom. Up close, the floor tag falls out of
  the picture. So:
  - **4K, landscape (best):** camera about **1.4 m** from the robot (1.3-1.5 m), lens about
    **40 cm** above the floor, pointing level at the hips. The small 32 mm tags are then about
    60 px wide in the picture, plenty for good angles.
  - **Only 1080p:** turn the phone **upright (portrait)** and film from about **1.0 m**, lens about
    40 cm high. In 1080p landscape from 1.4 m the small tags would be only about 32 px wide,
    too small to measure angles well. Calibrate in portrait too (step 3): a calibration made
    with the phone sideways does not fit an upright video, and the software refuses it.
- **Check the picture before you start.** The floor tag, both feet and the top of the torso flag
  must all be inside the frame, with a little margin. If the floor tag is cut off, the software
  cannot tell which way is up, and the body angles cannot be compared with the IMU.
- **Light:** bright and even. Daylight or several lamps, with no strong shadows across the tags
  and no reflections on them. Bright light also lets the phone use a fast shutter, which means less blur.
- Use the **same phone settings as the calibration**: resolution, 60+ fps, 1x lens, stabilisation off.
  Lock focus and exposure on the robot.

## 5. Film the run, with the tap protocol

The phone and the robot have separate clocks. A sharp tap that both of them see is how the
software lines them up.

1. Start filming.
2. Start the robot run as usual, with logging on (`deploy_standing.py ... --log`).
3. Once the robot is standing, **tap the pelvis firmly once**. Use a quick, sharp knock with a
   finger, in view of the camera, and pull your hand straight away. Then wait 2 seconds.
4. Do the experiment (pushes, standing, whatever the run is for).
5. At the end, while the robot is still standing, **tap the pelvis again** and wait 2 seconds.
   The second tap measures how far the two clocks drift apart. If the experiment is a push
   meant to knock it over, do this end tap *before* that push.
6. Stop the robot run, then stop filming.

If the robot falls before you get to the end tap, that is fine: the fall itself is the end
event, because both the camera and the robot's log saw it. Do not tap it after it has
fallen. The robot stops logging the moment it falls, so the log never sees that tap.

## 6. Process it (on the laptop)

```bash
python scripts/camera/video_pose.py run.mp4 --camera config/camera/pixel7_4k60.yaml
#   -> run.tags.csv  run.links.csv  run.pose.json ; prints how often each tag was seen
python scripts/camera/sync.py run.tags.csv logs/pi/<the run's log>.csv
#   -> run.aligned.csv  run.sync.json ; prints the clock offset, the drift and camera-vs-encoder
```

Check the printout from `video_pose.py`. Every tag you meant to film should be seen in most
frames, with an error below about 0.5 px. A tag seen in 0% of frames is hidden, too small,
reflecting light, or missing its white border. "CAMERA MOVED?" means the tripod moved
during the video.

Check that `sync.py` found the **start** and **end** events at the times you tapped (or the
fall, for the end). If it prints **SYNC NOT VERIFIED**, the two clocks could not be checked
against each other: look at the event times it printed and compare them with when you
tapped before using the result. It still writes the files, but `run.sync.json` says
`"verified": false`.

`run.aligned.csv` has one row per log row (same `t_wall`) with the camera's link angles
(`cam.*`), the same joint angles from the encoders (`enc.*`) and the IMU's pelvis angle
(`imu.pg.*` is what the robot's controller saw, smoothed on the robot; `imu.acc.*` is the
raw accelerometer) side by side. If the floor tag was never seen, the body-part angles
(`cam.pelvis.pitch` and so on) are left blank, because they would be measured from the
camera's tilt instead of from the floor; the joint angles are still filled in. Give that file, the log and the two JSON files to Claude.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| A tag is never detected | Hidden, too far away or too small in the frame, a reflection on it, or the white border cut off. |
| A tag flickers in and out | Motion blur. Add light, or use a higher frame rate. |
| Many `jump` flags | The tag's pose flips between its two possible tilts: it is too small in the picture. Move the camera closer. (`ambiguous` alone is normal for a tag seen face-on: the tilt is chosen by continuity, and only a `jump` means that choice went wrong.) |
| Positions are off by a few % | The print was scaled. Check the 100 mm bar. |
| Calibration RMS > 1 px | Stabilisation was on, autofocus moved, the board was curled, or the frames were blurred. |
| No events found in sync | The tap was too soft or out of view. A push or a fall also works as an event if it is in both recordings. |
| SYNC NOT VERIFIED, one event | Only one tap matched. Check its time against when you tapped. Next time, tap at the end too (before any push that may knock it over). |
| "a stronger motion match ... against the protocol" | The video and the log only line up if filming started after the robot run. If that is what happened, rerun with the `--offset-guess` it prints. Otherwise ignore it: something moved in the video that the robot did not feel, like handling it before the run. |
| Body angles blank in `aligned.csv` | The floor tag was out of the picture or unreadable (`world_ref` = 0). Move the camera back (step 4). |
