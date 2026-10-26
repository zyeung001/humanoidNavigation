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
| 0 | floor | Flat on the floor, between the camera and the robot, about 40 cm from the camera. The arrow on its page points the way the robot faces. Tape it down at all four corners so it stays flat. |
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
- **Side view**, about **1 m** from the robot (0.8-1.0 m), with the lens at **hip height** (about 45 cm),
  tilted a little down so that the floor tag, both feet and the top of the torso flag are all in the picture.
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
5. At the end, with the robot still standing if it can, **tap the pelvis again** and wait 2 seconds.
   The second tap measures how far the two clocks drift apart.
6. Stop the robot run, then stop filming.

If the robot falls before the end, tap it where it lies before you stop.

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

Check that `sync.py` found the **start** and **end** events at the times you tapped.

`run.aligned.csv` has one row per log row (same `t_wall`) with the camera's link angles
(`cam.*`), the same joint angles from the encoders (`enc.*`) and the IMU's pelvis angle
(`imu.*`) side by side. Give that file, the log and the two JSON files to Claude.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| A tag is never detected | Hidden, too far away or too small in the frame, a reflection on it, or the white border cut off. |
| A tag flickers in and out | Motion blur. Add light, or use a higher frame rate. |
| Many `jump` flags | The tag's pose flips between its two possible tilts: it is too small in the picture. Move the camera closer. (`ambiguous` alone is normal for a tag seen face-on: the tilt is chosen by continuity, and only a `jump` means that choice went wrong.) |
| Positions are off by a few % | The print was scaled. Check the 100 mm bar. |
| Calibration RMS > 1 px | Stabilisation was on, autofocus moved, the board was curled, or the frames were blurred. |
| No events found in sync | The tap was too soft or out of view. A push or a fall also works as an event if it is in both recordings. |
