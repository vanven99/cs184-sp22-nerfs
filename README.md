# VRNeRF

## A VR Viewer for Nvidia's Instant-NGP

## Requirements

* A working installation of Instant-NGP
* A VR headset compatible with SteamVR
* A NVIDIA GPU capable of running Instant-NGP (we recommend RTX 3080+)

 Installation and Usage

1. Install dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
2. Copy [projfiles/run.py](https://github.com/vanven99/cs184-sp22-nerfs/blob/main/projfiles/run.py) into the scripts folder of your Instant-NGP install

3. Run [projfiles/pyopenvr.py](https://github.com/vanven99/cs184-sp22-nerfs/blob/main/projfiles/pyopenvr.py)
```bash
python pyopenvr.py [--stereo]
```

4. Once you receive a "waiting for client" message, run the [run.py](https://github.com/vanven99/cs184-sp22-nerfs/blob/main/projfiles/run.py) script from Instant-NGP with the VR flag enabled.
```bash
python scripts/run.py --scene [scene] --vr [--stereo]
```

## Options

### run.py
```bash
$ python scripts/run.py --help
usage: run.py [-h] [Everything normally in Instant-NGP] [--size SIZE] [--vr] [--stereo] [--translation_scale TRANSLATION_SCALE] [--camera_offset CAMERA_OFFSET [CAMERA_OFFSET ...]] [--initial_rotation INITIAL_ROTATION [INITIAL_ROTATION ...]]

Run neural graphics primitives testbed with additional configuration & output options

optional arguments:
  [Everything normally in Instant-NGP]
  -h, --help            show this help message and exit
  --size SIZE           Sets height and width simultaneously (default 200 if vr enabled)
  --vr                  Enable vr
  --stereo              Enables stereo imaging for VR (must also be enabled for pyopenvr.py)
  --translation_scale TRANSLATION_SCALE
                        Scales positional movement in VR
  --camera_offset CAMERA_OFFSET [CAMERA_OFFSET ...]
                        Offset of initial camera position (x y z)
  --initial_rotation INITIAL_ROTATION [INITIAL_ROTATION ...]
                        Additional rotation of initial camera position, in degrees (x y z)
```

### pyopenvr.py
```bash
$ python projfiles/pyopenvr.py --help
usage: pyopenvr.py [-h] [--size SIZE] [--stereo] [--debug] [--focus_distance FOCUS_DISTANCE]

Run a VR Viewer for Instant-NGP

optional arguments:
  -h, --help            show this help message and exit
  --size SIZE           The resolution of the input image (default 200)
  --stereo              Enable stereo imaging (must also be enabled for run.py)
  --debug               Enable debug logging
  --focus_distance FOCUS_DISTANCE
                        Distance to focus eyes, in meters
```

## Current issues
* Tracking is currently not one-to-one with headset movement

## Future work
* Using light fields to speed up rendering
* Adaptive focus