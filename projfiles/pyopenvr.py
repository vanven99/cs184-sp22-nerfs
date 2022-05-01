import sys
import time
import openvr
import win32pipe, win32file, pywintypes
import numpy as np

import texture

hmd = openvr.init(openvr.VRApplication_Scene)
vr_sys = openvr.VRSystem()
assert openvr.VRCompositor()

poses = []
poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)

#Pipe code for interprocess communication: Images NOT supported yet, still figuring out how to recover
#PNG images from bytes and display them to the vive

#handle = win32file.CreateFile(
#                r'\\.\pipe\Foo',
#                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
#                0,
#                None,
#                win32file.OPEN_EXISTING,
#                0,
#                None
#            )

#resp = win32file.ReadFile(handle, 64*1024)
#print(f"message: {resp}")

overlay = openvr.VROverlay()
key = "image"
name = "image"
#ov = overlay.createOverlay(key, name)

#OpenVR gets the NeRF image by reading from a file. Will replace with the far
#more efficient pipe method for interprocess communication when able.
# For now, putting cat.png into both eyes.
GLUint_L = texture.texture("cat.png", 1800)
GLUint_R = texture.texture("cat.png", 1800)

left_eye_texture = openvr.Texture_t()
left_eye_texture.eType = openvr.TextureType_OpenGL
left_eye_texture.eColorSpace = openvr.ColorSpace_Gamma
left_eye_texture.handle = int(GLUint_L)
right_eye_texture = openvr.Texture_t()
right_eye_texture.eType = openvr.TextureType_OpenGL
right_eye_texture.eColorSpace = openvr.ColorSpace_Gamma
right_eye_texture.handle = int(GLUint_R )

try:
    openvr.VRCompositor().submit(openvr.Eye_Left, left_eye_texture)
    openvr.VRCompositor().submit(openvr.Eye_Right, right_eye_texture)
except:
    print("failed")
# openvr.VRCompositor().submit(openvr.Eye_Right, left_eye_texture)


#overlay.setOverlayFromFile(ov, "C:/Users/Cyrus/Desktop/nerfsProj/projfiles/fox_base.png")
#overlay.showOverlay(ov
#
#
poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
#print(poses)
#openvr.VRCompositor().fadeGrid(1, 1)
print('yeet')
while True:
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    openvr.VRCompositor().clearLastSubmittedFrame()
    try:
        openvr.VRCompositor().submit(openvr.Eye_Left, left_eye_texture)
        openvr.VRCompositor().submit(openvr.Eye_Right, right_eye_texture)
    except Exception as e:
        print("failed:", repr(e))
    time.sleep(1/90)
    

    #pass