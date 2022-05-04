import sys
import time
import openvr
import win32pipe, win32file, pywintypes
import numpy as np

import texture

#Pipe code for interprocess communication: Images NOT supported yet, still figuring out how to recover
#PNG images from bytes and display them to the vive

def vr_client_setup():
    handle = win32file.CreateFile(
                r'\\.\pipe\Foo',
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None
            )
    return handle

def vr_client_read_image(handle):
    resp = win32file.ReadFile(handle, 64*1024*10000)
    print(f"message: {resp}")

    png_array = list(resp[1])
    print("png array: ", png_array)
    print("length: ", len(png_array))
    return png_array

#ov = overlay.createOverlay(key, name)

#OpenVR gets the NeRF image by reading from a file. Will replace with the far
#more efficient pipe method for interprocess communication when able.
# For now, putting cat.png into both eyes.

def get_overlay_texture_from_bytes(png_array, width, height):
    GLUint_L = texture.texture_from_bytes(png_array, width, height)
    GLUint_R = texture.texture_from_bytes(png_array, width, height)

    #GLUint_L = texture.texture("cat.png", 1800)
    #GLUint_R = texture.texture("cat.png", 1800)

    left_eye_texture = openvr.Texture_t()
    left_eye_texture.eType = openvr.TextureType_OpenGL
    left_eye_texture.eColorSpace = openvr.ColorSpace_Gamma
    left_eye_texture.handle = int(GLUint_L)
    right_eye_texture = openvr.Texture_t()
    right_eye_texture.eType = openvr.TextureType_OpenGL
    right_eye_texture.eColorSpace = openvr.ColorSpace_Gamma
    right_eye_texture.handle = int(GLUint_R)
    return left_eye_texture, right_eye_texture

def render_frame(left_eye_texture, right_eye_texture):
    try:
        openvr.VRCompositor().submit(openvr.Eye_Left, left_eye_texture)
        openvr.VRCompositor().submit(openvr.Eye_Right, right_eye_texture)
    except:
        print("failed")

# openvr.VRCompositor().submit(openvr.Eye_Right, left_eye_texture)

#overlay.setOverlayFromFile(ov, "C:/Users/Cyrus/Desktop/nerfsProj/projfiles/fox_base.png")
#overlay.showOverlay(ov
#

def main():

    hmd = openvr.init(openvr.VRApplication_Scene)
    vr_sys = openvr.VRSystem()
    assert openvr.VRCompositor()

    poses = []
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)


    pipe_handle = vr_client_setup()
    png_bytes = vr_client_read_image(pipe_handle)
    left_texture, right_texture = get_overlay_texture_from_bytes(png_bytes, 100, 100)
    while True:
        poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
        openvr.VRCompositor().clearLastSubmittedFrame()
        render_frame(left_texture, right_texture)
        time.sleep(1/90)

    
if __name__ == '__main__':
    main()