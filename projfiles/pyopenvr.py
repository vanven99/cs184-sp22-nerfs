import sys
import time
import openvr
import win32pipe, win32file, pywintypes
import numpy as np

import texture

#Pipe code for interprocess communication: Images NOT supported yet, still figuring out how to recover
#PNG images from bytes and display them to the vive
def vr_server_setup():
    pipe = win32pipe.CreateNamedPipe(
    r'\\.\pipe\VR_SERVER',
    win32pipe.PIPE_ACCESS_DUPLEX,
    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
    1, 65536, 65536,
    0,
    None)
    return pipe

def vr_server_send_coords(handle):
    poses = []
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
    #print(hmd_pose.mDeviceToAbsoluteTracking)
    
    #print(type(hmd_pose.mDeviceToAbsoluteTracking[0][0]))
    win32file.WriteFile(handle, bytes(hmd_pose.mDeviceToAbsoluteTracking))
    
def vr_client_setup():
    handle = win32file.CreateFile(
                r'\\.\pipe\NGP_SERVER',
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
    #print(f"message: {resp}")

    png_array = list(resp[1])
    #print("png array: ", png_array)
    #print("length: ", len(png_array))
    return png_array

#

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
    overlay = openvr.VROverlay()

    #create SERVER
    vr_server_handle = vr_server_setup()
    
    #client CONNECTED
    print("waiting for client")
    win32pipe.ConnectNamedPipe(vr_server_handle, None)
    print("got client")

    #create CLIENT
    vr_client_handle = vr_client_setup()

    poses = []

    #create overlays
    right_overlay = overlay.createOverlay("right", "right")
    overlay.showOverlay(right_overlay)
    #left_overlay = overlay.createOverlay("left", "left")
    #overlay.showOverlay(left_overlay)
    
    #define overlay transform matrices
    right_matrix = openvr.HmdMatrix34_t()
    right_matrix.m[0][0] = 1
    right_matrix.m[0][1] = 0
    right_matrix.m[0][2] = 0
    right_matrix.m[0][3] = .12

    right_matrix.m[1][0] = 0
    right_matrix.m[1][1] = 1
    right_matrix.m[1][2] = 0
    right_matrix.m[1][3] = 0.08

    right_matrix.m[2][0] = 0.0
    right_matrix.m[2][1] = 0.0
    right_matrix.m[2][2] = 1.0
    right_matrix.m[2][3] = -.3
    
    left_matrix = openvr.HmdMatrix34_t()
    left_matrix.m[0][0] = 1
    left_matrix.m[0][1] = 0
    left_matrix.m[0][2] = 0
    left_matrix.m[0][3] = -.12

    left_matrix.m[1][0] = 0
    left_matrix.m[1][1] = 1
    left_matrix.m[1][2] = 0
    left_matrix.m[1][3] = 0.08

    left_matrix.m[2][0] = 0.0
    left_matrix.m[2][1] = 0.0
    left_matrix.m[2][2] = 1.0
    left_matrix.m[2][3] = -.3

    while True:
        #send pipe coords
        vr_server_send_coords(vr_server_handle)
        #read from pipe
        png_bytes = vr_client_read_image(vr_client_handle)
        left_texture, right_texture = get_overlay_texture_from_bytes(png_bytes, 100, 100)
        #overlay.setOverlayFlag(right_overlay, VROverlayFlags.SideBySide_Parallel, True);

        overlay.setOverlayTransformTrackedDeviceRelative(right_overlay, 0, right_matrix)
        #overlay.setOverlayTransformTrackedDeviceRelative(left_overlay, 0, left_matrix)
        poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
        
        #render_frame(left_texture, right_texture)
        #set overlay texture
        overlay.setOverlayTexture(right_overlay, right_texture)
        #overlay.setOverlayTexture(left_overlay, left_texture)
        time.sleep(.01)
        openvr.VRCompositor().clearLastSubmittedFrame()

    
if __name__ == '__main__':
    main()