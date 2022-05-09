import math
from mimetypes import init
import sys
import time
from tkinter import W
import openvr
import win32pipe, win32file, pywintypes
import numpy as np

import texture as tx

vr_sys = None
rotation = .1
left_eye_transform = None
right_eye_transform = None

def vr_server_setup():
    pipe = win32pipe.CreateNamedPipe(
    r'\\.\pipe\VR_SERVER',
    win32pipe.PIPE_ACCESS_DUPLEX,
    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
    1, 65536, 65536,
    0,
    None)
    return pipe

def convert_to_numpy(arr, fourbyfour = False):
    if fourbyfour:
        return np.array([[arr[i][j] for j in range(4)] for i in range(4)])
    return np.array([[arr[i][j] for j in range(4)] for i in range(3)])

def convert_to_hmdmatrix(arr):
    new_matrix = openvr.HmdMatrix34_t()
    for i in range(3):
        for j in range(4):
            new_matrix[i][j] = arr[i][j]
    return new_matrix
    

def z_rotate(theta):
    z_axis_rot = np.array([[math.cos(theta), -math.sin(theta), 0],
                        [math.sin(theta), math.cos(theta), 0],
                        [0, 0, 1]])
    return z_axis_rot
def x_rotate(theta):
    x_axis_rot = np.array([[1, 0, 0],
                        [0, math.cos(theta), -math.sin(theta)],
                        [0, math.sin(theta), math.cos(theta)]])
    return x_axis_rot

def y_rotate(theta):
    x_axis_rot = np.array([[math.cos(theta), 0, math.sin(0)],
                        [0, 1, 0],
                        [-math.sin(theta), 0, math.cos(theta)]])
    return x_axis_rot

# Sends *current* coordinates to instant-ngp pipe
def vr_server_send_coords(handle):
    # initial_matrix is an np.ndarray
    poses = []
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
    #print(hmd_pose.mDeviceToAbsoluteTracking)
    print()
    # print("projection", convert_to_numpy(vr_sys.getProjectionMatrix(0, .02, 50), fourbyfour=True))
    # print("eye to head", convert_to_numpy(vr_sys.getEyeToHeadTransform(0), fourbyfour=False))
    hmd = convert_to_numpy(hmd_pose.mDeviceToAbsoluteTracking)
    left_eye_matrix = np.hstack((y_rotate(rotation) @ hmd[:3, :3], np.array([hmd[:, 3] + left_eye_transform[:, 3]]).T))
    right_eye_matrix = np.hstack((y_rotate(-rotation) @ hmd[:3, :3], np.array([hmd[:, 3] + right_eye_transform[:, 3]]).T))
    left_eye_matrix = convert_to_hmdmatrix(left_eye_matrix)
    right_eye_matrix = convert_to_hmdmatrix(right_eye_matrix)

    print()
    print("hmd", convert_to_numpy(hmd_pose.mDeviceToAbsoluteTracking))
    print(convert_to_numpy(vr_sys.getProjectionMatrix(0, .02, 50), fourbyfour=True).shape)
    print(convert_to_numpy(hmd_pose.mDeviceToAbsoluteTracking).shape)

    left_bytes = win32file.WriteFile(handle, bytes(left_eye_matrix))
    right_bytes = win32file.WriteFile(handle, bytes(right_eye_matrix))

    # print("left bytes matrix", convert_to_numpy(hmd_pose.mDeviceToAbsoluteTracking) @ convert_to_numpy(vr_sys.getProjectionMatrix(0, .02, 50), fourbyfour=True))
    print("Sent", left_bytes, "for left matrix")
    print("Sent", right_bytes, "for right matrix")

# Sends specified matrix to instant-ngp pipe
def vr_server_send_matrix(handle, matrix):
    num_bytes = win32file.WriteFile(handle, bytes(matrix))
    print("Sent", num_bytes, "for specified matrix")

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
    resp = win32file.ReadFile(handle, 160000)
    return list(resp[1])
    # png_array = np.array(list(resp[1]))
    # print(png_array)
    # png_array = png_array.reshape(400, 400)
    # resp2 = win32file.ReadFile(handle, 160000)

    # png_array2 = np.array(list(resp2[1]))
    # png_array2 = png_array2.reshape(400, 400)

    #print("png array: ", png_array)
    #print("length: ", len(png_array))
    # return list(np.hstack((png_array, png_array2)).reshape((320000, 1)))
    png_array.extend(png_array2)
    print(np.array(png_array).shape)
    return png_array


def get_overlay_texture_from_bytes(png_array, width, height):

    GLUint = tx.texture_from_bytes(png_array, width, height)

    texture = openvr.Texture_t()
    texture.eType = openvr.TextureType_OpenGL
    texture.eColorSpace = openvr.ColorSpace_Gamma
    texture.handle = int(GLUint)
    return texture

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
    

    global vr_sys
    hmd = openvr.init(openvr.VRApplication_Scene)
    vr_sys = openvr.VRSystem()
    assert openvr.VRCompositor()
    overlay = openvr.VROverlay()

    #create SERVER
    vr_server_handle = vr_server_setup()
    global left_eye_transform
    global right_eye_transform
    left_eye_transform = convert_to_numpy(vr_sys.getEyeToHeadTransform(0))
    right_eye_transform = convert_to_numpy(vr_sys.getEyeToHeadTransform(1))
    
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
    # print(dir(vr_sys.VROverlayFlags))
    
    # overlay.setOverlayFlag(right_overlay, 1024, True)
    # overlay.setOverlayFlag(right_overlay, 1, True)
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

    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
    initial_matrix = hmd_pose.mDeviceToAbsoluteTracking
    left_eye = convert_to_numpy(vr_sys.getProjectionMatrix(0, .02, 50), fourbyfour=True)
    right_eye = convert_to_numpy(vr_sys.getProjectionMatrix(1, .02, 50), fourbyfour=True)
    # mid = left_eye + right_eye / 2
    # mid_matrix = convert_to_numpy(initial_matrix) @ mid
    help(overlay.setOverlayTransformTrackedDeviceRelative)
    vr_server_send_matrix(vr_server_handle, initial_matrix)
    while True:
        #send pipe coords
        vr_server_send_coords(vr_server_handle)
        #read from pipe
        #start = time.process_time()
        # print("left", vr_sys.getEyeToHeadTransform(0))
        # print("right", vr_sys.getEyeToHeadTransform(1))
        png_bytes_l = vr_client_read_image(vr_client_handle)
        png_bytes_r = vr_client_read_image(vr_client_handle)
        # png_bytes2 = vr_client_read_image(vr_client_handle)
        size = 200
        left_texture = get_overlay_texture_from_bytes(png_bytes_l, size, size)
        right_texture = get_overlay_texture_from_bytes(png_bytes_r, size, size)
        # _, left_texture = get_overlay_texture_from_bytes(png_bytes2, size, size)
        #openvr.VRCompositor().clearLastSubmittedFrame()
        # render_frame(left_texture, right_texture)
        #print(time.process_time() - start)
        #overlay.setOverlayFlag(right_overlay, VROverlayFlags.SideBySide_Parallel, True);
        # overlay.setOverlayTransformTrackedDeviceRelative(right_overlay, 0, right_matrix)
        # overlay.setOverlayTexture(right_overlay, left_texture)
        #overlay.setOverlayTransformTrackedDeviceRelative(left_overlay, 0, left_matrix)
        # poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
        
        render_frame(left_texture, right_texture)
        # time.sleep(0.25)
        #set overlay texture
        # overlay.setOverlayTexture(right_overlay, right_texture)
        # overlay.setOverlayTexture(left_overlay, left_texture)
        # openvr.VRCompositor().clearLastSubmittedFrame()

    
if __name__ == '__main__':
    main()
