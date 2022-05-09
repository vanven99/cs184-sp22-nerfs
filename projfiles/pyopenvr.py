import math
import openvr
import win32pipe, win32file, pywintypes
import numpy as np

import texture as tx

# Globals
debug = False
vr_sys = None
# How much to rotate eyes towards each other (rad)
rotation = .1
left_eye_transform = None
right_eye_transform = None
# Image Resolution
size = 200

# Utils
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
    
# Rotation helpers
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

# VR Utils
def vr_server_setup():
    pipe = win32pipe.CreateNamedPipe(
    r'\\.\pipe\VR_SERVER',
    win32pipe.PIPE_ACCESS_DUPLEX,
    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
    1, 65536, 65536,
    0,
    None)
    return pipe


# Sends *current* coordinates to instant-ngp pipe
def vr_server_send_coords(handle):
    # initial_matrix is an np.ndarray
    poses = []
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]

    hmd = convert_to_numpy(hmd_pose.mDeviceToAbsoluteTracking)
    left_eye_matrix = np.hstack((y_rotate(rotation) @ hmd[:3, :3], np.array([hmd[:, 3] + left_eye_transform[:, 3]]).T))
    right_eye_matrix = np.hstack((y_rotate(-rotation) @ hmd[:3, :3], np.array([hmd[:, 3] + right_eye_transform[:, 3]]).T))
    left_eye_matrix = convert_to_hmdmatrix(left_eye_matrix)
    right_eye_matrix = convert_to_hmdmatrix(right_eye_matrix)

    left_bytes = win32file.WriteFile(handle, bytes(left_eye_matrix))
    right_bytes = win32file.WriteFile(handle, bytes(right_eye_matrix))

    if debug:
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
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]

    initial_matrix = hmd_pose.mDeviceToAbsoluteTracking

    # Send initial headset position
    vr_server_send_matrix(vr_server_handle, initial_matrix)

    while True:
        # Send current position for each eye
        vr_server_send_coords(vr_server_handle)

        #read from pipe
        png_bytes_l = vr_client_read_image(vr_client_handle)
        png_bytes_r = vr_client_read_image(vr_client_handle)

        # Create textures
        left_texture = get_overlay_texture_from_bytes(png_bytes_l, size, size)
        right_texture = get_overlay_texture_from_bytes(png_bytes_r, size, size)
        
        render_frame(left_texture, right_texture)
    
if __name__ == '__main__':
    main()
