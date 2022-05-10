import math
import openvr
import win32pipe, win32file, pywintypes
import numpy as np
import argparse

import texture as tx

# Globals
debug = False
vr_sys = None
# How much to rotate eyes towards each other (rad)
rotation = .1
left_eye_transform = None
right_eye_transform = None
args = {}

def parse_args():
    parser = argparse.ArgumentParser(description="Run a VR Viewer for Instant-NGP")
    parser.add_argument("--size", default=200, type=int, help="The resolution of the input image (default 200)")
    parser.add_argument("--stereo", default=False, help="Enable stereo imaging (must also be enabled for run.py)", action='store_true')
    parser.add_argument("--debug", default=False, help="Enable debug logging", action='store_true')
    parser.add_argument("--focus_distance", default=1, type=float, help="Distance to focus eyes, in meters")

    global args
    args = parser.parse_args()

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
    return np.array([[math.cos(theta), -math.sin(theta), 0],
                        [math.sin(theta), math.cos(theta), 0],
                        [0, 0, 1]])

def x_rotate(theta):
    return np.array([[1, 0, 0],
                        [0, math.cos(theta), -math.sin(theta)],
                        [0, math.sin(theta), math.cos(theta)]])

def y_rotate(theta):
    return np.array([[math.cos(theta), 0, -math.sin(theta)],
                        [0, 1, 0],
                        [math.sin(theta), 0, math.cos(theta)]])

# Calculate rotation for focus
def set_rotation(focusDistance):
    global rotation
    c = math.sqrt(left_eye_transform[:, 3][0]**2 + focusDistance**2)
    rotation = math.pi / 2 - math.asin(focusDistance / c)
    if args.debug:
        print("Rotating vision by :", rotation, "radians")

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


# Sends *current* coordinates to instant-ngp pipe for each eye
def vr_server_send_coords_stereo(handle):
    poses = []
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]

    hmd = convert_to_numpy(hmd_pose.mDeviceToAbsoluteTracking)
    left_eye_matrix = np.hstack((y_rotate(-rotation) @ hmd[:3, :3], np.array([hmd[:, 3] + hmd[:3, :3] @ left_eye_transform[:, 3]]).T))
    right_eye_matrix = np.hstack((y_rotate(rotation) @ hmd[:3, :3], np.array([hmd[:, 3] + hmd[:3, :3] @ right_eye_transform[:, 3]]).T))
    left_eye_matrix = convert_to_hmdmatrix(left_eye_matrix)
    right_eye_matrix = convert_to_hmdmatrix(right_eye_matrix)

    left_bytes = win32file.WriteFile(handle, bytes(left_eye_matrix))
    right_bytes = win32file.WriteFile(handle, bytes(right_eye_matrix))

    if args.debug:
        print("Sent", left_bytes, "for left matrix")
        print("Sent", right_bytes, "for right matrix")

def vr_server_send_coords(handle):
    poses = []
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]

    sent_bytes = win32file.WriteFile(handle, bytes(hmd_pose.mDeviceToAbsoluteTracking))

    if args.debug:
        print("Sent", sent_bytes, "for hmd matrix")

# Sends specified matrix to instant-ngp pipe
def vr_server_send_matrix(handle, matrix):
    num_bytes = win32file.WriteFile(handle, bytes(matrix))
    if args.debug:
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
    resp = win32file.ReadFile(handle, args.size * args.size * 4)
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
    if not args.stereo:
        overlay = openvr.VROverlay()
    
    #create SERVER
    vr_server_handle = vr_server_setup()
    global left_eye_transform
    global right_eye_transform
    left_eye_transform = convert_to_numpy(vr_sys.getEyeToHeadTransform(0))
    right_eye_transform = convert_to_numpy(vr_sys.getEyeToHeadTransform(1))

    if args.stereo:
        set_rotation(args.focus_distance)
    
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

    if not args.stereo:
        # create overlay
        overlay_obj = overlay.createOverlay("main", "main")
        overlay.showOverlay(overlay_obj)
        
        # define overlay transform matrices
        overlay_matrix = openvr.HmdMatrix34_t()
        overlay_matrix.m[0][0] = 1
        overlay_matrix.m[0][1] = 0
        overlay_matrix.m[0][2] = 0
        overlay_matrix.m[0][3] = 0

        overlay_matrix.m[1][0] = 0
        overlay_matrix.m[1][1] = 1
        overlay_matrix.m[1][2] = 0
        overlay_matrix.m[1][3] = 0.08

        overlay_matrix.m[2][0] = 0.0
        overlay_matrix.m[2][1] = 0.0
        overlay_matrix.m[2][2] = 1.0
        overlay_matrix.m[2][3] = -.3

    while True:
        if (args.stereo):
            # Send current position for each eye
            vr_server_send_coords_stereo(vr_server_handle)

            # Read from pipe
            png_bytes_l = vr_client_read_image(vr_client_handle)
            png_bytes_r = vr_client_read_image(vr_client_handle)

            # Create textures
            left_texture = get_overlay_texture_from_bytes(png_bytes_l, args.size, args.size)
            right_texture = get_overlay_texture_from_bytes(png_bytes_r, args.size, args.size)
            
            render_frame(left_texture, right_texture)
        else:
            # Send current position for headset
            vr_server_send_coords(vr_server_handle)

            # Read from pipe
            png_bytes = vr_client_read_image(vr_client_handle)

            # Create texture
            texture = get_overlay_texture_from_bytes(png_bytes, args.size, args.size)
            
            # Display overlay
            overlay.setOverlayTransformTrackedDeviceRelative(overlay_obj, 0, overlay_matrix)
            overlay.setOverlayTexture(overlay_obj, texture)
    
if __name__ == '__main__':
    parse_args()
    main()
