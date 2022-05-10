#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from xmlrpc.client import Boolean
import commentjson as json

import numpy as np

from common import *
from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf

from tqdm import tqdm

import pyngp as ngp # noqa
import win32pipe, win32file, pywintypes

def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")
	parser.add_argument("--size", type=int, default=0, help="Sets height and width simultaneously")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")

	parser.add_argument("--vr", default=False, help="enable vr", action='store_true')
	parser.add_argument("--stereo", default=False, help="Enables stereo imaging for VR", action='store_true')
	parser.add_argument("--translation_scale", default=3, type=float, help="Scales positional movement in VR")

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()

	if args.size:
		args.width = args.size
		args.height = args.size

	if args.mode == "":
		if args.scene in scenes_sdf:
			args.mode = "sdf"
		elif args.scene in scenes_nerf:
			args.mode = "nerf"
		elif args.scene in scenes_image:
			args.mode = "image"
		elif args.scene in scenes_volume:
			args.mode = "volume"
		else:
			print(args.scene)
			print(scenes_image)
			raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	if args.mode == "sdf":
		mode = ngp.TestbedMode.Sdf
		configs_dir = os.path.join(ROOT_DIR, "configs", "sdf")
		scenes = scenes_sdf
	elif args.mode == "volume":
		mode = ngp.TestbedMode.Volume
		configs_dir = os.path.join(ROOT_DIR, "configs", "volume")
		scenes = scenes_volume
	elif args.mode == "nerf":
		mode = ngp.TestbedMode.Nerf
		configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
		scenes = scenes_nerf
	elif args.mode == "image":
		mode = ngp.TestbedMode.Image
		configs_dir = os.path.join(ROOT_DIR, "configs", "image")
		scenes = scenes_image

	base_network = os.path.join(configs_dir, "base.json")
	if args.scene in scenes:
		network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
		base_network = os.path.join(configs_dir, network+".json")
	network = args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)


	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)

	if args.mode == "sdf":
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		scene=args.scene
		if not os.path.exists(args.scene) and args.scene in scenes:
			scene = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset"])
		testbed.load_training_data(scene)

	if args.load_snapshot:
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)
	else:
		testbed.reload_network_from_file(network)

	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw*sh > 1920*1080*4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh)

	testbed.shall_train = args.train if args.gui else True

	testbed.nerf.render_with_camera_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]
	if args.mode == "sdf":
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Optionally match nerf paper behaviour and train on a
		# fixed white bg. We prefer training on random BG colors.
		# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
		# testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps
	if n_steps < 0:
		n_steps = 100000

	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="step") as t:
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				t.update(testbed.training_step - old_training_step)
				t.set_postfix(loss=testbed.loss)
				old_training_step = testbed.training_step

	if args.save_snapshot:
		print("Saving snapshot ", args.save_snapshot)
		testbed.save_snapshot(args.save_snapshot, False)

	if args.test_transforms:
		print("Evaluating test transforms from ", args.test_transforms)
		with open(args.test_transforms) as f:
			test_transforms = json.load(f)
		data_dir=os.path.dirname(args.test_transforms)
		totmse = 0
		totpsnr = 0
		totssim = 0
		totcount = 0
		minpsnr = 1000
		maxpsnr = 0

		# Evaluate metrics on black background
		testbed.background_color = [0.0, 0.0, 0.0, 1.0]

		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.rendering_min_transmittance = 1e-4

		testbed.fov_axis = 0
		testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
		testbed.shall_train = False

		with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
			for i, frame in t:
				p = frame["file_path"]
				if "." not in p:
					p = p + ".png"
				ref_fname = os.path.join(data_dir, p)
				if not os.path.isfile(ref_fname):
					ref_fname = os.path.join(data_dir, p + ".png")
					if not os.path.isfile(ref_fname):
						ref_fname = os.path.join(data_dir, p + ".jpg")
						if not os.path.isfile(ref_fname):
							ref_fname = os.path.join(data_dir, p + ".jpeg")
							if not os.path.isfile(ref_fname):
								ref_fname = os.path.join(data_dir, p + ".exr")

				ref_image = read_image(ref_fname)

				# NeRF blends with background colors in sRGB space, rather than first
				# transforming to linear space, blending there, and then converting back.
				# (See e.g. the PNG spec for more information on how the `alpha` channel
				# is always a linear quantity.)
				# The following lines of code reproduce NeRF's behavior (if enabled in
				# testbed) in order to make the numbers comparable.
				if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
					# Since sRGB conversion is non-linear, alpha must be factored out of it
					ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
					ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
					ref_image[...,:3] *= ref_image[...,3:4]
					ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
					ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])

				if i == 0:
					write_image("ref.png", ref_image)

				testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
				image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

				if i == 0:
					write_image("out.png", image)

				diffimg = np.absolute(image - ref_image)
				diffimg[...,3:4] = 1.0
				if i == 0:
					write_image("diff.png", diffimg)

				A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
				R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
				mse = float(compute_error("MSE", A, R))
				ssim = float(compute_error("SSIM", A, R))
				totssim += ssim
				totmse += mse
				psnr = mse2psnr(mse)
				totpsnr += psnr
				minpsnr = psnr if psnr<minpsnr else minpsnr
				maxpsnr = psnr if psnr>maxpsnr else maxpsnr
				totcount = totcount+1
				t.set_postfix(psnr = totpsnr/(totcount or 1))

		psnr_avgmse = mse2psnr(totmse/(totcount or 1))
		psnr = totpsnr/(totcount or 1)
		ssim = totssim/(totcount or 1)
		print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")

	if args.save_mesh:
		res = args.marching_cubes_res or 256
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}]")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res])

	if args.width:
		if ref_transforms:
			testbed.fov_axis = 0
			testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
			if not args.screenshot_frames:
				args.screenshot_frames = range(len(ref_transforms["frames"]))
			print(args.screenshot_frames)
			for idx in args.screenshot_frames:
				print("idx hit")
				f = ref_transforms["frames"][int(idx)]
				cam_matrix = f["transform_matrix"]
				testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
				outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

				# Some NeRF datasets lack the .png suffix in the dataset metadata
				if not os.path.splitext(outname)[1]:
					outname = outname + ".png"

				print(f"rendering {outname}")
				image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
				os.makedirs(os.path.dirname(outname), exist_ok=True)
				write_image(outname, image)
				print("image", image)
				print("cam_matrix", cam_matrix)
		elif args.vr:
			outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
			print(f"Rendering {outname}.png")

			#create INSTANT SERVER
			pipe_to_vr = win32pipe.CreateNamedPipe(
			r'\\.\pipe\NGP_SERVER',
			win32pipe.PIPE_ACCESS_DUPLEX,
			win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
			1, 65536, 65536,
			0,
			None)

			#create INSTANT CLIENT
			handle = win32file.CreateFile(
			r'\\.\pipe\VR_SERVER',
			win32file.GENERIC_READ | win32file.GENERIC_WRITE,
			0,
			None,
			win32file.OPEN_EXISTING,
			0,
			None
            )
			
			#WAIT FOR CLIENT
			print("waiting for client")
			win32pipe.ConnectNamedPipe(pipe_to_vr, None)
			print("got client")
			
			initial_camera = np.array([[1, 0, 0, 0],
			                            [0, -1, 0, 0],
			 						   [0, 0, 1, 0]])

			# Translates coordinate systems
			negation_matrix = np.array([[-1, 1, -1, 1],
										[-1, 1, -1, 1],
										[-1, 1, -1, 1]])

			# Read initial headset position
			hmd_bytes = win32file.ReadFile(handle, 48)
			camera_coord_array = np.frombuffer(hmd_bytes[1], dtype=np.float32)
			vr_matrix = camera_coord_array.reshape((3, 4)).copy()

			vr_matrix[:, 3] *= args.translation_scale

			vr_matrix = np.multiply(vr_matrix, negation_matrix)

			# Calculate transform from initial position to initial camera position
			transform_matrix = initial_camera[:3, :3] @ np.linalg.inv(vr_matrix[:3, :3])

			while True:

				if args.stereo:
					#Read coords from pipe
					left_eye_bytes = win32file.ReadFile(handle, 48)
					right_eye_bytes = win32file.ReadFile(handle, 48)

					left_coord_array = np.frombuffer(left_eye_bytes[1], dtype=np.float32)
					right_coord_array = np.frombuffer(right_eye_bytes[1], dtype=np.float32)
					left_matrix = left_coord_array.reshape((3, 4))
					right_matrix = right_coord_array.reshape((3, 4))
					
					left_matrix = np.multiply(left_matrix, negation_matrix)
					right_matrix = np.multiply(right_matrix, negation_matrix)

					# Transform current position to camera space
					transformed_left_rotation = transform_matrix @ left_matrix[:3, :3]
					transformed_right_rotation = transform_matrix @ right_matrix[:3, :3]

					# Concatenate translations
					left_eye_matrix = np.hstack((transformed_left_rotation, np.array([left_matrix[:, 3]]).T))
					right_eye_matrix = np.hstack((transformed_right_rotation, np.array([right_matrix[:, 3]]).T))

					# Roll to account for nerf to ngp translation
					new_camera_l = np.roll(left_eye_matrix, 1, axis=0)
					new_camera_r = np.roll(right_eye_matrix, 1, axis=0)

					# Scale translations
					new_camera_l[:, 3] *= args.translation_scale
					new_camera_r[:, 3] *= args.translation_scale

					# Render images
					testbed.set_nerf_camera_matrix(new_camera_l)
					image_l = testbed.render(args.width, args.height, args.screenshot_spp, True)
					testbed.set_nerf_camera_matrix(new_camera_r)
					image_r = testbed.render(args.width, args.height, args.screenshot_spp, True)

					# Send left image
					# convert to bytes
					img = np.copy(image_l)
					# Unmultiply alpha
					img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
					img[...,0:3] = linear_to_srgb(img[...,0:3])
					img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
					#write bytes to pipe

					# Send right image
					win32file.WriteFile(pipe_to_vr, bytes(img))
					img = np.copy(image_r)
					# Unmultiply alpha
					img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
					img[...,0:3] = linear_to_srgb(img[...,0:3])
					img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
					#write bytes to pipe
					win32file.WriteFile(pipe_to_vr, bytes(img))

				else:
					#Read coords from pipe
					hmd_bytes = win32file.ReadFile(handle, 48)

					camera_coord_array = np.frombuffer(hmd_bytes[1], dtype=np.float32)
					vr_matrix = camera_coord_array.reshape((3, 4))
					
					vr_matrix = np.multiply(vr_matrix, negation_matrix)

					# Transform current position to camera space
					transformed_vr_matrix = transform_matrix @ vr_matrix[:3, :3]

					# Concatenate translations
					vr_matrix = np.hstack((transformed_vr_matrix, np.array([vr_matrix[:, 3]]).T))

					# Roll to account for nerf to ngp translation
					new_camera = np.roll(vr_matrix, 1, axis=0)

					# Scale translations
					new_camera[:, 3] *= args.translation_scale

					# Render images
					testbed.set_nerf_camera_matrix(new_camera)
					image = testbed.render(args.width, args.height, args.screenshot_spp, True)

					# Send image
					# convert to bytes
					img = np.copy(image)
					# Unmultiply alpha
					img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
					img[...,0:3] = linear_to_srgb(img[...,0:3])
					img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
					#write bytes to pipe
					win32file.WriteFile(pipe_to_vr, bytes(img))
