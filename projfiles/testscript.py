for idx in args.screenshot_frames:
    start_time = time.time()
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
    write_image(outname, imagfor idx in args.screenshot_frames:
    start_time = time.time()
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
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)e)
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)