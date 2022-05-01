//
// Created by Cyrus on 4/24/2022.
//

#include "openvr-master/headers/openvr.h"
#include <stdio.h>

using namespace vr;

void check_error(int line, EVRInitError error) { if (error != 0) printf("%d: error %s\n", line, VR_GetVRInitErrorAsSymbol(error)); }

int main(int argc, char **argv) { (void) argc; (void) argv;
    EVRInitError error;
    VR_Init(&error, vr::VRApplication_Overlay);
    check_error(__LINE__, error);

    VROverlayHandle_t handle;
    VROverlay()->CreateOverlay ("image", "image", &handle); /* key has to be unique, name doesn't matter */
    VROverlay()->SetOverlayFromFile(handle, "/cat.jpg");
    VROverlay()->SetOverlayWidthInMeters(handle, 3);
    VROverlay()->ShowOverlay(handle);

    vr::HmdMatrix34_t transform = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f, -2.0f
    };
    VROverlay()->SetOverlayTransformAbsolute(handle, TrackingUniverseStanding, &transform);

    while (true) { }
    return 0;
}