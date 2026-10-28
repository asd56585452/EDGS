import os
import re

def is_cam_match(cam_name, test_view_pattern):
    c_base = os.path.splitext(cam_name)[0].lower()
    p_base = os.path.splitext(test_view_pattern)[0].lower()
    if c_base == p_base:
        return True
    c_m = re.match(r"^([a-zA-Z_-]*)(\d+)$", c_base)
    p_m = re.match(r"^([a-zA-Z_-]*)(\d+)$", p_base)
    if c_m and p_m and c_m.group(1) == p_m.group(1):
        if int(c_m.group(2)) == int(p_m.group(2)):
            return True
    return False

def scene_cameras_train_test_split(scene, verbose=False, test_views=None):
    """
    Iterate over resolutions in the scene. If test_views is specified (e.g. 'cam00' or 'cam0'),
    isolate matching cameras into the test set and keep the rest in train.
    Otherwise, if no test cameras exist, extract every 8th camera to test (Kerbl et al.).
    """
    if verbose: print("Preparing train and test sets split...")
    if isinstance(test_views, str):
        test_views = [tv.strip() for tv in test_views.split(",") if tv.strip()]

    for resolution in scene.train_cameras.keys():
        if len(scene.test_cameras[resolution]) == 0:
            all_cams = scene.train_cameras[resolution]
            if test_views:
                test_cams = []
                train_cams = []
                for cam in all_cams:
                    cam_name = getattr(cam, "image_name", "")
                    if any(is_cam_match(cam_name, tv) for tv in test_views):
                        test_cams.append(cam)
                    else:
                        train_cams.append(cam)
                scene.test_cameras[resolution] = test_cams
                scene.train_cameras[resolution] = train_cams
                if verbose:
                    test_names = [getattr(c, "image_name", str(i)) for i, c in enumerate(test_cams)]
                    print(f"Isolated test views {test_names} ({len(test_cams)} cameras). Train set has {len(train_cams)} cameras.")
            else:
                if verbose:
                    print(f"Found no test_cameras for resolution {resolution}. Move every 8th camera out ouf total "+\
                          f"{len(all_cams)} train cameras to the test set now")
                N = len(all_cams)
                scene.test_cameras[resolution] = [all_cams[idx] for idx in range(0, N) 
                                                  if idx % 8 == 0]
                scene.train_cameras[resolution] = [all_cams[idx] for idx in range(0, N)
                                                   if idx % 8 != 0]
                if verbose:
                    print(f"Done. Now train and test sets contain each {len(scene.train_cameras[resolution])} and " + \
                          f"{len(scene.test_cameras[resolution])} cameras respectively.")

    return
