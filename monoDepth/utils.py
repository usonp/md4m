import json

import cv2
import numpy as np
from numpy.polynomial import Polynomial as Poly
import matplotlib.pyplot as plt

import torch


if __name__ == '__main__':
    print('Please do not run this script, it only contains useful functions for other scripts')
    exit(1)

def load_model(depth_model, encoder='vitl'):

    # Dummy
    model = None
    transform = None

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f'Loading {depth_model} model to device {DEVICE}...')

    # Depth Anything 2
    if "DA2" in depth_model:
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        # Separate between metric and standard
        checkpoint_path = "/home/jup/DLFVV_integration/FVVAnything/DA_checkpoints"
        if "metric" in depth_model:
            from depth_anything_v2_metric.dpt import DepthAnythingV2
            # Using indoor weights
            dataset = 'hypersim'
            max_depth = 20
            depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
            depth_anything.load_state_dict(torch.load(f'{checkpoint_path}/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
        else:
            from depth_anything_v2.dpt import DepthAnythingV2
            depth_anything = DepthAnythingV2(**model_configs[encoder])
            depth_anything.load_state_dict(torch.load(f'{checkpoint_path}/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model =  depth_anything.to(DEVICE).eval()

    # Depth Pro
    elif "Depth_pro" in depth_model:
        # Import the package
        from depth_pro import create_model_and_transforms
        from depth_pro.depth_pro import DepthProConfig

        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri="/home/jup/fvv-live-deep-learning/MonoDepth/ml-depth-pro/checkpoints/depth_pro.pt",
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        model, transform = create_model_and_transforms(
            config=config,
            device=DEVICE,
            # precision=torch.float32,
            precision=torch.half,
        )
        model.eval()

    # UniDepth
    elif "UniDepth" in depth_model:

        # Import the package
        from unidepth.models import UniDepthV2
        name = f"unidepth-v2-{encoder}14"
        # Load from HugginFace
        model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
        model.resolution_level = 9
        model.interpolation_mode = "bilinear"
        model = model.to(DEVICE).eval()
    else:
        raise ValueError(f'Invalid depth model {depth_model}')
    
    print(f'Finished loading!')
    return model, transform


def crop_to_closest_14(mat):
    h, w = mat.shape[:2]
    new_h = (h // 14) * 14
    new_w = (w // 14) * 14
    # Compute the difference to crop equally from all sides (if needed)
    h_offset = (h - new_h) // 2
    w_offset = (w - new_w) // 2
    cropped_mat = mat[h_offset:h_offset+new_h, w_offset:w_offset+new_w]
    return cropped_mat


def pad_to_closest_14(mat, padding='symmetric'):
    h, w = mat.shape[:2]
    new_h = (h + 13) // 14 * 14  # Adding 13 ensures rounding up to the nearest multiple of 14
    new_w = (w + 13) // 14 * 14 
    pad_h = new_h - h
    pad_w = new_w - w
    padded_mat = np.pad(mat, ((0, pad_h), (0, pad_w), (0, 0)), mode=padding)
    return padded_mat


def compute_depth(color, depth_model, model, intrinsics, transform=None, device='cuda', padding='symmetric'):
    # Depth Anything 2
    if 'DA2' in depth_model:
        original_shape = color.shape
        color = crop_to_closest_14(color)
        cropped_shape = color.shape
            
        depth = model.infer_image(color, color.shape[0])

        # Pad the result back to the original resolution
        padding_x = (original_shape[0] - cropped_shape[0]) // 2
        padding_y = (original_shape[1] - cropped_shape[1]) // 2
        depth = np.pad(depth, [(padding_x, padding_x), (padding_y, padding_y)], padding)

        if 'metric' in depth_model:
            return depth
        else:
            # Depth is actually disp, we transform it to depth
            depth = 1.0 / np.clip(depth, 1e-4, 1e4)
            # depth *= (original_shape[0] / intrinsics['f_x'])

    # Depth Pro
    elif depth_model == 'Depth_pro':
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        f_px = torch.tensor([intrinsics['f_x']]).to(device)
        color = transform(color)
        prediction = model.infer(color, f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy().squeeze()  # Depth in [m].

    # UniDepth
    elif depth_model == 'UniDepth':
        from unidepth.utils.camera import Pinhole
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = torch.from_numpy(color).permute(2, 0, 1)
        intrinsics = Pinhole(
            params=torch.tensor([
                intrinsics['f_x'],
                intrinsics['f_y'],
                intrinsics['c_x'],
                intrinsics['c_y']
            ])
        )
        prediction = model.infer(color, camera=intrinsics)
        depth = prediction["depth"].detach().cpu().numpy().squeeze()  # Depth in [m].
    else:
        raise ValueError(f'Invalid depth model {depth_model}')

    return depth


def show_depth(window, depth, Z_near=None, Z_far=None, colormap=cv2.COLORMAP_INFERNO):
    if Z_near is None:
        max_val = depth.max()
    else:
        min_val = Z_far
    if Z_far is None:
        min_val = depth.min()
    else:
        max_val = Z_near
    depth_uint8 = np.clip((depth - min_val) / (max_val - min_val) * 255, 0, 255).astype('uint8')
    show_depth = cv2.applyColorMap(depth_uint8, colormap)
    cv2.imshow(window, show_depth)


def fix_depth_linear(depth, regression_data):
    
    b = regression_data.get('slope')
    c = regression_data.get('offset')

    return (b * depth + c) * depth


def normalize_depth_16bits(depth, Z_near, Z_far):
    depth = np.clip(depth.astype(np.float64), Z_near, Z_far)
    depth = (depth - Z_near) / (Z_far - Z_near)  # Normalize to [0, 1]
    depth = depth * 65535  # Scale to [0, 65535] for uint16
    return depth.astype(np.uint16)


# Note that the x and y are flipped compared to OpenCV
def bilinear_interpolation(img, x, y):
    """Interpolate the value of a grayscale image at subpixel coordinates (x, y)."""
    h, w = img.shape

    # Get the four surrounding pixel coordinates
    x1, x2 = int(np.floor(x)), int(np.ceil(x + 1e-10))
    y1, y2 = int(np.floor(y)), int(np.ceil(y + 1e-10))

    # Ensure the points are within bounds
    if x1 < 0 or x2 >= w or y1 < 0 or y2 >= h:
        raise ValueError("Point is outside the image bounds")

    # Get pixel values at the four corners
    q11 = img[y1, x1]
    q21 = img[y1, x2]
    q12 = img[y2, x1]
    q22 = img[y2, x2]

    if (x2 - x1) == 0 or (y2 - y1) == 0:
        print('Invalid pixel coordinate found in interpolation:')
        print(f'({x}, {y}) -> ({x1},{y1}) ({x2},{y2})')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


def intrinsic_dictionary_from_file(IntrisicsPath, rectified_images=False):
    if rectified_images:
        intrinsics_key = 'new_camera_matrix'
    else:
        intrinsics_key = 'camera_matrix'

    IntrisicDict = {}
    with open(IntrisicsPath, 'r') as f:
        intrinsics = json.load(f)
    for i in intrinsics.keys():
        # camera_matrix/new_camera_matrix if we rectify the images
        IntrisicDict[int(i)] = {}
        IntrisicDict[int(i)]['f_x'] = intrinsics[i][intrinsics_key][0][0]
        IntrisicDict[int(i)]['f_y'] = intrinsics[i][intrinsics_key][1][1]
        IntrisicDict[int(i)]['c_x'] = intrinsics[i][intrinsics_key][0][2]
        IntrisicDict[int(i)]['c_y'] = intrinsics[i][intrinsics_key][1][2]
    return IntrisicDict


def coolPrint(message, color="cyan"):
    colors = {
        "black": "\033[1;30m",
        "red": "\033[1;31m",
        "green": "\033[1;32m",
        "yellow": "\033[1;33m",
        "blue": "\033[1;34m",
        "magenta": "\033[1;35m",
        "cyan": "\033[1;36m",
        "white": "\033[1;37m",
        "reset": "\033[0m"
    }
    
    color_code = colors.get(color.lower(), colors["reset"])
    print(f"{color_code}{message}{colors['reset']}")