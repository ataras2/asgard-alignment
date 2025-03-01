
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,  median_filter
from scipy.optimize import leastsq
import toml  # Make sure to install via `pip install toml` if needed
import argparse
import os
import json
import time

from xaosim.shmlib import shm

try:
    from asgard_alignment import controllino as co
    myco = co.Controllino('172.16.8.200')
    controllino_available = True
    print('controllino connected')
    
except:
    print('WARNING Controllino cannot connect. WILL NOT MOVE SOURCE OUT FOR DARK')
    controllino_available = False 


def detect_pupil(image, sigma=2, threshold=0.5, plot=True, savepath=None):
    """
    Detects an elliptical pupil (with possible rotation) in a cropped image using edge detection 
    and least-squares fitting. Returns both the ellipse parameters and a pupil mask.

    The ellipse is modeled by:

        ((x - cx)*cos(theta) + (y - cy)*sin(theta))^2 / a^2 +
        (-(x - cx)*sin(theta) + (y - cy)*cos(theta))^2 / b^2 = 1

    Parameters:
        image (2D array): Cropped grayscale image containing a single pupil.
        sigma (float): Standard deviation for Gaussian smoothing.
        threshold (float): Threshold factor for edge detection.
        plot (bool): If True, displays the image with the fitted ellipse overlay.
        savepath (str): If provided, the plot is saved to this path.

    Returns:
        (center_x, center_y, a, b, theta, pupil_mask)
          where (center_x, center_y) is the ellipse center,
                a and b are the semimajor and semiminor axes,
                theta is the rotation angle in radians,
                pupil_mask is a 2D boolean array (True = inside ellipse).
    """
    # Normalize the image
    image = image / image.max()
    
    # Smooth the image
    smoothed_image = gaussian_filter(image, sigma=sigma)
    
    # Compute gradients (Sobel-like edge detection)
    grad_x = np.gradient(smoothed_image, axis=1)
    grad_y = np.gradient(smoothed_image, axis=0)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold edges to create a binary mask
    binary_edges = edges > (threshold * edges.max())
    
    # Get edge pixel coordinates
    y_coords, x_coords = np.nonzero(binary_edges)
    
    # Initial guess: center from mean, radius from average distance, and theta = 0.
    def initial_guess(x, y):
        center_x = np.mean(x)
        center_y = np.mean(y)
        r_init = np.sqrt(np.mean((x - center_x)**2 + (y - center_y)**2))
        return center_x, center_y, r_init, r_init, 0.0  # (cx, cy, a, b, theta)
    
    # Ellipse model function with rotation.
    def ellipse_model(params, x, y):
        cx, cy, a, b, theta = params
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_shift = x - cx
        y_shift = y - cy
        xp =  cos_t * x_shift + sin_t * y_shift
        yp = -sin_t * x_shift + cos_t * y_shift
        # Model: xp^2/a^2 + yp^2/b^2 = 1 => residual = sqrt(...) - 1
        return np.sqrt((xp/a)**2 + (yp/b)**2) - 1.0

    # Fit via least squares.
    guess = initial_guess(x_coords, y_coords)
    result, _ = leastsq(ellipse_model, guess, args=(x_coords, y_coords))
    center_x, center_y, a, b, theta = result
    
    # Create a boolean pupil mask for the fitted ellipse
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_shift = xx - center_x
    y_shift = yy - center_y
    xp = cos_t * x_shift + sin_t * y_shift
    yp = -sin_t * x_shift + cos_t * y_shift
    pupil_mask = (xp/a)**2 + (yp/b)**2 <= 1

    if plot:
        # Overlay for visualization
        overlay = np.zeros_like(image)
        overlay[pupil_mask] = 1
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="gray", origin="upper")
        plt.contour(binary_edges, colors="cyan", linewidths=1)
        plt.contour(overlay, colors="red", linewidths=1)
        plt.scatter(center_x, center_y, color="blue", marker="+")
        plt.title("Detected Pupil with Fitted Ellipse")
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()
    
    return center_x, center_y, a, b, theta, pupil_mask


def compute_affine_from_ellipse(ell1, ell2):
    """
    Computes an affine transformation that maps points from frame 1 to frame 2
    using the ellipse parameters from each frame.

    ell1, ell2: (cx, cy, a, b, theta, pupil_mask) or (cx, cy, a, b, theta) 
                The pupil_mask is ignored here; only the numeric parameters are used.

    Returns:
      T (ndarray): 3x3 affine transformation matrix mapping frame 1 -> frame 2.
      T_inv (ndarray): 3x3 inverse transformation matrix.
    """
    # Unpack numeric ellipse parameters
    cx1, cy1, a1, b1, theta1 = ell1[:5]
    cx2, cy2, a2, b2, theta2 = ell2[:5]
    
    # Rotation matrices
    R1 = np.array([[np.cos(theta1), -np.sin(theta1)],
                   [np.sin(theta1),  np.cos(theta1)]])
    R2 = np.array([[np.cos(theta2), -np.sin(theta2)],
                   [np.sin(theta2),  np.cos(theta2)]])
    
    # Relative scaling matrix
    S = np.diag([a2/a1, b2/b1])
    
    # Linear part of the transform
    A = R2 @ S @ np.linalg.inv(R1)
    
    # Translation: map center of ellipse 1 to center of ellipse 2
    c1 = np.array([cx1, cy1])
    c2 = np.array([cx2, cy2])
    t = c2 - A @ c1
    
    # Build full homogeneous 3x3 matrix
    T = np.array([
        [A[0,0], A[0,1], t[0]],
        [A[1,0], A[1,1], t[1]],
        [0,      0,      1     ]
    ])
    T_inv = np.linalg.inv(T)
    
    return T, T_inv


def warp_image_manual(image_in, T, output_shape=None, method='nearest'):
    """
    Manually warp an image using a 3x3 affine transform matrix T that maps
    input (x_in, y_in) -> output (x_out, y_out).
    """
    if output_shape is None:
        output_shape = image_in.shape

    T_inv = np.linalg.inv(T)
    out_height, out_width = output_shape
    image_out = np.zeros((out_height, out_width), dtype=image_in.dtype)

    for y_out in range(out_height):
        for x_out in range(out_width):
            p_out = np.array([x_out, y_out, 1.0])
            p_in = T_inv @ p_out
            x_in, y_in = p_in[0], p_in[1]

            if method == 'nearest':
                x_nn = int(round(x_in))
                y_nn = int(round(y_in))
                if (0 <= x_nn < image_in.shape[1]) and (0 <= y_nn < image_in.shape[0]):
                    image_out[y_out, x_out] = image_in[y_nn, x_nn]

            elif method == 'bilinear':
                x0 = int(np.floor(x_in))
                y0 = int(np.floor(y_in))
                dx = x_in - x0
                dy = y_in - y0
                if (0 <= x0 < image_in.shape[1]-1) and (0 <= y0 < image_in.shape[0]-1):
                    I00 = image_in[y0,   x0  ]
                    I01 = image_in[y0,   x0+1]
                    I10 = image_in[y0+1, x0  ]
                    I11 = image_in[y0+1, x0+1]
                    Ixy = (1 - dx)*(1 - dy)*I00 + dx*(1 - dy)*I01 \
                          + (1 - dx)*dy*I10     + dx*dy*I11
                    image_out[y_out, x_out] = Ixy

    return image_out


def save_pupil_data_toml(beam_id, ellipse_params, toml_path):
    """
    Writes the pupil ellipse parameters and boolean pupil mask to a TOML file.

    The TOML structure will look like:

    [beam 1]
      [beam 1.pupil ellipse fit]
      center_x = ...
      center_y = ...
      a = ...
      b = ...
      theta = ...

      [beam 1.pupil mask]
      mask = [ [true, false, ...], [...], ...]

    Parameters:
        beam_id (int): The beam number (e.g. 1, 2, etc.)
        ellipse_params (tuple): (cx, cy, a, b, theta, pupil_mask)
        toml_path (str): Path to the TOML file to write.
    """
    cx, cy, a, b, theta, pupil_mask = ellipse_params

    # Convert the boolean mask to a nested list of booleans
    mask_list = pupil_mask.tolist()  # shape => Nx x Ny of True/False

    new_data = {
        f"beam{beam_id}": {
            "pupil_ellipse_fit": {
                "center_x": float(cx),
                "center_y": float(cy),
                "a": float(a),
                "b": float(b),
                "theta": float(theta),
            },
            "pupil_mask": {
                "mask": mask_list
            }
        }
    }

    # Check if file exists; if so, load and update.
    if os.path.exists(toml_path):
        try:
            current_data = toml.load(toml_path)
        except Exception as e:
            print(f"Error loading TOML file: {e}")
            current_data = {}
    else:
        current_data = {}

    # Update current data with new_data (beam specific)
    #current_data.update(new_data)
    current_data = recursive_update(current_data, new_data)
    # Write the updated data back to the TOML file.
    with open(toml_path, "w") as f:
        toml.dump(current_data, f)


def recursive_update(orig, new):
    """
    Recursively update dictionary 'orig' with 'new' without overwriting sub-dictionaries.
    """
    for key, value in new.items():
        if (key in orig and isinstance(orig[key], dict) 
            and isinstance(value, dict)):
            recursive_update(orig[key], value)
        else:
            orig[key] = value
    return orig


def get_bad_pixel_indicies( imgs, std_threshold = 20, mean_threshold=6):
    # To get bad pixels we just take a bunch of images and look at pixel variance and mean

    ## Identify bad pixels
    mean_frame = np.mean(imgs, axis=0)
    std_frame = np.std(imgs, axis=0)

    global_mean = np.mean(mean_frame)
    global_std = np.std(mean_frame)
    bad_pixel_map = (np.abs(mean_frame - global_mean) > mean_threshold * global_std) | (std_frame > std_threshold * np.median(std_frame))

    return bad_pixel_map


def interpolate_bad_pixels(img, bad_pixel_map):
    filtered_image = img.copy()
    filtered_image[bad_pixel_map] = median_filter(img, size=3)[bad_pixel_map]
    return filtered_image






parser = argparse.ArgumentParser(description="Baldr Pupil Fit Configuration.")

# Camera shared memory path
parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
)

# TOML file path; default is relative to the current file's directory.
default_toml = os.path.join("config_files", "baldr_config_#.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")
parser.add_argument(
    "--toml_file",
    type=str,
    default=default_toml,
    help="TOML file pattern (replace # with beam) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_ids",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[1, 2, 3, 4],
    help="Comma-separated beam IDs. Default: 1,2,3,4"
)

# Plot: default is True, with an option to disable.
parser.add_argument(
    "--plot", 
    dest="plot",
    action="store_true",
    default=True,
    help="Enable plotting (default: True)"
)


args=parser.parse_args()


# pupil_coords 

# with open( "config_files/file_paths.json") as f:
#     default_path_dict = json.load(f)

#baldr_pupils_path = toml_file#default_path_dict["pupil_crop_toml"] #"/home/asg/Progs/repos/asgard-alignment/config_files/baldr_pupils_coords.json"
# Load the TOML file
with open(args.toml_file.replace('#',f'{args.beam_ids[0]}') ) as file:
    pupildata = toml.load(file)

    # Extract the "baldr_pupils" section
    baldr_pupils = pupildata.get("baldr_pupils", {})



# shm path to FULL () imagr 
mySHM = shm(args.global_camera_shm)

img_raw = mySHM.get_data()

# try get dark and build bad pixel mask 
if controllino_available:

    myco.turn_off("SBB")
    time.sleep(2)
    
    dark_raw = mySHM.get_data()

    myco.turn_on("SBB")
    time.sleep(2)

    bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
else:
    dark_raw = mySHM.get_data()

    bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
# reduce (bad pixel mask)



if hasattr(img_raw, "__len__"):
    if len( np.array(img_raw).shape ) == 3:
        img=np.mean(img_raw, axis=0)
    elif len( np.array(img_raw).shape ) == 2:
        img=img_raw
    else:
        raise UserWarning(f"image in shared memory address {args.target} doresnt have 2 or 3 dimensions. not a valid image(s)")
else:
    raise UserWarning(f"image in shared memory address {args.target} doresnt length attribute!")

for beam_id in args.beam_ids:
    # get the cropped image 
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])

    # mask 
    ell1 = detect_pupil(cropped_img, sigma=2, threshold=0.5, plot=args.plot,savepath=f"delme{beam_id}.png")

    save_pupil_data_toml(beam_id=beam_id, ellipse_params=ell1, toml_path=args.toml_file.replace('#',f'{beam_id}'))




# done

# ----------------------------
# Example simulation / verification
# ----------------------------
# in the real system, want argparse to run from command line with argunments:
# camera shared memory path. default: /dev/shm/cred1.im.shm
# toml file to write to / edit : default : *current file path*/../config_files/baldr_config.toml
# beam ids. default: [1,2,3,4]
# plot. boolean indicating if we should plot. stored true


# # get a global image . if shape == 3 take mean. else keep image. readin pupil_crop and extract each beam. 
# #  for each beam run  detect_pupil(img_b#, sigma=2, threshold=0.5, plot=True)
# # then write results to toml file baldr_config.toml (also keep baldr pupil coords here) 


# # baldr_config.toml

# [pupil_coords]

# [pupil mask] - i need this for defing quadrants in the next step of fine alignment

# [pupil ellipse fit]

if __name__ == "__main__":

    print('done')

    # below is a simulation of this algorithm and defining aline transforms of pupils
    
    # Nx, Ny = 24, 24
    # R = 6
    # r = 1
    # noise_rms = 0.1

    # x_pixels = np.arange(Nx)
    # y_pixels = np.arange(Ny)
    # X, Y = np.meshgrid(x_pixels, y_pixels)

    # # Create a base pupil image (Frame 1)
    # pup = np.zeros((Ny, Nx))
    # filt = ((X - Nx//2)**2 + (Y/1.2 - Ny//2)**2 < R**2) & \
    #        ((X - Nx//2)**2 + (Y/1.2 - Ny//2)**2 > r**2)
    # pup[filt] = 1
    # img1 = pup + noise_rms * np.random.randn(Ny, Nx)

    # plt.figure()
    # plt.imshow(img1, cmap="gray")
    # plt.title("Frame 1")
    # plt.show()

    # # Define a known affine transform: scale, rotation, translation
    # angle_deg = 15
    # angle_rad = np.deg2rad(angle_deg)
    # scale_x = 1.2
    # scale_y = 0.8
    # tx, ty = (2, -1)

    # # Build T_known (input -> output)
    # cos_a = np.cos(angle_rad)
    # sin_a = np.sin(angle_rad)
    # T_known = np.array([
    #     [scale_x * cos_a, -scale_y * sin_a, tx],
    #     [scale_x * sin_a,  scale_y * cos_a, ty],
    #     [0,               0,               1 ]
    # ])

    # # Generate "Frame 2" by warping Frame 1 with T_known
    # img2 = warp_image_manual(img1, T_known, output_shape=(Ny, Nx), method='bilinear')
    # img2 += noise_rms * np.random.randn(Ny, Nx)

    # plt.figure()
    # plt.imshow(img2, cmap="gray")
    # plt.title("Frame 2 (Simulated)")
    # plt.show()

    # # Detect ellipses in both frames (returns ellipse params + pupil mask)
    # ell1 = detect_pupil(img1, sigma=2, threshold=0.5, plot=True)
    # ell2 = detect_pupil(img2, sigma=2, threshold=0.5, plot=True)
    # print("Ellipse 1:", ell1[:5])
    # print("Ellipse 2:", ell2[:5])

    # # Example: save beam=1 data from the first frame
    # save_pupil_data_toml(beam_id=1, ellipse_params=ell1, toml_path="beam1_test.toml")

    # # 5) Compute the estimated transform from ellipse parameters
    # T_est, T_est_inv = compute_affine_from_ellipse(ell1, ell2)
    # print("T_est:\n", T_est)
    # print("T_est_inv:\n", T_est_inv)

    # # 6) Warp Frame 1 with the estimated transform
    # img1_to_2 = warp_image_manual(img1, T_est, output_shape=(Ny, Nx), method='nearest')

    # # 7) Display results
    # plt.figure(figsize=(12,4))
    # plt.subplot(1,3,1)
    # plt.imshow(img2, cmap="gray")
    # plt.title("Frame 2 (Simulated)")

    # plt.subplot(1,3,2)
    # plt.imshow(img1_to_2, cmap="gray")
    # plt.title("Frame 1 warped by T_est")

    # plt.subplot(1,3,3)
    # resid = img2 - img1_to_2
    # plt.imshow(resid, cmap="gray")
    # plt.title("Residual (Frame 2 - Warped Frame 1)")
    # plt.show()






