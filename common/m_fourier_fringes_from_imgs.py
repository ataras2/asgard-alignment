import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage as nd

from tqdm import tqdm


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fourier Fringes from Images Script")
    parser.add_argument(
        "--savepath", type=str, required=True, help="Path to the image data (type: str)"
    )
    parser.add_argument(
        "--pswidth",
        type=int,
        default=24,
        help="Width of the power spectrum (default: 24) (type: int)",
    )
    parser.add_argument(
        "--xcrop",
        type=int,
        default=512,
        help="Number of pixels to crop from the x-axis (default: 512) (type: int)",
    )
    parser.add_argument(
        "--ycrop",
        type=int,
        default=512,
        help="Number of pixels to crop from the y-axis (default: 512) (type: int)",
    )
    args = parser.parse_args()

    pth = args.savepath
    pswidth = args.pswidth
    xcrop = args.xcrop
    ycrop = args.ycrop

    log_file = open(os.path.join(pth, "analysis_log.txt"), "w")

    def log_and_print(message):
        print(message)
        log_file.write(message + "\n")

    log_and_print(f"path exists: {os.path.exists(pth)}")

    # Load image data
    data = np.load(os.path.join(pth, "img_stack.npz"))

    ims = data["img_stack"]
    positions = data["positions"]
    imshape = ims.shape
    nsets = imshape[0]
    ims_per_set = imshape[1]

    # crop
    # ims = ims[:, :, 460:770, 850:1250]

    print(ims.shape)

    imshape = ims.shape

    ims = ims.reshape((nsets * ims_per_set, imshape[2], imshape[3]))[:, :ycrop, :xcrop]

    print(ims.shape)
    # crop:
    # ims = ims[:, 600:900, 150:400]

    max_pwr = np.zeros((nsets * ims_per_set))

    im_av = np.zeros_like(ims)
    print()
    for i in tqdm(range(nsets * ims_per_set)):
        # print(f"Processing image {i} of {nsets * ims_per_set}\r", end="")
        imps = np.abs(np.fft.rfft2(ims[i])) ** 2
        imps[:pswidth, :pswidth] = 0
        imps[-pswidth:, :pswidth] = 0
        imps_smoothed = nd.gaussian_filter(imps, 5)
        max_pwr[i] = np.max(imps_smoothed)

    max_pwr = np.reshape(max_pwr, (nsets, ims_per_set))
    # print(f"Found max power at {np.argmax(max_pwr)}")

    plt.figure(1)
    plt.plot(positions, np.sum(max_pwr, axis=1))

    plt.savefig(os.path.join(pth, "power_plot.png"))

    log_and_print(f"Maximum power: {np.max(max_pwr)}")
    log_and_print(
        f"Found at position {positions[int(np.argmax(max_pwr.flatten())/ims_per_set)]}"
    )

    bias = np.median(max_pwr)
    mad = np.median(np.abs(max_pwr - bias))
    snr = (np.max(max_pwr) - bias) / (1.4826 * mad)

    log_and_print(f"Maximum signal to noise ratio: {snr}")

    plt.figure(2)
    plt.imshow(ims[np.argmax(max_pwr)])

    plt.savefig(os.path.join(pth, "max_power_image.png"))

    log_file.close()

    # import pdb

    # pdb.set_trace()


if __name__ == "__main__":
    main()
