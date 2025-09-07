
import numpy as np 
from common import phasemask_centering_tool as pct
from pyBaldr import utilities as util
from typing import Dict, Tuple, Any
from scipy.optimize import curve_fit

def _gaussian2d(coords, A, x0, y0, sx, sy, theta, B):
    """
    Elliptical 2D Gaussian with rotation.
    coords: (x, y) 1D arrays of same length
    """
    x, y = coords
    ct, st = np.cos(theta), np.sin(theta)
    xr = (x - x0) * ct + (y - y0) * st
    yr = -(x - x0) * st + (y - y0) * ct
    return B + A * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

def fit_gaussian_on_res(res):
    """
    Fit a 2D Gaussian to the 'mean' values in `res` and add
    res[(x,y)]['gaussian_fit'] = G(x,y) for each sample.

    Returns (params, res) where params = {A, x0, y0, sx, sy, theta, B}.
    """
    # ---- gather data
    keys = list(res.keys())
    if not keys:
        raise ValueError("res is empty.")

    try:
        x = np.array([float(k[0]) for k in keys], dtype=float)
        y = np.array([float(k[1]) for k in keys], dtype=float)
    except Exception as e:
        raise ValueError("res keys must be (x,y) tuples.") from e

    try:
        z = np.array([float(res[k]["mean"]) for k in keys], dtype=float)
    except Exception as e:
        raise ValueError("res[k] must contain a numeric 'mean' field.") from e

    if np.any(~np.isfinite(z)):
        # keep only finite points
        m = np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
        x, y, z = x[m], y[m], z[m]
        keys = [kk for kk, keep in zip(keys, m) if keep]

    if x.size < 6:
        raise ValueError("Not enough points to fit a 2D Gaussian (need >= 6).")

    # ---- initial guesses
    B0 = float(np.median(z))
    A0 = float(max(np.max(z) - B0, 1e-9))
    w = np.clip(z - B0, 0.0, np.inf)
    if w.sum() > 0:
        x0 = float((x * w).sum() / w.sum())
        y0 = float((y * w).sum() / w.sum())
    else:
        imax = int(np.argmax(z))
        x0, y0 = float(x[imax]), float(y[imax])

    xrange = float(x.max() - x.min()) if x.size else 1.0
    yrange = float(y.max() - y.min()) if y.size else 1.0
    sx0 = max(xrange / 4.0, 1e-6)
    sy0 = max(yrange / 4.0, 1e-6)
    theta0 = 0.0

    p0 = [A0, x0, y0, sx0, sy0, theta0, B0]
    lb = [0.0, x.min() - xrange, y.min() - yrange, 1e-6, 1e-6, -np.pi, -np.inf]
    ub = [np.inf, x.max() + xrange, y.max() + yrange, np.inf, np.inf,  np.pi,  np.inf]

    # ---- fit
    popt, _ = curve_fit(
        _gaussian2d, (x, y), z, p0=p0, bounds=(lb, ub), maxfev=20000
    )
    A, x0, y0, sx, sy, theta, B = map(float, popt)

    # ---- append fitted values at the sampled points
    z_fit = _gaussian2d((x, y), *popt)
    for (k, gval) in zip(keys, z_fit):
        # Use your preferred key spelling; here we use 'gaussian_fit'
        res[k]["gaussian_fit"] = float(gval)
        res["x0_peak"] = float(x0)
        res["y0_peak"] = float(y0)

    params = {"A": A, "x0": x0, "y0": y0, "sx": sx, "sy": sy, "theta": theta, "B": B}
    return params, res

def process_scan( scan_data , method=None, **kwargs):
    """
    scan_data[(x,y)] = value i.e. its a dictionary 
    
    process data from scanner data which is just a dictionary with
    coordinate tuple as key and corresponding frame (2D array like) values.

    different methods of processing can be applied. If None is specified 
    then process_scan just returns scan_data  

    kwargs is any additional fields that are required for a specific method
    e.g. cluster analysis method might need details about how many clusters

    """

    plot_results = kwargs.get("plot",False)
    savePlot = kwargs.get("savePlot",False)
    save_results = kwargs.get("save_results",False)

    if method is None:
        return scan_data
    
    elif method.lower() == 'frame_aggregate':
        res = {}
        for k,v in scan_data.items():
            res[k] = {"mean":np.nanmean( v ), 
                      "std":np.nanstd( v ), 
                      "median":np.median(v)} 
        return res
    
    elif method.lower() == 'pupil_aggregate':
        res = {}
        for k,v in scan_data.items():

            # we only keep the generated pupil mask
            _, _, _, _, _, pupil_mask = util.detect_pupil(
                v, sigma=2, threshold=0.5, plot=False, savepath=None
            )  # pct.detect_circle


            # in 12x12 mode we have ~113 pixels in pupil, always assume ~2 are bad 
            # so we filter out above 98th percentile (naive safety to avoid bad pixels)
            qFilt = np.array(v) < np.quantile(np.array(v)[pupil_mask], 0.98 )

            pupil_mask &= qFilt # add it to the pupil_mask

            res[k] = {"mean":np.nanmean( np.array(v)[pupil_mask] ), 
                      "std":np.nanstd( np.array(v)[pupil_mask] ), 
                      "median":np.median( np.array(v)[pupil_mask] )} 

        return res 


    elif method.lower() == 'gaus_fit':
        res = {}
        for k,v in scan_data.items():

            # we only keep the generated pupil mask
            _, _, _, _, _, pupil_mask = util.detect_pupil(
                v, sigma=2, threshold=0.5, plot=False, savepath=None
            )  # pct.detect_circle


            # in 12x12 mode we have ~113 pixels in pupil, always assume ~2 are bad 
            # so we filter out above 98th percentile (naive safety to avoid bad pixels)
            qFilt = np.array(v) < np.quantile(np.array(v)[pupil_mask], 0.98 )

            pupil_mask &= qFilt # add it to the pupil_mask

            
            res[k] = {"mean":np.nanmean( np.array(v)[pupil_mask] ), 
                      "std":np.nanstd( np.array(v)[pupil_mask] ), 
                      "median":np.median( np.array(v)[pupil_mask] )} 

            params, res = fit_gaussian_on_res(res)
            print("Peak at (x0,y0) =", params["x0"], params["y0"])

        return res 
        

    elif method.lower() == 'frame_cluster':
        N_clusters = kwargs.get("N_clusters",3) # does 3 clusters if not specified
        print("to test")

        data_dict_ed = {
                    tuple(map(float, key.strip("()").split(","))): value
                    for key, value in scan_data.items()
                }

        x_points = np.array([float(x) for x, _ in data_dict_ed.keys()])
        y_points = np.array([float(y) for _, y in data_dict_ed.keys()])

        image_list = np.array(list(scan_data.values()))

        # the below function returns dict: A dictionary with keys:
        # - "centers" (list): List of tuples (x, y, radius) for each detected pupil.
        # - "clusters" (list): Cluster labels for each image.
        # - "centroids" (ndarray): Centroids of the clusters.
    
        raw_res = pct.cluster_analysis_on_searched_images(
            images=image_list,
            detect_circle_function=pct.detect_circle,
            n_clusters=int(N_clusters),
            plot_clusters=plot_results,
        )

        centersRaw = raw_res['centers']
        clustersRaw = raw_res['clusters']
        centroidsRaw =  raw_res['centroids']

        # reformatting results (untested!)
        res = {} 
        for ii, (cent, clus, roid) in enumerate( zip(centersRaw,clustersRaw,centroidsRaw) ):  
            res[(x_points[ii],y_points[ii])] = {"centers": cent},{"clusters":clus,"centroids":roid}

        if plot_results:
            fig, ax = pct.plot_cluster_heatmap(x_points, y_points, res["clusters"])

        return res
    

    elif method.lower() == 'pupil_PSD':
        print("to do")
        return None
    elif method.lower() == 'phasemask_detection_1':
        print("to do") 
        return None
    else:
        raise NotImplementedError(f"method {method} is not implemented")
    
