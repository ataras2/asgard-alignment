
import numpy as np 
from common import phasemask_centering_tool as pct
from pyBaldr import utilities as util
from typing import Dict, Tuple, Any
from scipy.optimize import curve_fit

import ast
from scipy.optimize import curve_fit

def _gaussian2d(coords, A, x0, y0, sx, sy, theta, B):
    x, y = coords
    ct, st = np.cos(theta), np.sin(theta)
    xr = (x - x0) * ct + (y - y0) * st
    yr = -(x - x0) * st + (y - y0) * ct
    return B + A * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

def _to_xy(k) -> Tuple[float, float]:
    if isinstance(k, (tuple, list)) and len(k) == 2:
        return float(k[0]), float(k[1])
    if isinstance(k, str):
        try:
            t = ast.literal_eval(k)
            if isinstance(t, (tuple, list)) and len(t) == 2:
                return float(t[0]), float(t[1])
        except Exception:
            pass
        s = k.strip().lstrip("(").rstrip(")")
        x_str, y_str = s.split(",", 1)
        return float(x_str), float(y_str)
    raise ValueError(f"Unparsable key: {k!r}")

def fit_gaussian_on_res(
    res: Dict[Any, Dict[str, Any]]
):
    """
    Fits a 2D Gaussian to res[*]['mean'] (keys may be '(x,y)' strings or (x,y) tuples).
    Adds res[key]['gaussian_fit'] at each sampled point and also
    stores the peak coordinates at:
        res['x0_peak'] = x0
        res['y0_peak'] = y0
    Returns (params, res) with params = {A,x0,y0,sx,sy,theta,B}.
    """
    if not res:
        raise ValueError("res is empty.")

    # Use only entries that look like sample points (have a 'mean' field)
    items = [(k, v) for k, v in res.items() if isinstance(v, dict) and 'mean' in v]
    if not items:
        raise ValueError("No sample entries with a 'mean' field found in res.")

    keys = [k for k, _ in items]
    x = np.array([_to_xy(k)[0] for k in keys], dtype=float)
    y = np.array([_to_xy(k)[1] for k in keys], dtype=float)
    z = np.array([float(res[k]['mean']) for k in keys], dtype=float)

    mfin = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mfin.sum() < 6:
        raise ValueError("Not enough finite points to fit a 2D Gaussian (need >= 6).")
    x, y, z = x[mfin], y[mfin], z[mfin]
    keys_fin = [kk for kk, keep in zip(keys, mfin) if keep]

    # Initial guesses
    B0 = float(np.median(z))
    A0 = float(max(np.max(z) - B0, 1e-9))
    w = np.clip(z - B0, 0.0, np.inf)
    if w.sum() > 0:
        x0 = float((x * w).sum() / w.sum())
        y0 = float((y * w).sum() / w.sum())
    else:
        i = int(np.argmax(z)); x0, y0 = float(x[i]), float(y[i])
    xrange = float(max(x) - min(x)); yrange = float(max(y) - min(y))
    sx0 = max(xrange / 4.0, 1e-6); sy0 = max(yrange / 4.0, 1e-6); theta0 = 0.0

    p0 = [A0, x0, y0, sx0, sy0, theta0, B0]
    lb = [0.0, x.min() - xrange, y.min() - yrange, 1e-6, 1e-6, -np.pi, -np.inf]
    ub = [np.inf, x.max() + xrange, y.max() + yrange, np.inf, np.inf,  np.pi,  np.inf]

    popt, _ = curve_fit(_gaussian2d, (x, y), z, p0=p0, bounds=(lb, ub), maxfev=20000)
    A, x0, y0, sx, sy, theta, B = map(float, popt)

    # Fitted values at sampled points
    z_fit = _gaussian2d((x, y), *popt)
    for k, g in zip(keys_fin, z_fit):
        res[k]['gaussian_fit'] = float(g)

    # Store peak coords at top level as requested
    # DONT INCLUDE THESE BECAUSE IT CHANGES STANDARDS THAT CAN LEAD TO ERRORS LATER 
    #res['x0_peak'] = x0
    #res['y0_peak'] = y0

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
    
