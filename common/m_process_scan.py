
import numpy as np 
from common import phasemask_centering_tool as pct
from pyBaldr import utilities as util


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

            res[k] = {"mean":np.nanmean( np.array(v)[pupil_mask] ), 
                      "std":np.nanstd( np.array(v)[pupil_mask] ), 
                      "median":np.median( np.array(v)[pupil_mask] )} 

        return res 
        
    elif method.lower() == 'frame_cluster':
        N_clusters = kwargs.get("N_clusters",3) # does 3 clusters if not specified
        print("to do")

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
    
        res = pct.cluster_analysis_on_searched_images(
            images=image_list,
            detect_circle_function=pct.detect_circle,
            n_clusters=int(N_clusters),
            plot_clusters=plot_results,
        )

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
    
