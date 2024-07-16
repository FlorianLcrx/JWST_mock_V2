import numpy as np
import glob
from msafit.utils.array_utils import find_pixel
from astropy.io import fits
from msafit.fpa.detector import DetectorCutout
from astropy.nddata import bitmask


# save the sigma afor the the uncertainties save this array and the  spec with the added noise 

def get_hdr_info(si,sj,qd,filt,disp,array="1x3",sloc=0):

    obs_info = {"instrument":{},"geometry":{}}
    obs_info["instrument"]["filter"] = filt
    obs_info["instrument"]["disperser"] = disp
    obs_info["geometry"]["shutter_array"] = array
    obs_info["geometry"]["quadrant"] = qd
    obs_info["geometry"]["shutter_i"] = si
    obs_info["geometry"]["shutter_j"] = sj
    obs_info["geometry"]["source_shutter"] = sloc

    return obs_info


def get_obs(fname,ename,si,sj,qd,filt,disp,line_wave,object_id=0,array='1x3',sloc=0,pad_x=20,pad_y=5,norm_const=1):
    """Read in file and extract key parameters and data 
    
    Parameters
    ----------
    fname : list
        filename path for data array
    ename : list
        filename path for error array
    si : int
        shutter index
    sj : int
        shutter index
    qd : int
        quadrant
    filt : str
        filter
    disp : str
        disperser
    line_wave : float
        line wavelength in AA
    object_id : int, optional
        id number
    array : str, optional
        number of shutters opened
    sloc : int, optional
        location of source
    pad_x : int, optional
        number of pixels padded to detector cutout in x direction
    pad_y : int, optional
        number of pixels padded to detector cutout in y direction
    
    Returns
    -------
    dict
        dictionary with header info and spectral data
        also includes coordinates needed for making cutouts
    """
    print('fname',fname)
   
    obs_info = get_hdr_info(si,sj,qd,filt,disp,array,sloc)
    obs_info['OBS_ID'] = object_id
    obs_info['NEXP'] = 1
    
    #print("np.load(fnames)",np.load(fnames))
    obs_info['data'] = np.load(fname) * norm_const
    obs_info['unc'] = np.load(ename) * norm_const
    obs_info['mask'] = np.ones(obs_info['data'].shape,dtype=int)
# Which bit of the detector is sliced out, based on wavelength used to make the mock in the first place 
#pad x and pad y should be indentical to the spec thing so should be fine but check in case error 
     
    detector = DetectorCutout(obs_info["instrument"]["filter"],
               obs_info["instrument"]["disperser"],
               obs_info["geometry"]["quadrant"],
               obs_info["geometry"]["shutter_i"],
               obs_info["geometry"]["shutter_j"],
               obs_info["geometry"]["source_shutter"])

    x_l = detector.get_trace_x(line_wave)
    y_l = detector.get_trace_y(line_wave)

    if x_l<0: 
        ind_xl, ind_yl = find_pixel(x_l,y_l,detector.sca491[0],detector.sca491[1]) 
    elif x_l>0: 
        ind_xl, ind_yl = find_pixel(x_l,y_l,detector.sca492[0],detector.sca492[1]) 
    else:
        raise RunTimeError("Something is wrong with the data -\
                            unexpected mix between SCA491 and SCA492")

    xlow,xup,ylow,yup = detector._find_aperture(ind_xl,ind_yl,pad_x,pad_y)

    obs_info["pix_cood"] = [ylow,yup,xlow,xup]

    return obs_info

# this is the function imported in the other script, allows for multiple observations , so different positions etc.. 
# There has been change and it wasn't run 

def select_obs(obs_dir, fnames,enames, shutters_i, shutters_j, qd, filt, disp, line_wave,**kwargs):
    """Select data and process for fitting with msafit
    
    Parameters
    ----------
    obs_dir : str
        directory in which files are stored
    fnames : list
        list of filenames containing mock data
    enames : list
        list of filenames containing error array
    shutters_i : list
        list of shutter indices
    shutters_j : list
        list of shutter indices
    qd : int
        quadrant
    filt : str
        filter
    disp : str
        disperser
    line_wave : float
        observed wavelength (in AA) used to construct mock data
    **kwargs
        Description
    
    Returns
    -------
    list
        contains one dictionary per file
    """
    obs_list = []
    for idx in range(len(fnames)):
        obs_dict = get_obs(obs_dir+fnames[idx],obs_dir+enames[idx],shutters_i[idx],shutters_j[idx],qd,filt,disp,line_wave,**kwargs)
        obs_list.append(obs_dict)

    return sorted(obs_list, key=lambda d: d["OBS_ID"])