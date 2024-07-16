import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import matplotlib  as mpl
import h5py
import matplotlib.colors as colors
from tqdm import tqdm
from scipy import stats
import cmasher as cmr
import seaborn as sns
import warnings
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from matplotlib.colors import LogNorm


def Defining_new_coords (Coord_stars,mp,R_star) : 

    # Initialing tensor Mij
    Mijup = np.zeros((3, 3))
    Mijdown = np.zeros((3, 3))

    radius = np.sqrt(np.sum(Coord_stars**2, axis=1))
    radius_cond = radius <= R_star*6
    if mp[radius_cond] is None or len(mp) < 1000:
        radius_cond = radius <= R_star*8
        if mp[radius_cond] is None or len(mp) < 1000:
            radius_cond = radius <= R_star*10
            if mp[radius_cond] is None or len(mp) < 1000:
                radius_cond = radius <= R_star*12
                if mp[radius_cond] is None or len(mp) < 1000:
                    radius_cond = radius <= R_star*14
                    if mp[radius_cond] is None or len(mp) < 1000:
                        radius_cond = radius <= R_star*16
                        if mp[radius_cond] is None or len(mp) < 1000:
                            radius_cond = radius <= R_star*18
                            if mp[radius_cond] is None or len(mp) < 1000:
                                radius_cond = radius <= R_star*20
                                if mp[radius_cond] is None or len(mp) < 1000:
                                    radius_cond = radius <= R_star*40

    for i in range(3):
        rpi = Coord_stars[:, i]
        for j in range(3):
            rpj = Coord_stars[:, j]

            Mijup[i, j] = np.sum(mp[radius_cond] * rpi[radius_cond] * rpj[radius_cond])
            Mijdown[i, j] = np.sum(mp[radius_cond])

    Mij = Mijup / Mijdown
        
    eigenvalues, eigenvectors = np.linalg.eig(Mij)

    Axis_length = np.sqrt(eigenvalues)

    a = np.max(Axis_length)
    c = np.min(Axis_length)
    b_cond = np.where((Axis_length > c) & (Axis_length < a))
    b = (Axis_length[b_cond])[0]

    # Initialize ratios
    old_ca_ratio = c / a
    old_ba_ratio = b / a

    # Set convergence threshold
    convergence_threshold = 0.01

    old_a = a
    old_b = b
    old_c = c

    anti_infinit_break = 0

    while True:
        rp_newframe = np.array([transform_particle_coords(particle_coords, eigenvectors) for particle_coords in Coord_stars])
        sorted_indices = np.argsort(eigenvalues)[::-1]
        x, y, z = rp_newframe[:, sorted_indices[0]], rp_newframe[:, sorted_indices[1]], rp_newframe[:, sorted_indices[2]]
        q = b/a
        s = c/a
    
        rp_wave = (x)**2 + (y**2/q**2) + (z**2/s**2)

        r_max = (((old_a**2)/(old_b*old_c))**(2/3)) * ((30)**2)

        cond_rad_max = rp_wave <= r_max

        Mijrup = np.zeros((3, 3))
        Mijrdown = np.zeros((3, 3))

        for i in range(3):
            rpi = Coord_stars[:, i]
            for j in range(3):
                rpj = Coord_stars[:, j]
                Mijrup[i, j] = np.sum((mp[cond_rad_max]/rp_wave[cond_rad_max]) * rpi[cond_rad_max] * rpj[cond_rad_max])
                Mijrdown[i, j] = np.sum(mp[cond_rad_max]/rp_wave[cond_rad_max])

        Mijr = Mijrup / Mijrdown

        if np.any(np.isnan(Mijr)) or np.any(np.isinf(Mijr)):
            print('for subhalo = ',subhalo_id)
            raise ValueError("Mijr contains NaN or Inf values before eigen computation")
            
        
        eigenvalues, eigenvectors = np.linalg.eig(Mijr)
    
        Axis_length = np.sqrt(eigenvalues)
    
        int_a = np.max(Axis_length)
        int_c = np.min(Axis_length)
        int_b_cond = np.where((Axis_length > int_c) & (Axis_length < int_a))
        int_b = (Axis_length[int(int_b_cond[0])])
    
        eigenvalues_scaled = eigenvalues

        Axis_length_scaled = np.sqrt(eigenvalues_scaled)* (((a*b*c)**(1/3))/((int_a*int_b*int_c)**(1/3)))

        new_a = np.max(Axis_length_scaled)
        new_c = np.min(Axis_length_scaled)
        b_cond = np.where((Axis_length_scaled > new_c) & (Axis_length_scaled < new_a))
        new_b = (Axis_length_scaled[int(b_cond[0])])    

        new_ca_ratio = new_c / new_a
        new_ba_ratio = new_b / new_a

        fractional_change_ca = abs(new_ca_ratio - old_ca_ratio) / old_ca_ratio
        fractional_change_ba = abs(new_ba_ratio - old_ba_ratio) / old_ba_ratio

        epsilon = 1 - (new_c/new_a)
        Ttriaxial = ((new_a**2) - (new_b**2)) / ((new_a**2)-(new_c**2))
        
        if fractional_change_ca < convergence_threshold and fractional_change_ba < convergence_threshold:
            break
        
        if anti_infinit_break > 20:
                break

        old_ca_ratio = new_ca_ratio
        old_ba_ratio = new_ba_ratio

        old_a = a
        old_b = new_b
        old_c = new_c

        anti_infinit_break += 1

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    if any(v is None for v in [new_a, new_b, new_c, epsilon, Ttriaxial, eigenvectors]):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    return (new_a, new_b, new_c, epsilon, Ttriaxial, eigenvectors)    


def Kinematics_of_the_galaxie(coord,particule_velocities, subhalo_velocity, eigenvectors,R_star,scaling_factor):
    
    coord_in_new_ref_frame = transform_particle_coords(coord,eigenvectors)

    speed_along_c = transform_particle_coords(particule_velocities,eigenvectors[:,2])
    speed_along_b = transform_particle_coords(particule_velocities,eigenvectors[:,1])
    speed_along_c_subhalo =  transform_particle_coords(subhalo_velocity,eigenvectors[:,2])
    speed_along_b_subhalo =  transform_particle_coords(subhalo_velocity,eigenvectors[:,1])
    
    Part_velocity_along_c = speed_along_c - speed_along_c_subhalo
    Part_velocity_along_b = speed_along_b - speed_along_b_subhalo

    #defining the slit
    xmax = 2*R_star
    xmin = -xmax
    ymax = (2/5)*R_star
    ymin = -ymax
    xmin_exclude = -1 * R_star
    xmax_exclude = 1 * R_star
          
    Slit_condition = (coord_in_new_ref_frame[:,0]>=2*xmin) & (coord_in_new_ref_frame[:,0]<=2*xmax) & (coord_in_new_ref_frame[:,1]>=ymin) & (coord_in_new_ref_frame[:,1]<=ymax) & (coord_in_new_ref_frame[:,2]>=ymin) & (coord_in_new_ref_frame[:,2]<=ymax)
    Slit_star_cord = coord_in_new_ref_frame[:,0][Slit_condition]
    Slit_star_vel =  Part_velocity_along_b[Slit_condition]

    #Computing velocity profile along the slit 
    bin_width = 0.5 * scaling_factor
    bins = np.arange(2*xmin, (2*xmax) + bin_width, bin_width)

    mean_velocities = []
    mean_coords = []

    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_end = bins[i+1]
        particules_in_bin = (Slit_star_cord >= bin_start) & (Slit_star_cord < bin_end)
        velocities_in_bin = Slit_star_vel[particules_in_bin]
        coord_in_bin = Slit_star_cord[particules_in_bin]
        mean_velocity = np.mean(velocities_in_bin)
        mean_coord = np.mean(coord_in_bin)
        mean_velocities.append(mean_velocity)
        mean_coords.append(mean_coord)

    Cond_radVmax = (mean_coords > xmin) & (mean_coords < xmax)
    if len(np.array(mean_velocities)) != 0 :
        V_max = np.nanmax(np.abs(mean_velocities))
    else : 
        V_max = np.nan
    if len(np.array(mean_velocities)[Cond_radVmax]) != 0 :
        V_max_comp = np.nanmax(np.abs(np.array(mean_velocities)[Cond_radVmax]))
    else : 
        V_max_comp = np.nan
        
    #Velocity dispersion
    distance_from_origin = np.sqrt(coord_in_new_ref_frame[:,0]**2 + coord_in_new_ref_frame[:,2]**2)
    
    std_lof = []
    mean_coords_std = []

    total_radius_kpc = 2 * R_star
    ring_width_kpc = 0.5 * scaling_factor

    # Iterate over each ring
    for inner_radius_kpc in np.arange(0, total_radius_kpc, ring_width_kpc):
        outer_radius_kpc = inner_radius_kpc + ring_width_kpc
    
        within_cylinder_mask = (distance_from_origin < outer_radius_kpc) & (distance_from_origin >= inner_radius_kpc) & (coord_in_new_ref_frame[:,2]>=ymin) & (coord_in_new_ref_frame[:,2]<=ymax)
        velocities_in_bin = Part_velocity_along_c[within_cylinder_mask]
        coord_in_bin = coord_in_new_ref_frame[within_cylinder_mask]
        rad_of_part_bin = np.sqrt(coord_in_bin[:,0]**2 + coord_in_bin[:,1]**2)

        mean_disp = np.std(velocities_in_bin)
        mean_coord = np.mean(rad_of_part_bin)

        std_lof.append(mean_disp)
        mean_coords_std.append(mean_coord)
    
    mean_coords_cond_for_R_2R = mean_coords_std >= R_star

    std_lof = np.array(std_lof)
    V_disp = np.nanmean(std_lof[mean_coords_cond_for_R_2R])

    return (Part_velocity_along_b,Part_velocity_along_c,mean_velocities,mean_coords,V_max,V_max_comp,V_disp,std_lof,mean_coords_std) 
    

def transform_particle_coords(particle_coords, trans_matrix_scaled):
    return np.dot(particle_coords, trans_matrix_scaled)


def velocity_dispersion(velocities):
    mean_velocity = np.mean(velocities)
    squared_deviations = (velocities - mean_velocity) ** 2
    dispersion = np.sqrt(np.mean(squared_deviations))
    return dispersion


def process_subfind_id(indices, basepath, snapnum, hubble_param, Subhalo_pos, Subhalo_Velocity, scaling_factor, R_star):

    r_star = R_star[indices]
    Subhalo_gas_part = il.snapshot.loadSubhalo(basepath, snapNum=snapnum, id=indices, partType='gas', fields=["StarFormationRate","Masses","Coordinates","Velocities"])
    
    Star_forming_cond = Subhalo_gas_part["StarFormationRate"] > 0
    Mass_gas = Subhalo_gas_part["Masses"][Star_forming_cond] * 1e10 / hubble_param
    Coord_gas = (Subhalo_gas_part["Coordinates"][Star_forming_cond]-Subhalo_pos[indices]) * scaling_factor / hubble_param
    velocity_gas = Subhalo_gas_part["Velocities"][Star_forming_cond]*np.sqrt(scaling_factor)
    
    a, b, c, epsilon, Ttriaxial, eigenvectors = Defining_new_coords(Coord_gas, Mass_gas, r_star)
    if not np.isnan(a):
        Part_velocity_along_b ,Part_velocity_along_c,mean_velocities,mean_coords,V_max,V_max_comp,V_disp,std_lof,mean_coords_std =  Kinematics_of_the_galaxie(Coord_gas,velocity_gas, Subhalo_Velocity[indices], eigenvectors, r_star, scaling_factor)
    else : 
        Part_velocity_along_b ,Part_velocity_along_c,mean_velocities,mean_coords,V_max,V_max_comp,V_disp,std_lof,mean_coords_std =  Kinematics_of_the_galaxie(Coord_gas,velocity_gas, Subhalo_Velocity[indices], np.eye(3), r_star, scaling_factor)

    # Create a dictionary with the variables as keys and their values
    result_dict = {
        "a": a,
        "b": b,
        "c": c,
        "epsilon": epsilon,
        "Ttriaxial": Ttriaxial,
        "eigenvectors": eigenvectors,
        "Part_velocity_along_b":Part_velocity_along_b,
        "Part_velocity_along_c":Part_velocity_along_c,
        "mean_rotational_velocities":mean_velocities,
        "mean_rotational_coords":mean_coords,
        "max_rot_vel":V_max,
        "max_rot_vel_comp":V_max_comp,
        "V_disp":V_disp,
        "std_lof":std_lof,
        "mean_coords_std":mean_coords_std
    }

    return result_dict


Snap_nums = [99,50,33,25] #z = 0,1,2,3,4,5,6

for i in range(len(Snap_nums)):
    snapnum = Snap_nums[i]
    file_path = f'/virgotng/universe/IllustrisTNG/TNG50-1/output/snapdir_{snapnum:03d}/snap_{snapnum:03d}.0.hdf5'
    
    with h5py.File(file_path, 'r') as f:
        header = dict( f['Header'].attrs.items())
        
    Redshift = header['Redshift']
    hubble_param = header['HubbleParam']
    scaling_factor = 1.0 / (1+Redshift)

    basePath = "/virgotng/universe/IllustrisTNG/TNG50-1/output"
    fields = ["SubhaloMassType",'SubhaloPos',"SubhaloFlag","SubhaloHalfmassRadType","SubhaloVel"]
    subgroups = il.groupcat.loadSubhalos(basePath,snapnum,fields=fields)

    gas_mass =  (subgroups["SubhaloMassType"][:,0] * 1e10 / hubble_param)

    Half_mass_rad = subgroups["SubhaloHalfmassRadType"][:,4] * scaling_factor / hubble_param
    positions = subgroups["SubhaloPos"]
    subhalovel = subgroups["SubhaloVel"]

    SubhaloID = np.arange(len(subgroups["SubhaloFlag"]))

    Flag_1 = subgroups["SubhaloFlag"] == 1
    Flag_2 = gas_mass > 10**8
    Flag_3 = Half_mass_rad > 0
    
    GalID = SubhaloID[(Flag_1) & (Flag_2) & (Flag_3)]

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    pool = mp.Pool(mp.cpu_count())
    
    partial_process_subfind_id = partial(process_subfind_id, basepath=basePath, snapnum=snapnum, hubble_param=hubble_param, Subhalo_pos=positions, Subhalo_Velocity = subhalovel, scaling_factor=scaling_factor, R_star=Half_mass_rad)
    results = list(tqdm(pool.imap(partial_process_subfind_id, GalID), total=len(GalID)))

    pool.close()
    pool.join()

    with h5py.File(f'New_result_gas_with_kinematics_slit25{snapnum}', 'w') as hf:  # Use 'a' for append mode
        # Create a group for each result
        for i, result in enumerate(results):
            group = hf.create_group(f'Subhalo_{GalID[i]}')
            # Save each key-value pair in the dictionary as a dataset
            for key, value in result.items():
                group.create_dataset(key, data=value)