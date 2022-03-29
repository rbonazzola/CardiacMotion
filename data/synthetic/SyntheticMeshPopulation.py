import os
import vedo
import numpy as np
import trimesh
from copy import copy
from scipy.special import sph_harm
import pyvista as pv
from IPython import embed
import re
import random
from argparse import Namespace
import pickle as pkl
import icosphere

class SyntheticMeshPopulation(object):

    def __init__(self, 
        N, T, l_max, freq_max, 
        amplitude_static_max, amplitude_dynamic_max, 
        mesh_resolution, random_seed=None, 
        verbose=False, cache=True, from_cache_if_exists=True, odir=None, ofile=None):

        '''
        params: 
          N (int): number of subjects
          T (int): number of time points across the cycle
          l_max (int): maximum l coefficient of the expansion in spherical harmonics
          freq_max (int): maximum multiple of the fundamental frequency
          amplitude_static_max (float): variance of the "content" coefficients 
          amplitude_dynamic_max (float): variance of the "style" coefficients
          mesh_resolution (int): resolution of the spherical mesh as taken by vedo.Sphere's constructor.
          random_seed (int): seed to produce the random coefficients of the population.
          verbose (boolean): if True, prints out the subject's indices (every 100) while generating them.
          cache (boolean): whether to cache this object to save time in the future.
          odir (string or None): directory where to store this object, if cache == True.
          ofile (string or None): 
    
        attributes:
          params:
          avg_shape_list: list of static meshes for the N individuals, i.e. N 3D meshes.
          time_seq_list: list of T moving meshes for the N individuals, i.e. T x N 3D meshes.
          coefs: coefficients for each of {1|sin(nt)|cos(nt)} * Ylm(theta, phi).
          ref_shape: Trimesh object representing the sphere.
        '''

        self.params = {
            "N": N,
            "T": T,
            "l_max": l_max,
            "freq_max": freq_max,
            "amplitude_static_max": amplitude_static_max,
            "amplitude_dynamic_max": amplitude_dynamic_max,
            "mesh_resolution": mesh_resolution,
            "random_seed": random_seed
        }
        self.params = Namespace(**self.params)

        self._ofile = ofile
        self._odir = odir

        # build output file name
        self._ofile = self._get_filename()

        if os.path.exists(self._ofile) and from_cache_if_exists:
            print("Retrieving synthetic population from cached file.")
            cached = pkl.load(open(self._ofile, "rb"))
            self.template = cached.template
            self.time_avg_meshes = cached.time_avg_meshes
            self.moving_meshes = cached.moving_meshes
            self.coefficients = cached.coefficients
            return

        self._verbose = verbose
        self._generate_population()

        if cache:
           self._save_population_as_pkl()

        
    # From here:
    # https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    def _appendSpherical_np(self, xyz):
          
          '''
          params:
              xyz: NumPy array representing the (x,y,z) coordinates for a point cloud.
              
          return:
              A NumPy array with 6 columns containing the input (x,y,z) coordinates
              and, additionally, the (r, theta, phi) spherical coordinates        
          '''
          
          ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
          xy = xyz[:,0]**2 + xyz[:,1]**2        
          ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
          ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down      
          ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
          return ptsnew
           
    
    def _sample_coefficients(self, l_max, freq_max=None, amplitude_max=0.1, random_seed=None):
            
        if random_seed is not None:
            random.seed(random_seed)
            
        if freq_max is not None:
            amplitude_lmn = {"sin":{}, "cos":{}}
            for n in range(1, freq_max):
              for l in range(l_max+1):
                 for m in range(-l, l+1):
                   amplitude_lmn["cos"][(l,m,n)] = random.gauss(0, amplitude_max)     
                   amplitude_lmn["sin"][(l,m,n)] = random.gauss(0, amplitude_max)     
            return amplitude_lmn
    
        else:
            amplitude_lm = {}      
            for l in range(l_max+1):
               for m in range(-l, l+1):
                 amplitude_lm[(l,m)] = random.gauss(0, amplitude_max)     
            return amplitude_lm



    def _generate_population(self):
            
        # sphere = vedo.Sphere(res=self.params.mesh_resolution).to_trimesh()
        sphere = icosphere.icosphere(nu=self.params.mesh_resolution)
        sphere = trimesh.Trimesh(vertices=sphere[0], faces=sphere[1])
        sphere_coords = self._appendSpherical_np(sphere.vertices)[:,-2:] # Get theta and phi coordinates

        Y_lm, f_lmn, g_lmn = cache_base_functions(sphere_coords, self.params.l_max, self.params.freq_max, Nt=self.params.T)
    
        avg_shape_list = []
        time_seq_list = []
        coefs = []
    
        if self.params.random_seed is not None:
            random.seed(self.params.random_seed)
    
        # Loop through individuals
        for i in range(self.params.N):
    
            if self._verbose and (i % 100) == 0: print(i)
    
            # Generate latent variables for a single individual        
            amplitude_lm0 = self._sample_coefficients(
                l_max=self.params.l_max, 
                freq_max=None, 
                amplitude_max=self.params.amplitude_static_max
            )

            amplitude_lmn = self._sample_coefficients(
                l_max=self.params.l_max, 
                freq_max=self.params.freq_max, 
                amplitude_max=self.params.amplitude_dynamic_max
            )
            coefs.append([amplitude_lm0, amplitude_lmn])
            #####
                        
            # Perform static (content) deformations
            static_deformations = np.zeros(sphere_coords.shape[0])        

            for l, m in Y_lm:
                static_deformations += amplitude_lm0[l, m] * Y_lm[l, m]
                
            avg_deformed_sphere = sphere.copy()

            avg_deformed_sphere.vertices = sphere.vertices + np.multiply(
                sphere.vertices,
                np.kron(np.ones((3, 1)), static_deformations).transpose()
            )
    
            # average deformation of the shape across the cycle (i.e. the "content" component)
            avg_shape_list.append(avg_deformed_sphere)
    
            # Loop through time to perform dynamic (style) deformations        
            mesh_i = []            
            for t in range(self.params.T):
                deformed_sphere = avg_deformed_sphere.copy()                            
                
                dynamic_deformations = np.zeros(sphere_coords.shape[0])                        
                for l, m, n in amplitude_lmn["sin"]:
                    dynamic_deformations += amplitude_lmn["sin"][l, m, n] * f_lmn[l, m][n][:, t]
                    dynamic_deformations += amplitude_lmn["cos"][l, m, n] * g_lmn[l, m][n][:, t]
                        
                deformed_sphere.vertices += np.multiply(
                    sphere.vertices, np.kron(np.ones((3, 1)), dynamic_deformations).transpose()
                )
    
                mesh_i.append(deformed_sphere)
            
            time_seq_list.append(mesh_i)
    
        # convert avg_shape_list (population of time-averaged meshes) to 3D NumPy array
        avg_meshes_np = np.array([ np.array(pp.vertices) for pp in avg_shape_list ]).astype('float32')
        
        # convert time_seq_list (population of moving meshes) to NumPy 4D array
        moving_meshes_np = np.array([ np.array([ np.array(mesh_t.vertices) for mesh_t in subj ]) for subj in time_seq_list ]).astype('float32')
        
        self.template = sphere        
        self.time_avg_meshes = avg_meshes_np
        self.moving_meshes = moving_meshes_np
        self.coefficients = coefs        
    


    def _get_filename(self):

        '''
        Produces an output PKL filename for the object. This filename is used for:
        - retrieving the object from a file if the file already exists.
        - save the object to a file if it doesn't exist already and cache == True
        '''

        FILE_PATTERN = "synthetic_population__N_{N}__T_{T}_" + \
        "_sigma_c_{amplitude_static_max}__sigma_s_{amplitude_dynamic_max}_" + \
        "_lmax_{l_max}__nmax_{freq_max}__res_{mesh_resolution}__seed_{random_seed}.pkl"             
 
        if self._ofile is None:
            self._ofile = FILE_PATTERN.format(**self.params.__dict__)

        if self._odir is None:
            self._odir = "."        

        self._object_hash = hash(tuple(self.params.__dict__.values())) % 1000000
        self._odir = os.path.join(self._odir, f"synthetic_{self._object_hash}")
        ofile = os.path.join(self._odir, self._ofile)

        return ofile


    def _save_population_as_pkl(self):

        '''
        Saves this object as a pickle file.
        '''
                
        os.makedirs(os.path.dirname(self._ofile), exist_ok=True)

        with open(self._ofile, "wb") as ff:
            pkl.dump(self, ff)


    def render_mesh_as_png(self, mesh3D, filename,
                           camera_position='xy', show_edges=False,
                           **kwargs)

        '''  
        Produces a png file representing a static 3D mesh.
        - params
        ::mesh4D:: a sequence of Trimesh mesh objects. 
        ::mesh_connectivity:: faces
        ::filename:: the name of the output png file. 
        ::camera_position::
        
        - return:  
        None, only produces the png file. 
        '''

        plotter = pv.Plotter(off_screen=True)
        connectivity = mesh3D.faces
        connectivity = np.c_[np.ones(connectivity.shape[0]) * 3, connectivity].astype(int)

        mesh = pv.PolyData(mesh3D.vertices, connectivity)
        plotter.camera_position = camera_position
        actor = plotter.add_mesh(mesh, show_edges=show_edges)
        plotter.screenshot(filename if filename.endswith("png") else filename + ".png")


    def _generate_gif(self, mesh4D, mesh_connectivity, filename, camera_position='xy', show_edges=False, **kwargs):
        
        '''
        Produces a gif file representing the motion of the input mesh.
        
        params:
          mesh4D: a sequence of Trimesh mesh objects.
          mesh_connectivity: faces in a PyVista-friendly format
          filename: the name of the output gif file.
          camera_position:
          
        return:
          None, only produces the gif file.
        '''
        
        pv.set_plot_theme("document")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # plotter = pv.Plotter(shape=(1, len(camera_positions)), notebook=False, off_screen=True)
        plotter = pv.Plotter(notebook=False, off_screen=True)
            
        # Open a gif
        plotter.open_gif(filename) 
    
        kk = pv.PolyData(np.array(mesh4D[0]), mesh_connectivity)
        # plotter.add_mesh(kk, smooth_shading=True, opacity=0.5 )#, show_edges=True)
        plotter.add_mesh(kk, show_edges=show_edges) 
        
        for t, _ in enumerate(mesh4D):
            kk = pv.PolyData(np.array(mesh4D[t]), mesh_connectivity)
            plotter.camera_position = camera_position
            plotter.update_coordinates(kk.points, render=False)
            plotter.render()             
            plotter.write_frame()
        
        plotter.close()
        
        
    def generate_gif_population(self, N_gifs=10, camera_positions=['xy', 'yz', 'xz'], verbose=False, **kwargs):
        
        '''
          population: population of moving meshes.
          N_gifs: number of gifs to be generated.
        
        '''
        conn = self.template.faces
        conn = np.c_[3 * np.ones(conn.shape[0]), conn].astype(int)  # add column of 3 to make it PyVista-compatible
        
        for i, moving_mesh in enumerate(self.moving_meshes):
            if verbose: print(i)
            for camera_position in camera_positions:        
                filename = os.path.join(self._odir , "gifs/subject{}_{}.gif".format(i, camera_position))
                self._generate_gif(moving_mesh, conn, filename, camera_position, **kwargs)        
            if i == N_gifs:
                break

######################## END OF CLASS ########################



### CACHE BASE FUNCTIONS ###
    
def cache_sin_and_cos(freq_max, Nt):
    
    '''
    Generate spherical harmonics for a set of locations across the sphere.
    params:
       freq_max: maximum multiple of the fundamental frequency.
       Nt: number of equispaced time points on the [0,2*pi] interval
    returns:
       two dictionaries in a tuple, where the keys of each are the n indices 
       of the sine and cosine functions
    '''
    
    sin_n = { w: np.array([np.sin(2*w*np.pi*i/Nt) for i in range(Nt)]) for w in range(1, freq_max+1) }    
    cos_n = { w: np.array([np.cos(2*w*np.pi*i/Nt) for i in range(Nt)]) for w in range(1, freq_max+1) }

    return sin_n, cos_n


def cache_Ylm(sphere_coords, l_max):
    
    '''
    Generate (real) spherical harmonics for a set of locations across (theta, phi) the sphere.
    params:
       sphere_coords: numpy.array of shape (M, 2) where M is the number of points, 
                      containing the (theta, phi) coordinates of the points of the discretization.
       indices: list of (l,m) indices
    returns:
       a dictionary where the keys are the (l,m) indices 
       and the values are real spherical harmonics evaluated 
       at the (theta, phi) angles in sphere_coords
    '''
    
    Y_lm = {}
    
    indices = [ (l,m) for l in range(l_max+1) for m in range(-l,l+1) ]
    for l, m in indices:
        
        # Real spherical harmonics (defined in terms of the complex-valued ones)
        if m < 0:
            Y_lm[(l,m)] = np.array([
              np.sqrt(2) * sph_harm(-m, l, *sphere_coords[k]).imag
              for k in range(sphere_coords.shape[0])      
            ])
        elif m > 0:
            Y_lm[(l,m)] = np.array([
              np.sqrt(2) * sph_harm(m, l, *sphere_coords[k]).real
              for k in range(sphere_coords.shape[0])
            ])
        else:
            Y_lm[(l,m)] = np.array([
              sph_harm(m, l, *sphere_coords[k]).real
              for k in range(sphere_coords.shape[0])
            ])
            
    return Y_lm


def cache_base_functions(sphere_coords, l_max, freq_max, Nt):
        
    sin_n, cos_n = cache_sin_and_cos(freq_max, Nt)
    Y_lm = cache_Ylm(sphere_coords, l_max)
    
    f_lmn = {}
    g_lmn = {}
    
    for l, m in Y_lm:
        f_lmn[l,m] = {}
        g_lmn[l,m] = {}
        for n in sin_n:
             f_lmn[l,m][n] = np.dot( np.expand_dims(Y_lm[l,m],1), np.expand_dims(sin_n[n], 0) )       
             g_lmn[l,m][n] = np.dot( np.expand_dims(Y_lm[l,m],1), np.expand_dims(cos_n[n], 0) )
    
    return Y_lm, f_lmn, g_lmn
    
#############################################
