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

# From here:
# https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def appendSpherical_np(xyz):
      
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
    
    sin_n = { 
        w: np.array([np.sin(2*w*np.pi*i/Nt) for i in range(Nt)]) 
        for w in range(1,freq_max) 
    }
    
    cos_n = { 
        w: np.array([np.cos(2*w*np.pi*i/Nt) for i in range(Nt)])
        for w in range(1,freq_max) 
    }

    return sin_n, cos_n


def cache_Ylm(sphere_coords, l_max):
    
    '''
    Generate spherical harmonics for a set of locations across the sphere.
    params:
       sphere_coords: numpy.array of shape (M, 2) where M is the number of points, 
                      containing the theta and phi coordinates of the points of the discretization.
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
              np.sqrt(2) * sph_harm(-m, l, *sphere_coords[k, 1:3]).imag
              for k in range(sphere_coords.shape[0])      
            ])
        elif m > 0:
            Y_lm[(l,m)] = np.array([
              np.sqrt(2) * sph_harm(m, l, *sphere_coords[k, 1:3]).real
              for k in range(sphere_coords.shape[0])
            ])
        else:
            Y_lm[(l,m)] = np.array([
              sph_harm(m, l, *sphere_coords[k, 1:3]).real
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

##########################################

def sample_coefficients(l_max, freq_max=None, amplitude_max=0.1, random_seed=None):
        
    if random_seed is not None:
       random.seed(random_seed)
        
    if freq_max is not None:
      amplitude_lmn = {"sin":{}, "cos":{}}
      for n in range(1,freq_max):
        for l in range(l_max+1):
           for m in range(-l,l+1):
             amplitude_lmn["cos"][(l,m,n)] = random.gauss(0, amplitude_max)     
             amplitude_lmn["sin"][(l,m,n)] = random.gauss(0, amplitude_max)     
      return amplitude_lmn

    else:
      amplitude_lm = {}      
      for l in range(l_max+1):
         for m in range(-l, l+1):
           amplitude_lm[(l,m)] = random.gauss(0, amplitude_max)     
      return amplitude_lm    


def generate_population(N, T, l_max, freq_max, amplitude_static_max, amplitude_dynamic_max, mesh_resolution, random_seed, verbose=False):
    
    '''
    params: 
      N:
      T: 
      l_max:
      freq_max:
      amplitude_static_max:
      amplitude_dynamic_max:
      random_seed:
      mesh_resolution:

    return value:
      avg_shape_list: 
      time_seq_list: list of moving meshes for the N individuals.
      coefs: coefficients for each of {1|sin(nt)|cos(nt)} * Ylm(theta, phi).
    '''

    sphere = vedo.Sphere(res=mesh_resolution).to_trimesh()
    sphere_coords = sphere.vertices    
    Y_lm, f_lmn, g_lmn = cache_base_functions(sphere_coords, l_max, freq_max, Nt=T)

    avg_shape_list = []
    time_seq_list = []
    coefs = []

    # Loop through individuals
    for i in range(N):

        if verbose and (i % 100) == 0: print(i)

        # Generate latent variables for a single individual
        mesh_i = []
        amplitude_lm0 = sample_coefficients(l_max, amplitude_max=amplitude_static_max)
        amplitude_lmn = sample_coefficients(l_max, freq_max, amplitude_max=amplitude_dynamic_max)
        coefs.append([amplitude_lm0, amplitude_lmn])
        #####

        deformations = np.ones(sphere_coords.shape[0])
        
        # Perform static (content) deformations
        static_deformations = np.zeros(sphere_coords.shape[0])
        for l, m in Y_lm:
            static_deformations += amplitude_lm0[l, m] * Y_lm[l, m]
            
        avg_deformed_sphere = sphere.copy()
        avg_deformed_sphere.vertices = sphere.vertices + np.multiply(
            sphere.vertices,
            np.kron(np.ones((3, 1)), static_deformations).transpose()
        )

        # average deformation of the shape across the cycle (i.e. the content component)
        avg_shape_list.append(avg_deformed_sphere)

        # Loop through time to perform dynamic (style) deformations
        for t in range(T):
            
            deformed_sphere = sphere.copy()
            dynamic_deformations = np.zeros(sphere_coords.shape[0])            
            
            for l, m, n in amplitude_lmn["sin"]:
                dynamic_deformations += amplitude_lmn["sin"][l, m, n] * f_lmn[l, m][n][:, t]
                dynamic_deformations += amplitude_lmn["cos"][l, m, n] * g_lmn[l, m][n][:, t]
                    
            deformed_sphere.vertices = avg_deformed_sphere.vertices + np.multiply(
                sphere.vertices, np.kron(np.ones((3, 1)), dynamic_deformations).transpose()
            )

            mesh_i.append(deformed_sphere)
        
        time_seq_list.append(mesh_i)

    return avg_shape_list, time_seq_list, coefs