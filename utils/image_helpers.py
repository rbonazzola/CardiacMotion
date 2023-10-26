import numpy as np
from PIL import Image
import imageio
import os
import pyvista as pv

from tqdm import tqdm

# subj_idx_w = widgets.IntSlider(min=1, max=len(cardiac_dataset))

def generate_gif(mesh4D, faces, filename, camera_position='xy', show_edges=False, **kwargs):
        
        '''
        Produces a gif file representing the motion of the input mesh.
        
        params:
          ::mesh4D:: a sequence of Trimesh mesh objects.
          ::faces:: array of F x 3 containing the indices of the mesh's triangular faces.
          ::filename:: the name of the output gif file.
          ::camera_position:: camera position for pyvista plotter (check relevant docs)
          
        return:
          None, only produces the gif file.
        '''

        import pyvista as pv
        
        connectivity = np.c_[np.ones(faces.shape[0]) * 3, faces].astype(int)
                
        pv.set_plot_theme("document")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # plotter = pv.Plotter(shape=(1, len(camera_positions)), notebook=False, off_screen=True)
        pv.start_xvfb()
        plotter = pv.Plotter(notebook=False, off_screen=True)
            
        # Open a gif
        plotter.open_gif(filename)

        try:
            # if mesh3D is torch.Tensor, this your should run OK
            mesh4D = mesh4D.cpu().numpy()[0].astype("float32")
        except AttributeError:
            pass

        pv_mesh = pv.PolyData(mesh4D[0], connectivity)
        # plotter.add_mesh(kk, smooth_shading=True, opacity=0.5 )#, show_edges=True)
        plotter.add_mesh(pv_mesh, show_edges=show_edges, opacity=0.5, color="red") 
        
        for t, _ in tqdm(enumerate(mesh4D)):
            # print(t)
            pv_mesh = pv.PolyData(mesh4D[t], connectivity)
            plotter.camera_position = camera_position
            plotter.update_coordinates(pv_mesh.points, render=False)
            plotter.render()             
            plotter.write_frame()
        
        plotter.close()
        
        return filename

    
def merge_pngs_horizontally(png1, png2, output_png):
    # https://www.tutorialspoint.com/python_pillow/Python_pillow_merging_images.htm
    # Read the two images
    image1 = Image.open(png1)
    image2 = Image.open(png2)
    # resize, first image
    image1_size = image1.size
    # image2_size = image2.size
    new_image = Image.new('RGB', (2 * image1_size[0], image1_size[1]), (250, 250, 250))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))
    new_image.save(output_png, "PNG")


def merge_gifs_horizontally(gif_file1, gif_file2, output_file):
    # Create reader object for the gif
    gif1 = imageio.get_reader(gif_file1)
    gif2 = imageio.get_reader(gif_file2)

    # Create writer object
    new_gif = imageio.get_writer(output_file)

    for frame_number in range(gif1.get_length()):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        # here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()
