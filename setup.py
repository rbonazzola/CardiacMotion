from setuptools import setup, find_packages

setup(
  name="cardiac-motion-rl",
  version="0.0.1",
  author="Rodrigo Bonazzola (rbonazzola)",
  author_email="rodbonazzola@gmail.com",
  description="Python package for unsupervised phenotyping of dynamic cardiac meshes",
  long_description=open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/rbonazzola/CardiacMotion",
  packages=find_packages(),
  install_requires=[
    "numpy>=1.24.0",
    "meshio>=5.3.5",
    "vtk>=9.3.1",
    "vedo>2024.5.2",
    "scipy>=1.10.1",
    "trimesh>=4.5.3",
    "torch>=2.4.1",
    "torch-geometric>=2.6.1",
    "torch_cluster>=1.6.3",
    "torch_scatter>=2.1.2",
    "torch_sparse>=0.6.18",
    "pytorch-lightning>=2.4.0",
    "mlflow>=2.14.0",
    "tqdm",
    "icosphere",
    "easydict",
    "cardio-mesh"
  ],
  python_requires=">=3.8,<3.11"
)
