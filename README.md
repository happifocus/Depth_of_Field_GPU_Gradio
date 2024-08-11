# AI-Driven Depth of Field Simulations with Gradio and GPU

This project provides a tool to generate depth maps and apply a shallow depth of field effect to images using deep learning models. 
The application is built using PyTorch, Gradio, and Hugging Face Transformers.


## Features

- Generate depth maps from images using a pre-trained model.
- Apply a depth of field effect to images based on the generated depth map.
- Interactive web interface built with Gradio.


## Installation

Install Anaconda, create environment, install JupyterLab and open the notebook **DOF_GPU_Gradio.ipynb**:

1. **Create a project folder**

    Place downloaded notebook **DOF_GPU_Gradio.ipynb** into project folder.

2. **Anaconda Navigator**

   - ***Environments***: Create a new environment, providing a name and selecting the Python package. 

   - ***Home***: Install JupyterLab and launch it. 

3. **JupyterLab**

   Navigate to the project folder. Open the notebook: **DOF_GPU_Gradio.ipynb** and continue:

   Verify that you are working in the newly created environment by running:

   ````Python
   !conda env list


To set up the environment and install the necessary dependencies, follow these steps:

1. **Terminal of Anaconda environment: install PyTorch and CUDA support:**

   ```bash
   conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia


2. **DOF_GPU_Gradio.ipynb: check GPU availability:**

   ```Python
   import torch
   torch.cuda.is_available()

4. **Terminal of Anaconda environment: install Gradio and other dependencies:**

   ```bash
   conda install conda-forge::gradio
   conda install transformers
   pip install Pillow
   pip install opencv-python

5. **Import libraries:**

   Continue importing libraries, as shown in **DOF_GPU_Gradio.ipynb**

7. **Run the code:**

    Run the main code, as shown in **DOF_GPU_Gradio.ipynb**

    
## Usage

    1. Upload an original image (PNG/JPEG).
    2. Adjust the blur strength using the slider.
    3. Press 'Submit' to process the image.
    4. View and download the processed image and depth map.

Code Overview

    - Model Loading: Uses the DPTForDepthEstimation model from Hugging Face's Transformers library.
    - Depth Map Creation: Processes images to generate depth maps using the pre-trained model.
    - Depth of Field Effect: Applies a Gaussian blur to parts of the image based on the depth map.
    - Gradio Interface: Provides an easy-to-use web interface for image processing.

Logging

    The application uses Python's logging module to provide information about the processing steps and any errors encountered.
    
Cleanup

    Temporary files created during processing are automatically cleaned up upon exiting the application.

    
## License

    This project is licensed under the MIT License.


## Acknowledgments

    (Hugging Face Transformer)[https://huggingface.co/docs/transformers/index]
    (Gradio)[https://www.gradio.app/]
    (PyTorch)[https://pytorch.org/]
