## Optical Flow and Covolutional Neural Network

Implementation of Optical Flow algorithm and the VGG-16 CNN

## How to run the code

Recommended Python : 3.7.10

Recommended IDE : SPYDER/JUPYTER NOTEBOOK

Libraries Used : Numpy, Opencv, Tensorflow, Datetime

Problem_1.py : Code for Optical Flow solutions of problem 1

	1> To run the Problem_1.py, please make sure that the required video is in the same directory as the code.

	2> If you want to write the videos in the directory for problem 1.1 or 1.2.1 or 1.2.2 then please unhash the OUT, OUT_1 and OUT_2 lines in the code.


Problem_2.py : Code for Image Classification 

	1> The code was executed on a machine with 16GB RAM, AMD Ryzen 7 processor, NVIDIA GeForce GTX 1660 Ti 6GB graphics card. Please use a machine with similar or higher configurations to the Problem_2.py

	2> Please paste the "directory" of the dataset carefully on line 24 of the code

	3> SPYDER can be used to run the code as well as JUPYTER NOTEBOOK (carefully placing the lines inside the IN[*] blocks into separate, consecutive cells)

	4> The input images were resized for efficient computation.

	5> Create an environment and run the following commands to run the code

'''
conda create -n new-env-name
'''
(where new-env-name is the name of the new environment)
'''
conda activate new-env-name
'''
'''
pip install tensorflow-gpu
'''
(please install pip if not already)
'''
conda install -c conda-forge cudnn
'''
you can use spyder or jupyter notebook for running the code or just the terminal

to install spyder or jupyter notebook
'''
conda install spyder
conda install jupyter-lab
'''
