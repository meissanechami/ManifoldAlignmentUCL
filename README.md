# The effectiveness for unsupervised structural alignment

The project has been written in Python 3 and Matlab. Two models have been cloned from other repositories: the code for is cloned from MAGAN https://github.com/KrishnaswamyLab/MAGAN and the matlab code for MAWC is cloned from https://github.com/all-umass/ManifoldAlignment. 

The datasets are generated using the data handler file and each experiment is shown on toy version in the aritificial_train.py, MNIST_train.py and GloVe_train.py with the procrustes model. Because the code for running the MAGAN is not straightforward a utils_magan.py file is provided which includes the correspondence loss and the training process. The MAWC model is ran from Matlab so datafiles with the variables to load in the environment are provided in order to run the train.m file. 
