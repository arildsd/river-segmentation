# river-segmentation

A Convolutional Neural Network Unet model to segment B&W aerial images into ecological segments.

## Installation

1. Clone the project.
2. Install conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
3. Make a new conda enviorment in the terminal:
conda create --name myenvname python=3.7
4. Install required packages:
conda install --yes --file requirements.txt
5. Activate the enviorment before using the programs in this project:
conda activate myenvname

## Make prediction with the model
The project includes a trained model that can be used to make predictions on black-and-white areal images of rivers. Download the model (https://drive.google.com/file/d/1GDZ25vxDchf-C_848oPZHNtp7Up7g5NN/view?usp=sharing)
To make a prediction:
python source/run_predictions.py model_path input_folder output_folder













This project is done for my master thesis at NTNU in Trondheim. 
