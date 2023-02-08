# mRNA Subcellular Localization Predictor
A machine learning-based tool to predict the subcellular localization of mRNA



## Data folder
Contains 4 floders:
* IDS_I: Independent Dataset I : 
         First test datasets used for testing only. These include Nine sublocalizations.
* IDS_II: ndependent Dataset II: 
         Second test data set used for testing only. These include Nine sublocalizations.
* TEST_01: Testset_01: 
         Independent dataset files used for testing only.
* Train: Training Dataset: 
         Contains all five Fasta Sequence Files used to train MSLP models. The fifth one is a combination of all Fastafiles.

## Code Folder:
* Shap analysis Jupyter Notebook
* Feature generation 
* Ablation study code


The folder contains python code to generate MLP Application Programming Interface (API) using.
FastAPI (a Web framework for developing RESTful APIs in Python). It is based on Pydantic and 
type hints to validate, serialize, and deserialize data, and automatically auto-generate OpenAPI documents.


 ## Docker Folder:
 * fasta folder contains sample fasta file
 * input folder contains uploaded fasta files
 * models folder contains MSLP-trained binary classifiers based on OvR strategy.
                One model for each of the five localizations
 * output folder: temp folder
 * util folder contains code and classes to generate the four types of features (Kmer, PseKNC, PSEEIIP, 
              TPCP, DPCP, Z_curve_48bit, and Z_curve_144bit.
 docker-compose.yaml, Dockerfie: Specs to build the MSLP container
 requirements.txt: All required packages to build the API within the container 
 localization_api.py: python file contains code and logic to build the API endpoints
 
## Docker Guideline:
**OPTION 1: Use the PREBUILT Docker image available on the public Docker Hub Account**

FROM USER SIDE (User Computer):
1. Install Docker Desktop (Windows, Linux, or MacBook iOS)
	You can download and install Docker Desktop from: https://www.docker.com/products/docker-desktop/ 

2. RUN the following command: Open a terminal session (Windows - command line or cmd, on MacBook it is terminal, on Ubuntu it is console app)
	- Type in the following command:
		docker run -p 7000:8000 smusleh/localization:mslp
	- Open the local browser and type in the address bar the following link:
	  http://127.0.0.1:7000/docs or http://localhost:7000/docs  

**OPTION 2: Build and run the Docker Container locally and run.**

FROM USER SIDE (User Computer):
1. Clone this directory to local folder in your system
2. Switch to the App folder, then run the following command:
	Docker compose up --build

Once the container is up and running (OPTION 1 or OPTION 2):

1. Open the local browser and type in the address bar the following link:
	http://127.0.0.1:7000/docs or http://localhost:7000/docs 

2. Expand POST /predict endpoint, then click on "Try it out"
   Click on the "Choose File" button and upload the fasta file, then CLICK on the "Execute" button
   Then all Fasta sequences get processed (all features generated in a "master" file, then
   each sequence in the master file passes through all classifiers, a prediction probability
   get generated, and then the sequence gets assigned to classifiers that have the highest 
   probability. The sequences are then of that classier class.

**Download Prediction Results: To download the predictions in CSV format.**   
   Type in the following link:
   http://localhost:7000/download_prediction_file
   
   To download the features file generated "master.csv" file, open a new tap on the browser and
   type in the following link:
   http://localhost:7000/download_feature_file
   


![image](https://user-images.githubusercontent.com/19537901/211297065-1a82aff3-34b1-4b75-ad8f-78f0a28e685b.png)



