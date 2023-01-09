import pickle
import os
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from util import FileProcessing, CheckAccPseParameter
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, status, Response, Query, File, HTTPException, UploadFile, Path, Body
from pydantic import BaseModel, Field
from typing import Optional, List
import shutil
import pyfastx
import re
import json
import time




app = FastAPI()


# load the models from disk
Cyto = "./models/Cytoplasm.sav"
Endo = "./models/Endoplasmic_Reticulum.sav"
Extr = "./models/Extracellular.sav"
Mito = "./models/Mitochondria.sav"
Nucl = "./models/Nucleus.sav"

Cyto_model = pickle.load(open(Cyto, 'rb'))
Endo_model = pickle.load(open(Endo, 'rb'))
Extr_model = pickle.load(open(Extr, 'rb'))
Mito_model = pickle.load(open(Mito, 'rb'))
Nucl_model = pickle.load(open(Nucl, 'rb'))


def predict_localization():
	fname = "master.csv"
	df = pd.read_csv("./input/" + fname)
	print(df.shape)
	print(df.head())
	df = df.drop(["SampleName", "label"], axis=1)
	print(df.shape)
	scaler = StandardScaler()
	df = scaler.fit_transform(df)
	
	xnew = df
 
	# make the predictions
	model_names = ["Cyto_model", "Endo_model", "Extr_model", "Mito_model", "Nucl_model"]
	models      = [Cyto_model, Endo_model, Extr_model, Mito_model, Nucl_model]

	results_df = pd.DataFrame()
	 
	for i in range(len(xnew)):
	    max_prob = 0
	    max_label = ""
	    for model_name, model in zip(model_names,models):
	        ynew  = model.predict_proba(xnew)
	        if ynew[i][1] >= max_prob:
	            max_prob = ynew[i][1]
	            max_label = model_name.split("_")[0]
	    results_df_2 = pd.DataFrame(data=[[i, max_prob , max_label]], 
	                            columns=['Sample_index', 'Prob', 'Predictions'])
	    results_df = pd.concat([results_df, results_df_2], ignore_index=True)

	results_df.to_csv("./output/"+"predictions"+".csv", encoding='utf-8', index=False)

	return results_df.to_dict()




def merge_feature_files(FeaFiles):


	dup_list = ['SampleName', 'label', 'lamada_1', 'lamada_2', 'lamada_3', 'lamada_4', 'lamada_5', 'lamada_6', 'lamada_7', 'lamada_8', 'lamada_9', 'lamada_10']
	df_list = []
	df_master = pd.DataFrame()

	for fn in FeaFiles:
		print("processing :", fn)
		df = pd.read_csv("./input/" + fn)

		if (fn in ["Kmer3.csv", "Kmer4.csv","Kmer5.csv"]):
			df = df.drop(['SampleName', 'label'], axis=1)
		
		elif (fn in ["PseKNC2.csv"]):
			df = df.drop(['SampleName', 'label'], axis=1)
			df.columns = 'Pse_' + df.columns

		elif (fn in ["PseKNC3.csv","PseKNC4.csv","PseKNC5.csv"]):
			df = df.drop(dup_list,axis=1)
			df.columns = 'Pse_' + df.columns

		elif fn == "PseEIIP.csv":
			df = df.drop(['SampleName', 'label'], axis=1)
			df.columns = 'EIIP_' + df.columns

		elif fn == "DPCP.csv":
			df = df.drop(['SampleName', 'label'], axis=1)

		elif fn == "TPCP.csv":
			df = df.drop(['SampleName', 'label'], axis=1)

		elif fn == "Z_curve_48bit.csv":
			df = df.drop(['SampleName', 'label'], axis=1)
			df.columns = df.columns.str.replace(".", "")

		elif fn == "Z_curve_144bit.csv":
			df = df.drop(['SampleName', 'label'], axis=1)
			df.columns = df.columns.str.replace(".", "")
		
		df_list.append(df)
		
	df_master = pd.concat(df_list, axis=1)
	print(df_master.shape)
	print(df_master.head())

	df_master.to_csv("./input/"+"master"+".csv", encoding='utf-8', index=False)


def is_dna_compliant(fasta_file):
    with open(fasta_file, 'r') as f:
        contents = f.read()
    sequences = contents.split('>')[1:]  # Split the contents into individual sequences and remove the first empty string
    for sequence in sequences:
        lines = sequence.split('\n')
        header = lines[0]
        seq = ''.join(lines[1:])  # Join the sequence data lines into a single string
        if not re.match(r'^[ACGT]+$', seq):  # Use a regular expression to check if the sequence consists only of A, C, T, and G
            return False
    return True

def validate_dna_dict(dictionary):
    valid_keys = ['A', 'C', 'T', 'G']
    for key in dictionary:
        if key not in valid_keys:
            return False
    return True


class UserInput(BaseModel):
	fname: str

	class Config:
		example = {
			"fname": "file.fasta"
		}

def reset_defaults():
	desc_default_para = {  # default parameter for descriptors
		'sliding_window': 5,
		'kspace': 3,
		'props': ['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
		'nlag': 3,
		'weight': 0.05,
		'lambdaValue': 10,
		'PseKRAAC_model': 'g-gap',
		'g-gap': 2,
		'k-tuple': 2,
		'RAAC_clust': 1,
		'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101',
		'kmer': 2,
		'mismatch': 1,
		'delta': 0,
		'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',
		'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)',
		'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)',
		'distance': 0,
		'cp': 'cp(20)',
	}
	return desc_default_para

@app.post('/predict')
async def Gen_Sequence_Predictions(file: UploadFile = File(...) ):
	desc_default_para = {  # default parameter for descriptors
		'sliding_window': 5,
		'kspace': 3,
		'props': ['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
		'nlag': 3,
		'weight': 0.05,
		'lambdaValue': 10,
		'PseKRAAC_model': 'g-gap',
		'g-gap': 2,
		'k-tuple': 2,
		'RAAC_clust': 1,
		'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101',
		'kmer': 2,
		'mismatch': 1,
		'delta': 0,
		'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',
		'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)',
		'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)',
		'distance': 0,
		'cp': 'cp(20)',
	}


	def calculate_descriptor(desc_fasta_file, desc_selected_descriptor, desc_seq_type):
		try:
			descriptor = None

			if desc_fasta_file != '' and desc_selected_descriptor != '':
				descriptor = FileProcessing.Descriptor(desc_fasta_file, desc_default_para)
				if descriptor.error_msg == '' and descriptor.sequence_number > 0:
					print('Calculating ...')
					result = False
					if descriptor.sequence_type == 'Protein':
						cmd = 'descriptor.' + descriptor.sequence_type + '_' + desc_selected_descriptor + '()'
						result = eval(cmd)
					else:
						if desc_selected_descriptor in ['DAC', 'TAC']:
							my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(
								desc_selected_descriptor, desc_seq_type, desc_default_para)
							result = descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer)
						elif desc_selected_descriptor in ['DCC', 'TCC']:
							my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(
								desc_selected_descriptor, desc_seq_type, desc_default_para)
							result = descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer)
						elif desc_selected_descriptor in ['DACC', 'TACC']:
							my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(
								desc_selected_descriptor, desc_seq_type, desc_default_para)
							result = descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer)
						elif desc_selected_descriptor in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC',
															'SCPseTNC']:
							my_property_name, my_property_value, ok = CheckAccPseParameter.check_Pse_arguments(
								desc_selected_descriptor, desc_seq_type, desc_default_para)
							cmd = 'descriptor.' + desc_selected_descriptor + '(my_property_name, my_property_value)'
							result = eval(cmd)
						else:
							cmd = 'descriptor.' + desc_selected_descriptor + '()'
							print("ran this", cmd)
							result = eval(cmd)
					print('Calculation complete.')
					return descriptor
			else:
				print('Warning: Please check your input!')

		except Exception as e:
			print("Error: ", e)

	# try:

	feature_types = {
		"Kmer": {
			'kmer': [2,3,4,5]
		},
		"PseKNC": {
			'kmer': [2,3,4,5]
		},
		"PseEIIP": {},
		"DPCP": {
			'Di-DNA-Phychem': 'Base stacking;Protein induced deformability;B-DNA twist;Dinucleotide GC Content;A-philicity;Propeller twist;Duplex stability:(freeenergy);Duplex tability(disruptenergy);DNA denaturation;Bending stiffness;Protein DNA twist;Stabilising energy of Z-DNA;Aida_BA_transition;Breslauer_dG;Breslauer_dH;Breslauer_dS;Electron_interaction;Hartman_trans_free_energy;Helix-Coil_transition;Ivanov_BA_transition;Lisser_BZ_transition;Polar_interaction;SantaLucia_dG;SantaLucia_dH;SantaLucia_dS;Sarai_flexibility;Stability;Stacking_energy;Sugimoto_dG;Sugimoto_dH;Sugimoto_dS;Watson-Crick_interaction;Twist;Tilt;Roll;Shift;Slide;Rise;Clash Strength;Roll_roll;Twist stiffness;Tilt stiffness;Shift_rise;Adenine content;Direction;Twist_shift;Enthalpy1;Twist_twist;Roll_shift;Shift_slide;Shift2;Tilt3;Tilt1;Tilt4;Tilt2;Slide (DNA-protein complex)1;Tilt_shift;Twist_tilt;Twist (DNA-protein complex)1;Tilt_rise;Roll_rise;Stacking energy;Stacking energy1;Stacking energy2;Stacking energy3;Propeller Twist;Roll11;Rise (DNA-protein complex);Tilt_tilt;Roll4;Roll2;Roll3;Roll1;Minor Groove Size;GC content;Slide_slide;Enthalpy;Shift_shift;Slide stiffness;Melting Temperature1;Flexibility_slide;Minor Groove Distance;Rise (DNA-protein complex)1;Tilt (DNA-protein complex);Guanine content;Roll (DNA-protein complex)1;Entropy;Cytosine content;Major Groove Size;Twist_rise;Major Groove Distance;Twist (DNA-protein complex);Purine (AG) content;Melting Temperature;Free energy;Tilt_slide;Major Groove Width;Major Groove Depth;Wedge;Free energy8;Free energy6;Free energy7;Free energy4;Free energy5;Free energy2;Free energy3;Free energy1;Twist_roll;Shift (DNA-protein complex);Rise_rise;Flexibility_shift;Shift (DNA-protein complex)1;Thymine content;Slide_rise;Tilt_roll;Tip;Keto (GT) content;Roll stiffness;Minor Groove Width;Inclination;Entropy1;Roll_slide;Slide (DNA-protein complex);Twist1;Twist3;Twist2;Twist5;Twist4;Twist7;Twist6;Tilt (DNA-protein complex)1;Twist_slide;Minor Groove Depth;Roll (DNA-protein complex);Rise2;Persistance Length;Rise3;Shift stiffness;Probability contacting nucleosome core;Mobility to bend towards major groove;Slide3;Slide2;Slide1;Shift1;Bend;Rise1;Rise stiffness;Mobility to bend towards minor groove'
		},
		"TPCP": {},
		"Z_curve_48bit": {},
		"Z_curve_144bit": {}
	}
	
	print("file", file, file.filename)

	contents = await file.read()

	# return {"filename": file.filename, "contents": contents}
	timenow = time.time()
	processing_seq_file_name = f"./seq_file-{timenow}.fasta" 
	with open(processing_seq_file_name, "w") as my_file:
		my_file.write(contents.decode())

	fa = pyfastx.Fasta(processing_seq_file_name)
	print(fa.type)
	print(fa.__len__())
	print(fa.composition)

	print(validate_dna_dict(fa.composition))
	if validate_dna_dict(fa.composition) is False:
		os.remove(processing_seq_file_name)
		os.remove(f"{processing_seq_file_name}.fxi")
		return Response(media_type="application/json", status_code=400, content=json.dumps({"err": "Invalid Fasta file", **fa.composition}))
	if fa.__len__() > 1000:
		os.remove(processing_seq_file_name)
		os.remove(f"{processing_seq_file_name}.fxi")
		return Response(media_type="application/json", status_code=400, content=json.dumps({"err": "Fasta file size exceeds limit - Max size 1000 Sequences" }))

	print({"fasta file type":fa.type, "Number of Sequences is ":fa.__len__(), "Fasta File DNA compisition is":fa.composition})  

	for featype in feature_types.keys():
		if featype == "Kmer" or featype == "PseKNC":
			kmer_values = feature_types[featype]["kmer"]
			for kmer in kmer_values:

				desc_default_para['kmer'] = kmer

				res = calculate_descriptor(processing_seq_file_name, featype, 'DNA')
				df = pd.DataFrame(res.encoding_array)
				print(df.head()) 
				df.to_csv("./input/"+ featype + str(kmer) + ".csv", header=None, encoding='utf-8', index=False)
				desc_default_para = reset_defaults()
		else:					
			if featype == "DPCP":
				desc_default_para['Di-DNA-Phychem'] =  feature_types[featype]["Di-DNA-Phychem"]
			res = calculate_descriptor(processing_seq_file_name, featype, 'DNA')
			df = pd.DataFrame(res.encoding_array)
			print(df.head())
			df.to_csv("./input/"+ featype + ".csv", header=None, encoding='utf-8', index=False)
			desc_default_para = reset_defaults()
	
	
	# Merge all feature files into one master file
	fea_files = ["Kmer2.csv", "Kmer3.csv", "Kmer4.csv", "Kmer5.csv", "PseKNC2.csv", "PseKNC3.csv", "PseKNC4.csv", "PseKNC5.csv", "PseEIIP.csv", "DPCP.csv", "TPCP.csv", "Z_curve_48bit.csv", "Z_curve_144bit.csv"]
	merge_feature_files(fea_files)
	
	os.remove(processing_seq_file_name)
	os.remove(f"{processing_seq_file_name}.fxi")
	
	return predict_localization()



@app.get('/download_feature_file', response_class=FileResponse, include_in_schema=False)
async def Feature_File():
	
	return "./input/master.csv"



@app.get('/download_prediction_file', response_class=FileResponse, include_in_schema=False)
async def Feature_File():
	
	return "./output/predictions.csv"




