import re, sys
import pandas as pd
import itertools
from collections import Counter

NUCLEOTIDES = 'ACTG'

def read_dna_Fasta(file):
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error,the input DNA sequence must be fasta format.')
        sys.exit(1)
    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACGT-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])
    return myFasta

def generate_list(k, nucleotides):
    ACGT_list=["".join(e) for e in itertools.product(nucleotides, repeat=k)]
    return ACGT_list


def parallel_cor_function(nucleotide1, nucleotide2, phyche_index):
    temp_sum = 0.0
    phyche_index_values = list(phyche_index.values())
    len_phyche_index = len(phyche_index_values[0])
    for u in range(len_phyche_index):
        temp_sum += pow(float(phyche_index[nucleotide1][u]) - float(phyche_index[nucleotide2][u]), 2)
    parallel_value=temp_sum / len_phyche_index   
    return parallel_value


def get_parallel_factor_psednc(lambda_, sequence, phyche_value):

    theta = []
    l = len(sequence)

    for i in range(1, lambda_ + 1):
        temp_sum = 0.0
        for j in range(0, l - 1 - lambda_):
            nucleotide1 = sequence[j] + sequence[j + 1]
            nucleotide2 = sequence[j + i] + sequence[j + i + 1]
            temp_sum += parallel_cor_function(nucleotide1, nucleotide2, phyche_value)
        theta.append(temp_sum / (l - i - 1))
    return theta


def generate_phychem_property(raw_property, new_property):
    if new_property is None or len(new_property) == 0:
        return raw_property
    for key in list(raw_property.keys()):
        raw_property[key].update(new_property[key])


def get_phychem_property_psednc():
    ''' The normalized values for the following physiochemical properties of dinucleotides in DNA
        these properties are Twist, Tilt, Roll, Shift, Slide and Rise
        Twist = 0.06, Tilt = 0.5, Roll = 0.27, Shift = 1.59, Slide = 0.11, and Rise = -0.11 and so on.
    '''
    raw_property = {'AA': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11],
                   'AC': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
                   'AG': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
                   'AT': [1.07, 0.22, 0.62, -1.02, 2.51, 1.17],
                   'CA': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
                   'CC': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
                   'CG': [-1.66, -1.22, -0.44, -0.82, -0.29, -1.39],
                   'CT': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
                   'GA': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
                   'GC': [-0.08, 0.22, 1.33, -0.35, 0.65, 1.59],
                   'GG': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
                   'GT': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
                   'TA': [-1.23, -2.37, -0.44, -2.24, -1.51, -1.39],
                   'TC': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
                   'TG': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
                   'TT': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11]}
    extra_phyche_index={}
    property_value = generate_phychem_property(raw_property, extra_phyche_index)
    return property_value


def frequency(t1_str, t2_str):
    i, j, tar_count = 0, 0, 0
    len_tol_str = len(t1_str)
    len_tar_str = len(t2_str)
    while i < len_tol_str and j < len_tar_str:
        if t1_str[i] == t2_str[j]:
            i += 1
            j += 1
            if j >= len_tar_str:
                tar_count += 1
                i = i - j + 1
                j = 0
        else:
            i = i - j + 1
            j = 0
    return tar_count

def gen_pseknc_vector(sequence_list, lambda_, w, k, phyche_value):

    kmer = generate_list(k, NUCLEOTIDES)
    header=['id']
    for f in range((4**k+lambda_)):
        header.append('pseknc.'+str(f))
    vector=[]
    vector.append(header)
    for sequence_ in sequence_list:
        name,sequence=sequence_[0],sequence_[1]
        if len(sequence) < k or lambda_ + k > len(sequence):
            error_info = "error, the sequence length must be larger than " + str(lambda_ + k)
            sys.stderr.write(error_info)
            sys.exit(0)
        fre_list = [frequency(sequence, str(key)) for key in kmer]
        fre_sum = float(sum(fre_list))
        fre_list = [e / fre_sum for e in fre_list]
        theta_list = get_parallel_factor_psednc(lambda_, sequence, phyche_value)
        theta_sum = sum(theta_list)
        denominator = 1 + w * theta_sum
        temp_vec = [round(f / denominator, 3) for f in fre_list]
        for theta in theta_list:
            temp_vec.append(round(w * theta / denominator, 4))
        sample=[name]
        sample=sample+temp_vec
        vector.append(sample)
    return vector


def Pseknc(input_data, k=3, lambda_=10, w=0.05):
    
    phyche_value = get_phychem_property_psednc()
    fastas=read_dna_Fasta(input_data)  
    vector=gen_pseknc_vector(fastas, lambda_, w, k, phyche_value)    
    return vector
