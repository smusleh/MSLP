
import re
from collections import Counter
import itertools
import pandas as pd
import numpy as np

'''
Python class to generate imporved Kmer features based on formulas in PLEK papper which uses 
improved Kmer scheme to representDNA seqences in Fasta file. The link to teh paper is
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4177586/pdf/12859_2013_Article_6586.pdf
'''
class Weighted_kmer:
    def __init__(self, fasta_file, k, pk, normalize=True):
        self.file = fasta_file
        self.k = k
        self.pk = pk
        self.normalize = normalize
        self.nucleotides='ACGT'
        
    def readfasta(self):
        with open(self.file) as f:
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
    
    def generate_list(self):
        ACGT_list=["".join(e) for e in itertools.product(self.nucleotides, repeat=self.k)]
        return ACGT_list
    
    def kmerArray(self,sequence):
        kmer = []
        for i in range(len(sequence) - self.k + 1):
            kmer.append(sequence[i:i + self.k])
        return kmer
    
    #def Generate_PLEK_Kmer_Features(self,file, k=2, PK=3, normalize=True):
    def Generate_weighted_kmer_Features(self):
        '''
            Generate imporved weighted Kmer features 
            input_data: input DNA Fasta file : Text file with seq id and seq of ACTG
            K : Kmer value (1,2,3,4,5,6,) : integer value
            normalize: boolean if TRUE normalize: COUNT / Seq-length
        '''
        fastas=self.readfasta()
        w = 1.0 / (4**(self.pk - self.k))
        print("w = ", w)
        vector = []
        header = ['id']
        if self.k < 1:
            print('k must be an integer and greater than 0.')
            return 0
        for kmer in itertools.product(self.nucleotides, repeat=self.k):
            header.append(''.join(kmer))
        vector.append(header)
        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            kmers = self.kmerArray(sequence)
            count = Counter()
            count.update(kmers)
            if self.normalize == True:
                for key in count:
                    count[key] = (count[key] / len(kmers)) * w
            code = [name]
            for j in range(1, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            vector.append(code)
        return vector
