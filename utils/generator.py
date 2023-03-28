# Revised from https://github.com/acmater/NK_Benchmarking/blob/master/utils/nk_utils/generate_datasets.py
# Distributed under BSD-3

# Thanks for https://github.com/acmater and https://github.com/yoavram/UnderTheRug

# This module is used to generate NK dataset

import pandas as pd
import numpy as np
import os

import numpy as np
import itertools

def collapse_single(protein):
    """
    Takes any iterable form of a single amino acid character sequence and returns a string representing that sequence.
    """
    return "".join([str(i) for i in protein])

def hamming(str1, str2):
    """Calculates the Hamming distance between 2 strings"""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def all_genotypes(N, AAs):
    """Fills the sequence space with all possible genotypes."""
    return np.array(list(itertools.product(AAs, repeat=N)))

def neighbors(sequence, sequence_space):
    """Gets neighbours of a sequence in sequence space based on Hamming
    distance."""
    hammings = []
    for i in sequence_space:
        hammings.append((i, hamming(sequence,i)))
    return [sequence  for  sequence, distance in hammings if distance==1]

def custom_neighbors(sequence, sequence_space, d):
    """Gets neighbours of a sequence in sequence space based on Hamming
    distance."""
    hammings = []
    for i in sequence_space:
        hammings.append((i, hamming(sequence,i)))
    return [sequence  for  sequence, distance in hammings if distance==d]

def genEpiNet(N, K):
    """Generates a random epistatic network for a sequence of length
    N with, on average, K connections"""
    return {
        i: sorted(np.random.choice(
            [n for n in range(N) if n != i],
            K,
            replace=False
        ).tolist() + [i])
        for i in range(N)
    }

def fitness_i(sequence, i, epi, mem):
    """Assigns a (random) fitness value to the ith amino acid that
    interacts with K other positions in a sequence, """
    #we use the epistasis network to work out what the relation is
    key = tuple(zip(epi[i], sequence[epi[i]]))
    #then, we assign a random number to this interaction
    if key not in mem:
        mem[key] = np.random.uniform(0, 1)
    return mem[key]


def fitness(sequence, epi, mem):
    """Obtains a fitness value for the entire sequence by summing
    over individual amino acids"""
    return np.mean([
        fitness_i(sequence, i, epi, mem) # ω_i
        for i in range(len(sequence))
    ])

def makeNK(N, K, AAs):
    """Make NK landscape with above parameters"""
    f_mem = {}
    epi_net = genEpiNet(N, K)
    sequenceSpace = all_genotypes(N,AAs)
    land = [(x,y) for x, y in zip(sequenceSpace, [fitness(i, epi=epi_net, mem=f_mem) for i in sequenceSpace])]
    return land, sequenceSpace, epi_net

def gen_distance_subsets(ruggedness,seq_len=5,library="ACDEFGHIKL",seed=None):
    """
    Takes a ruggedness, sequence length, and library and produces an NK landscape then separates it
    into distances from a seed sequence.

    ruggedness [int | 0-(seq_len-1)]  : Determines the ruggedness of the landscape
    seq_len : length of all of the sequences
    library : list of possible characters in strings
    seed    : the seed sequence for which distances will be calculated

    returns ->  {distance : [(sequence,fitness)]}
    """

    land_K2, seq, _ = makeNK(seq_len,ruggedness,library)

    if not seed:
        seed = np.array([x for x in "".join([library[0] for x in range(seq_len)])])

    subsets = {x : [] for x in range(seq_len+1)}
    for seq in land_K2:
        subsets[hamming(seq[0],seed)].append(seq)

    return subsets

def dataset_generation(directory="./Data",seq_len=5, library="ACDEFGHIKL", all_ruggedness = None, all_instance = 5):
    """
    Generates five instances of each possible ruggedness value for the NK landscape

    seq_len
    """

    if all_ruggedness == None:
        all_ruggedness = range(0, seq_len)
    if not os.path.exists(directory):
        os.mkdir(directory)

    datasets = {x : [] for x in range(seq_len)}

    for ruggedness in all_ruggedness:
        for instance in range(all_instance):
            print(f"Generating data for L={seq_len} R={ruggedness} A={library} I={instance}")

            subsets = gen_distance_subsets(ruggedness,seq_len, library)

            hold = []

            for i in subsets.values():
                for j in i:
                    hold.append([collapse_single(j[0]),j[1]])

            saved = np.array(hold)
            df = pd.DataFrame({"Sequence" : saved[:,0], "Fitness" : saved[:,1]})
            df.to_csv("{0}/R{1}L{2}E{3}V{4}.csv".format(directory,ruggedness,seq_len,len(library), instance))

    print ("All data generated. Data is stored in: {}".format(directory))