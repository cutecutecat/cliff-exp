# Revised from https://github.com/acmater/NK_Benchmarking/blob/master/utils/nk_utils/generate_datasets.py
# Distributed under BSD-3

# Thanks for https://github.com/acmater and https://github.com/yoavram/UnderTheRug

# This module is used to calculate extrema Ruggedness and RS Ruggedness

import time
import numpy as np
import pandas as pd
import copy
import tqdm as tqdm
import multiprocessing as mp
from functools import partial, reduce

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def collapse_concat(arrays, dim=0):
    """
    Takes an iterable of arrays and recursively concatenates them. Functions similarly
    to the reduce operation from python's functools library.
    Parameters
    ----------
    arrays : iterable(np.array)
        Arrays contains an iterable of np.arrays
    dim : int, default=0
        The dimension on which to concatenate the arrays.
    returns : np.array
        Returns a single np array representing the concatenation of all arrays
        provided.
    """
    if len(arrays) == 1:
        return arrays[0]
    else:
        return np.concatenate((arrays[0], collapse_concat(arrays[1:])))


class Protein_Landscape:
    """
    Class that handles a protein dataset

    Parameters
    ----------

    data : np.array

        Numpy Array containg protein data. Expected shape is (Nx2), with the first
        column being the sequences, and the second being the fitnesses

    seed_seq : str, default=None

        Enables the user to explicitly provide the seed sequence as a string

    seed_id : int,default=0

        Id of seed sequences within sequences and fitness

    csv_path : str,default=None

        Path to the csv file that should be imported using CSV loader function

    custom_columns : {"x_data"    : str,
                      "y_data"    : str
                      "index_col" : int}, default=None

        First two entries are custom strings to use as column headers when extracting
        data from CSV. Replaces default values of "Sequence" and "Fitness".

        Third value is the integer to use as the index column.

        They are passed to the function as keyword arguments

    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'

        String containing all allowable amino acids for tokenization functions

    saved_file : str, default=None

        Saved version of this class that will be loaded instead of instantiating a new one



    Attributes
    ----------
    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'

        String containing all allowable amino acids in tokenization functions

    sequence_mutation_locations : np.array(bool)

        Array that stores boolean values with Trues indicating that the position is
        mutated relative to the seed sequence

     mutated_positions: np.array(int)

        Numpy array that stores the integers of each position that is mutated

    d_data : {distance : index_array}

        A dictionary where each distance is a key and the values are the indexes of nodes
        with that distance from the seed sequence

    data : np.array

        Full, untokenized data. Two columns, first is sequences as strings, and second is fitnesses

    tokens : {tuple(tokenized_sequence) : index}

        A dictionary that stores a tuple format of the tokenized string with the index
        of it within the data array as the value. Used to rapidly perform membership checks

    sequences : np.array(str)

        A numpy array containing all sequences as strings

    seed_seq : str

        Seed sequence as a string

    tokenized : np.array, shape(N,L+1)

        Array containing each sequence with fitness appended onto the end.
        For the shape, N is the number of samples, and L is the length of the seed sequence

    mutation_array : np.array, shape(L*20,L)

        Array containing all possible mutations to produce sequences 1 amino acid away.
        Used by maxima generator to accelerate the construction of the graph.

        L is sequence length.

    self.hammings : np.array(N,)

        Numpy array of length number of samples, where each value is the hamming
        distance of the species at that index from the seed sequence.

    max_distance : int

        The maximum distance from the seed sequence within the dataset.

    graph : {tuple(tokenized_seq) : np.array[neighbour_indices]}

        A memory efficient storage of the graph that can be passed to graph visualisation packages

    num_minima : int

        The number of minima within the dataset

    num_maxima : int

        The number of maxima within the dataset

    extrema_ruggedness : float32

        The floating point ruggedness of the landscape calculated as the normalized
        number of maxima and minima.

    Written by Adam Mater, last revision 04-09-20
    """

    def __init__(
        self,
        seed_seq=None,
        seed_id=0,
        csv_path=None,
        custom_columns={"x_data": "Sequence", "y_data": "Fitness", "index_col": None},
        amino_acids="ACDEFGHIKLMNPQRSTVWY",
    ):
        self.csv_path = csv_path
        self.data = self.csvDataLoader(csv_path, **custom_columns)

        clock0 = time.time()
        self.amino_acids = amino_acids

        self.tokens = {
            x: y for x, y in zip(self.amino_acids, list(range(len(self.amino_acids))))
        }

        self.sequences = self.data[:, 0]
        self.fitnesses = self.data[:, 1]

        if seed_seq:
            self.seed_seq = seed_seq

        else:
            self.seed_id = seed_id
            self.seed_seq = self.sequences[self.seed_id]

        seq_len = len(self.seed_seq)

        self.tokenized = np.concatenate(
            (self.tokenize_data(), self.fitnesses.reshape(-1, 1)), axis=1
        )

        self.token_dict = {
            tuple(seq): idx for idx, seq in enumerate(self.tokenized[:, :-1])
        }

        self.mutated_positions = self.calc_mutated_positions()
        self.sequence_mutation_locations = self.boolean_mutant_array()
        # Stratifies data into different hamming distances

        # Contains the information to provide all mutants 1 amino acid away for a given sequence
        self.mutation_arrays = self.gen_mutation_arrays()

        subsets = {x: [] for x in range(seq_len + 1)}

        self.hammings = self.hamming_array()

        for distance in range(seq_len + 1):
            subsets[distance] = np.equal(
                distance, self.hammings
            )  # Stores an indexing array that isolates only sequences with that Hamming distance

        subsets = {k: v for k, v in subsets.items() if v.any()}

        self.max_distance = max(subsets.keys())

        self.d_data = subsets

        self.graph = self.build_graph()
        clock1 = time.time()

        self.num_minima, self.num_maxima = self.calculate_num_extrema()

        self.extrema_ruggedness = self.calc_extrema_ruggedness()
        clock2 = time.time()
        self.linear_slope, self.linear_RMSE, self.RS_ruggedness = self.rs_ruggedness()
        clock3 = time.time()

        self.extrema_time = clock3 - clock2 + clock1 - clock0
        self.rs_time = clock2 - clock0

    def get_distance(self, d, tokenize=False):
        """Returns all arrays at a fixed distance from the seed string

        Parameters
        ----------
        d : int

            The distance that you want extracted

        tokenize : Bool, False

            Whether or not the returned data will be in tokenized form or not.
        """
        assert d in self.d_data.keys(), "Not a valid distance for this dataset"

        if tokenize:
            return self.tokenized[self.d_data[d]]
        else:
            return self.data[self.d_data[d]]

    def get_mutated_positions(self, positions, tokenize=False):
        """
        Function that returns the portion of the data only where the provided positions
        have been modified.

        Parameters
        ----------
        positions : np.array(ints)

            Numpy array of integer positions that will be used to index the data.

        tokenize : Bool, default=False

            Boolean that determines if the returned data will be tokenized or not.
        """
        for pos in positions:
            assert pos in self.mutated_positions, "{} is not a valid position".format(
                pos
            )

        constants = np.setdiff1d(self.mutated_positions, positions)
        index_array = np.ones((len(self.seed_seq)), dtype=np.int8)
        index_array[positions] = 0
        mutated_indexes = np.all(
            np.invert(self.sequence_mutation_locations[:, constants]), axis=1
        )  # This line checks only the positions that have to be constant, and ensures that they all are

        if tokenize:
            return self.tokenized[mutated_indexes]
        else:
            return self.data[mutated_indexes]

    @staticmethod
    def hamming(str1, str2):
        """Calculates the Hamming distance between 2 strings"""
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def hamming_array(self, seq=None, data=None):
        """
        Function to calculate the hamming distances of every array using vectorized
        operations

        Function operates by building an array of the (Nxlen(seed sequence)) with
        copies of the tokenized seed sequence.

        This array is then compared elementwise with the tokenized data, setting
        all places where they don't match to False. This array is then inverted,
        and summed, producing an integer representing the difference for each string.

        Parameters:
        -----------
        seq : np.array[int], default=None

            Sequence which will be compared to the entire dataset.
        """
        if seq is None:
            tokenized_seq = np.array(self.tokenize(self.seed_seq))
        else:
            tokenized_seq = seq

        if data is None:
            data = self.tokenized[:, :-1]

        # hold_array     = np.zeros((len(self.sequences),len(tokenized_seq)))
        # for i,char in enumerate(tokenized_seq):
        #    hold_array[:,i] = char

        hammings = np.sum(np.invert(data == tokenized_seq), axis=1)

        return hammings

    @staticmethod
    def csvDataLoader(csvfile, x_data, y_data, index_col):
        """Simple helper function to load NK landscape data from CSV files into numpy arrays.
        Supply outputs to sklearn_split to tokenise and split into train/test split.

        Parameters
        ----------

        csvfile : str

            Path to CSV file that will be loaded

        x_data : str, default="Sequence"

            String key used to extract relevant x_data column from pandas dataframe of
            imported csv file

        y_data : str, default="Fitness"

            String key used to extract relevant y_data column from pandas dataframe  of
            imported csv file

        index_col : int, default=None

            Interger value, if provided, will determine the column to use as the index column

        returns np.array (Nx2), where N is the number of rows in the csv file

            Returns an Nx2 array with the first column being x_data (sequences), and the second being
            y_data (fitnesses)
        """

        data = pd.read_csv(csvfile, index_col=index_col)
        protein_data = data[[x_data, y_data]].to_numpy()

        return protein_data

    def tokenize(self, seq):
        """
        Simple static method which tokenizes an individual sequence
        """
        return [self.tokens[aa] for aa in seq]

    def boolean_mutant_array(self):
        return np.invert(self.tokenized[:, :-1] == self.tokenize(self.seed_seq))

    def calc_mutated_positions(self):
        """
        Determines all positions that were modified experimentally and returns the indices
        of these modifications.

        Because the Numpy code is tricky to read, here is a quick breakdown:

            self.tokenized is called, and the fitness column is removed by [:,:-1]
            Each column is then tested against the first
        """
        mutated_bools = np.invert(
            np.all(self.tokenized[:, :-1] == self.tokenize(self.seed_seq), axis=0)
        )  # Calculates the indices all of arrays which are modified.
        mutated_idxs = mutated_bools * np.arange(
            1, len(self.seed_seq) + 1
        )  # Shifts to the right so that zero can be counted as an idx
        return mutated_idxs[mutated_idxs != 0] - 1  # Shifts it back

    def tokenize_data(self):
        """
        Takes an iterable of sequences provided as one amino acid strings and returns
        an array of their tokenized form.

        Note : The tokenize function is not called and the tokens value is regenerated
        as it removes a lot of function calls and speeds up the operation significantly.
        """
        tokens = self.tokens
        return np.array([[tokens[aa] for aa in seq] for seq in self.sequences])

    def sklearn_data(
        self, data=None, distance=None, positions=None, split=0.8, shuffle=True
    ):
        """
        Parameters
        ----------
        data : np.array(NxM+1), default=None

            Optional data array that will be split. Added to the function to enable it
            to interface with lengthen sequences.

            Provided array is expected to be (NxM+1) where N is the number of data points,
            M is the sequence length, and the +1 captures the extra column for the fitnesses

        distance : int or [int], default=None

            The specific distance (or distances) from the seed sequence that the data should be sampled from.

        positions : [int], default=None

            The specific mutant positions that the data will be sampled from

        split : float, default=0.8, range [0-1]

            The split point for the training - validation data.

        shuffle : Bool, default=True

            Determines if the data will be shuffled prior to returning.

        returns : x_train, y_train, x_test, y_test

            All Nx1 arrays with train as the first 80% of the shuffled data and test
            as the latter 20% of the shuffled data.
        """

        assert 0 <= split <= 1, "Split must be between 0 and 1"

        if data is not None:
            data = data
        elif distance:
            if type(distance) == int:
                data = copy.copy(self.get_distance(distance, tokenize=True))
            else:
                data = collapse_concat(
                    [copy.copy(self.get_distance(d, tokenize=True)) for d in distance]
                )

        elif positions is not None:
            data = copy.copy(self.get_mutated_positions(positions, tokenize=True))
        else:
            data = copy.copy(self.tokenized)

        if shuffle:
            np.random.shuffle(data)

        split_point = int(len(data) * split)

        train = data[:split_point]
        test = data[split_point:]

        # Y data selects only the last column of Data
        # X selects the rest

        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]

        return (
            x_train.astype("int"),
            y_train.astype("float"),
            x_test.astype("int"),
            y_test.astype("float"),
        )

    def gen_mutation_arrays(self):
        leng = len(self.seed_seq)
        xs = np.arange(leng * len(self.amino_acids))
        ys = np.array(
            [[y for x in range(len(self.amino_acids))] for y in range(leng)]
        ).flatten()
        modifiers = np.array(
            [np.arange(len(self.amino_acids)) for x in range(leng)]
        ).flatten()
        return (xs, ys, modifiers)

    # Ruggedness Section
    def is_extrema(self, idx, graph=None):
        """
        Takes the ID of a sequence and determines whether or not it is a maxima given its neighbours

        Parameters:
        -----------
        idx : int

            Integer index of the sequence that will be checked as a maxima.
        """
        if graph is None:
            graph = self.graph

        neighbours = graph[tuple(self.tokenized[idx, :-1])]
        # print(self.fitnesses[neighbours])
        max_comparisons = np.greater(self.fitnesses[idx], self.fitnesses[neighbours])
        min_comparisons = np.less(self.fitnesses[idx], self.fitnesses[neighbours])
        if np.all(max_comparisons):
            return 1
        elif np.all(min_comparisons):  # Checks to see if the point is a minima
            return -1
        else:
            return 0

    def generate_mutations(self, seq):
        """
        Takes a sequence and generates all possible mutants 1 Hamming distance away
        using array substitution

        Parameters:
        -----------
        seq : np.array[int]

            Tokenized sequence array
        """
        seed = self.seed_seq
        hold_array = np.zeros(((len(seed) * len(self.amino_acids)), len(seed)))
        for i, char in enumerate(seq):
            hold_array[:, i] = char

        xs, ys, mutations = self.mutation_arrays
        hold_array[(xs, ys)] = mutations
        copies = np.invert(np.all(hold_array == seq, axis=1))
        return hold_array[copies]

    def calc_neighbours(self, seq, token_dict=None):
        """
        Takes a sequence and checks all possible neighbours against the ones that are actually present within the dataset.

        Parameters:
        -----------

        seq : np.array[int]

            Tokenized sequence array
        """
        if token_dict is None:
            token_dict = self.token_dict
        possible_neighbours = self.generate_mutations(seq)
        actual_neighbours = [
            token_dict[tuple(key)]
            for key in possible_neighbours
            if tuple(key) in token_dict
        ]
        return seq, actual_neighbours

    def calculate_num_extrema(self, idxs=None):
        """
        Calcaultes the number of maxima across a given dataset or array of indices
        """
        if idxs is None:
            idxs = range(len(self.sequences))
            graph = self.graph
        else:
            graph = self.build_graph(idxs=idxs)
            idxs = np.where(idxs)[0]

        print("Calculating the number of extrema")
        mapfunc = partial(self.is_extrema, graph=graph)
        results = np.array(list(map(mapfunc, tqdm.tqdm(idxs))))
        minima = -1 * np.sum(results[results < 0])
        maxima = np.sum(results[results > 0])
        return minima, maxima

    def calc_extrema_ruggedness(self):
        """
        Simple function that returns a normalized ruggedness value
        """
        ruggedness = (self.num_minima + self.num_maxima) / len(self.sequences)
        return ruggedness

    # Graph Section
    def build_graph(self, idxs=None):
        """
        Efficiently builds the graph of the protein landscape

        Parameters:
        -----------
        idxs : np.array[int]

            An array of integers that are used to index the complete dataset
            and provide a subset to construct a subgraph of the full dataset.
        """
        if idxs is None:
            print("Building Protein Graph for entire dataset")
            dataset = self.tokenized[:, :-1]
            token_dict = self.token_dict
            pool = mp.Pool(mp.cpu_count())

        else:
            print("Building Protein Graph For subset of length {}".format(sum(idxs)))
            dataset = self.tokenized[:, :-1][idxs]
            integer_indexes = np.where(idxs)[0]
            token_dict = {
                key: value
                for key, value in self.token_dict.items()
                if value in integer_indexes
            }
            if len(integer_indexes) < 100000:
                pool = mp.Pool(4)
            else:
                pool = mp.Pool(mp.cpu_count())

        mapfunc = partial(self.calc_neighbours, token_dict=token_dict)
        results = pool.map(mapfunc, tqdm.tqdm(dataset))
        neighbours = {tuple(key): value for key, value in results}
        return neighbours

    def extrema_ruggedness_subset(self, idxs):
        """
        Function that calculates the extrema ruggedness based on a subset of the
        full protein graph
        """
        minima, maxima = self.calculate_num_extrema(idxs=idxs)
        return (minima + maxima) / sum(idxs)

    def rs_ruggedness(self, log_transform=False, distance=None, split=1.0):
        """
        Returns the rs based ruggedness estimate for the landscape.

        Parameters
        ----------
        log_transform : bool, default=False

            Boolean value that determines if the base 10 log transform will be applied.
            The application of this was suggested in the work by Szengdo

        distance : int, default=None

            Determines the distance for data that will be sampled

        split : float, default=1.0, range [0-1]

            How much of the data is used to determine ruggedness
        """
        if distance:
            x_train, y_train, _, _ = self.sklearn_data(split=split, distance=distance)
        else:
            x_train, y_train, _, _ = self.sklearn_data(split=split)
        if log_transform:
            y_train = np.log10(y_train)

        lin_model = LinearRegression(n_jobs=mp.cpu_count()).fit(x_train, y_train)
        y_preds = lin_model.predict(x_train)
        coefs = lin_model.coef_
        rmse_predictions = np.sqrt(mean_squared_error(y_train, y_preds))
        slope = 1 / len(self.seed_seq) * sum([abs(i) for i in coefs])
        return [slope, rmse_predictions, rmse_predictions / slope]
