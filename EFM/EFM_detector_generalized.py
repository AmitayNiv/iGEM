from Bio.SeqIO import parse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from pandas import DataFrame, concat
from textwrap import wrap
import os
import glob
from pathlib import Path
import gzip
from os.path import join





### calculate part of sequence based on start and end indexes
def genome_cutter(start, end, seq):
    return seq[start:end]



def find_recombination_sites(example_seq, num_sites):

    ### count all sequences of length 16
    vectorizer = CountVectorizer(analyzer= 'char_wb', ngram_range=(16, 16))
    counter = vectorizer.fit_transform([example_seq]).toarray()

    ### find those that appear more than once
    sites_recombination = list(np.where(counter > 1)[1])

    if len(sites_recombination)==0:
        return DataFrame(columns = ['start', 'end'])

    ### get all sequences
    all_sites = vectorizer.get_feature_names()


    suspect_recombination = []

    for site in sites_recombination:
        ### get current site
        curr_seq = all_sites[site]
        ### get list of locations that are a match to the site
        list_regions = list(re.finditer(curr_seq.upper(), example_seq))
        ### extract coordinates from Match object
        list_regions = [x.span() for x in list_regions]

        suspect_recombination.extend(list_regions)

    ### this is now a list of tuples, containing coordinates of suspect recombination sites
    suspect_recombination = sorted(suspect_recombination)

    ### turn the tuples into a dataframe of start and end coordinates of the sites
    df_recombination = DataFrame(suspect_recombination, columns = ['start', 'end'])

    ### when we have matches larger than 16, they will turn into subsequent matches of 16. The following script is meant
    ### to join them together back to one larger region

    ### find the difference between current start and previous start, and between next end to current end. For ranges in
    ### the middle of a larger region, these will be 1 and 1. For our larger region, we only need the edges, so we get rid
    ### of the rest.
    df_recombination.loc[:, 'start_delta'] = df_recombination['start'] -df_recombination['start'].shift()
    df_recombination.loc[:, 'end_delta'] = df_recombination['end'].shift(-1)- df_recombination['end']
    df_recombination = df_recombination[(df_recombination.start_delta!=1.0)|(df_recombination.end_delta!=1.0)]

    ### starts of region will have end_delta==1, while ends of region will have start_delta = 1. We'll want to backpropagate
    ### the true end coordinate to the true start coordinate. So, we delete the end values in region starts, and backfill the end coordinates
    ### afterwards, we will keep only the region starts, which now have both coordinates correct
    df_recombination.loc[(df_recombination.end_delta == 1.0), 'end'] = None
    df_recombination.loc[:, 'end'] = df_recombination.loc[:, 'end'].fillna(method='bfill').astype(int)-1
    df_recombination = df_recombination[df_recombination.start_delta!=1.0][['start', 'end']]

    ### attach the segment of the genetic sequence marked by these coordinates
    df_recombination.loc[:, 'sequence'] = df_recombination.apply(lambda x: genome_cutter(x['start'], x['end'], example_seq), axis=1)

    ### merge same sequences, so can easily see where the duplicates are
    df_recombination = df_recombination.merge(df_recombination, on = 'sequence', suffixes = ('_1', '_2'))
    ### keep them as ordered matches - also gets rid of duplicates
    df_recombination = df_recombination[df_recombination.end_1<df_recombination.start_2]

    ### find length of site and distance between sites
    df_recombination.loc[:, 'location_delta'] = df_recombination.start_2-df_recombination.end_1
    df_recombination.loc[:, 'site_length'] = df_recombination.end_1-df_recombination.start_1

    ### insert empirical formula for mutation probability from paper.
    A = 5.8
    B = 1465.6
    C = 0
    alpha = 29


    df_recombination.loc[:, 'log10_prob_recombination_ecoli_1'] = (A+df_recombination['location_delta'])
    df_recombination.loc[:, 'log10_prob_recombination_ecoli_2'] = (-1*alpha/ df_recombination['site_length'])
    df_recombination.loc[:, 'log10_prob_recombination_ecoli_3'] = (df_recombination['site_length'])/(1+B*df_recombination['site_length']
                                                                                                     +C*df_recombination['location_delta'])

    df_recombination.loc[:, 'log10_prob_recombination_ecoli'] = ((df_recombination['log10_prob_recombination_ecoli_1'])**
                                                                 (df_recombination['log10_prob_recombination_ecoli_2']))*\
                                                                (df_recombination['log10_prob_recombination_ecoli_3'])

    df_recombination.loc[:, 'log10_prob_recombination_ecoli'] = df_recombination['log10_prob_recombination_ecoli'].apply(lambda x: np.log10(x))

    del df_recombination['log10_prob_recombination_ecoli_1']
    del df_recombination['log10_prob_recombination_ecoli_2']
    del df_recombination['log10_prob_recombination_ecoli_3']

    ### sort from mostly likely to mutate

    df_recombination = df_recombination.sort_values('log10_prob_recombination_ecoli', ascending=False)

    if type(num_sites) == int:
        df_recombination = df_recombination.head(num_sites)


    return df_recombination



def find_slippage_sites_length_L(sequence, L):

    ### this function takes a sequence and a length L, and finds all locations where sequences of L length repeat themselves
    ### back to back. For L=1, this means all locations where a nucleotides repeats 4 times or more. For L>1, all sites where
    ### a sequence of length L repeats 3 times or more.

    slippage_sites = []

    ### this process needs to repeated for all frameshifts up to L, because the repeating sequence can start in any frameshift.
    for frameshift in range(L):
        ### frame shift the whole sequence for ease of calculation
        curr_seq = sequence[frameshift:]
        ### split sequence into equal parts of length L (shortening the last part as needed).
        curr_seq_split = wrap(curr_seq, L)

        ### until what small sequence d owe need to check
        end_of_range = len(curr_seq_split)-2
        if L==1:
            end_of_range-=1
        for ii in range(end_of_range):

            ### in case of L>1, this expression is true when current sequence is equal to the next two. this is to mark
            ### the site that is prone to polymerase slippage
            is_followed2 = ((curr_seq_split[ii]==curr_seq_split[ii+1]) and (curr_seq_split[ii]==curr_seq_split[ii+2]) and L>1)
            ### relevant expression for L=1
            is_followed1 = ((curr_seq_split[ii]==curr_seq_split[ii+1]) and (curr_seq_split[ii]==curr_seq_split[ii+2]) and
                            (curr_seq_split[ii]==curr_seq_split[ii+3]) and L==1)

            ### save index of start and end of region, for L>1 and L = 1
            if is_followed2:
                curr_start = frameshift+ii*L
                curr_end = frameshift+L*(ii+3)
                slippage_sites.append((curr_start, curr_end))

            if is_followed1:
                curr_start = ii
                curr_end = ii+4
                slippage_sites.append((curr_start, curr_end))

    ### if no regions found, return empty dataframe
    if len(slippage_sites)==0:
        return DataFrame(columns = ['start', 'end', 'sequence', 'length_base_unit'])


    df_slippage = DataFrame(sorted(slippage_sites), columns = ['start', 'end'])

    ### once again, we have larger suspect regions represented as a sequence of small suspect regions. As before, we find
    ### the delta to nearby start and end indices. We get rid of middle regions, and from the edges find once again the
    ### site's coordinates

    df_slippage.loc[:, 'start_delta'] = df_slippage['start'] - df_slippage['start'].shift()
    df_slippage.loc[:, 'end_delta'] = df_slippage['end'].shift(-1) - df_slippage['end']
    df_slippage = df_slippage[(df_slippage.start_delta != 1.0) | (df_slippage.end_delta != 1.0)]
    df_slippage.loc[(df_slippage.end_delta == 1.0), 'end'] = None
    df_slippage.loc[:, 'end'] = df_slippage.loc[:, 'end'].fillna(method='bfill').astype(int)
    df_slippage = df_slippage[df_slippage.start_delta != 1.0][['start', 'end']]

    ### add sequence found, and length of base unit
    df_slippage.loc[:, 'sequence'] = df_slippage.apply(lambda x: genome_cutter(x['start'], x['end'], sequence), axis=1)
    df_slippage.loc[:, 'length_base_unit'] = L


    return df_slippage



def find_slippage_sites(seq, num_sites):

    ### create df of slippage sites for all base unit lengths up to 15
    slippage_sites_list = []
    for ii in range(1, 16):
        slippage_sites_list.append(find_slippage_sites_length_L(seq, ii))
    df_slippage = concat(slippage_sites_list, ignore_index=True)[['start', 'end', 'length_base_unit', 'sequence']]

    ### find nmber of repeats per site, and calculate mutation rate from empirical formula
    df_slippage.loc[:, 'num_base_units'] = df_slippage.sequence.apply(lambda x: len(x))/df_slippage.length_base_unit

    df_slippage.loc[:, 'log10_prob_slippage_ecoli'] = -4.749+0.063*df_slippage['num_base_units']
    df_slippage.loc[df_slippage.length_base_unit==1, 'log10_prob_slippage_ecoli'] = -12.9+0.729*df_slippage['num_base_units']

    ### return slippage sites, sorted by risk and limited in number of sites
    df_slippage= df_slippage.sort_values(['log10_prob_slippage_ecoli', 'length_base_unit'], ascending=[False, False])

    if type(num_sites) == int:
        df_slippage = df_slippage.head(num_sites)


    return df_slippage



def motif_prob_extractor(methylation_sites_path):
    ### open text file for extraction
    with open(methylation_sites_path, "r") as handle:
        motif_raw_data = handle.read()

    ### split text by motif
    motif_data_split = motif_raw_data.split('MOTIF')[1:]

    ### initiate dictionary which will keep motif probabilities
    motif_probs = {}

    for row in motif_data_split:
        ### get motif name from first row
        motif_name = row.split("\n")[0].split('_')[-1]
        ### get number of nucleotides in motif
        num_nucleotides = int(row.split("w= ")[1].split(' ')[0])
        ### get table of probabilities
        table_probs = [x for x in row.split("\n")[2:] if x!= '']

        ### extract num of nucleotides and probability of nucleotide per index
        motif_probs_curr = {x:{} for x in range(num_nucleotides)}
        motif_probs_curr['num_nucleotides'] = num_nucleotides

        for ii, prob_row in enumerate(table_probs):
            row_table_split = [float(x) for x in prob_row.split('\t')]
            motif_probs_curr[ii]['A'] = row_table_split[0]
            motif_probs_curr[ii]['C'] = row_table_split[1]
            motif_probs_curr[ii]['G'] = row_table_split[2]
            motif_probs_curr[ii]['T'] = row_table_split[3]

        motif_probs[motif_name] = motif_probs_curr

    ### define and sort a summarizing dataframe for later merge
    df_site_probs = DataFrame.from_dict(motif_probs).T.reset_index().rename(columns = {'index':'matching_motif'})
    cols = sorted([x for x in list(df_site_probs) if type(x)==int])
    df_site_probs = df_site_probs[['matching_motif', 'num_nucleotides']+cols]

    return motif_probs, df_site_probs




def site_motif_grader(start_index, motif, example_seq):
    ### find a similarity measure between the sequence starting in current index, and current motif

    conjugate_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

    num_nucleotides = motif['num_nucleotides']

    ### calculate the current sequence, and the reverse cojugate ending in current index
    curr_seq = example_seq[start_index:(start_index+num_nucleotides)]
    curr_seq_conjugate = ''.join([conjugate_dict[x] for x in curr_seq[::-1]])

    ### if we're already in the end, return probability 0.
    if len(curr_seq)<num_nucleotides:
        return num_nucleotides, -np.inf

    ### translate current sequence to log of match score with current motif
    curr_probs = [np.log10(motif[ii][x]) for ii, x in enumerate(curr_seq)]
    curr_probs_conj = [np.log10(motif[ii][x]) for ii, x in enumerate(curr_seq_conjugate)]

    ### take the mean is a total match score to the motif. Becuase we are in log space, we get the harmonic mean probability.
    mean_prob_log10 = np.mean(curr_probs)
    mean_prob_conj = np.mean(curr_probs_conj)

    ### take higher match between forward and conjugate
    mean_prob_log10 = max(mean_prob_log10, mean_prob_conj)

    return num_nucleotides, mean_prob_log10


def calc_max_site(start_index, example_seq, motif_probs):
    ### find highest matching site per index, and its descriptive parameters

    ### initiate all parameters
    log10_site_match = -np.inf
    end_index = start_index
    matching_motif = ''
    actual_site = ''
    actual_site_rev_conj = ''

    conjugate_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

    for motif in motif_probs:
        ### for the current motif, calculate site name, number pf nucleotides, and score
        site_name = motif
        num_nucleotides, curr_prob_log10 = site_motif_grader(start_index, motif_probs[motif], example_seq)

        ### if the current score is the best, replace current values
        if curr_prob_log10>log10_site_match:
            log10_site_match = curr_prob_log10
            matching_motif = site_name
            end_index = start_index+num_nucleotides-1
            actual_site = example_seq[start_index:(end_index+1)]
            actual_site_rev_conj = ''.join([conjugate_dict[x] for x in actual_site[::-1]])

    return (actual_site, actual_site_rev_conj, matching_motif, start_index, end_index, log10_site_match)



def site_ranker(example_seq, num_sites, motif_probs):

    ### per start index, extract best matching site and score, arrange in dataframe, sort and keep highest scores

    site_ranking_list = []

    for ii in range(len(example_seq)):
        site_ranking_curr = calc_max_site(ii, example_seq, motif_probs)
        site_ranking_list.append(site_ranking_curr)

    df_methylation = DataFrame(site_ranking_list, columns=['actual_site', 'actual_site_rev_conj', 'matching_motif', 'start_index', 'end_index', 'log10_site_match'])

    df_methylation = df_methylation.sort_values('log10_site_match', ascending=False)

    if type(num_sites) == int:
        df_methylation = df_methylation.head(num_sites)

    return df_methylation






def suspect_site_extractor(example_seq, compute_methylation, num_sites, methylation_sites_path, extension = ''):


    sites_collector = {}
    df_recombination = find_recombination_sites(example_seq, num_sites)

    df_slippage = find_slippage_sites(example_seq, num_sites)

    sites_collector['df_recombination'+extension] = df_recombination
    sites_collector['df_slippage'+extension] = df_slippage


    ### do methylation only if requested
    if compute_methylation == True:
        motif_probs, df_site_probs = motif_prob_extractor(methylation_sites_path)

        df_methylation = site_ranker(example_seq, num_sites, motif_probs)
        df_methylation = df_methylation.merge(df_site_probs, on='matching_motif', how='left')
        sites_collector['df_methylation' + extension] = df_methylation

    return sites_collector



#############################################################################################################
### feature generation


# def data_analyzer(df_slippage, df_recombination, df_methylation):
#
#
#
#     return df_features
#
#

def data_handler(data, file, output_path, compute_methylation, num_sites, methylation_sites_path, input_folder):

    if type(data) == list:

        recombination_collector = []
        slippage_collector = []
        methylation_collector = []

        for ii, record in enumerate(data[:30]):
            example_seq = str(record.seq)

            curr_sites_collector = suspect_site_extractor(example_seq, compute_methylation, num_sites, methylation_sites_path, extension = '_'+str(ii))

            df_recombination = curr_sites_collector['df_recombination_'+str(ii)]
            if len(df_recombination) >0:
                df_recombination.loc[:, 'sequence_number'] = str(ii)
            recombination_collector.append(df_recombination)

            df_slippage = curr_sites_collector['df_slippage_' + str(ii)]
            if len(df_slippage) >0:
                df_slippage.loc[:, 'sequence_number'] = str(ii)
            slippage_collector.append(df_slippage)

            if compute_methylation == True:
                df_methylation = curr_sites_collector['df_methylation_' + str(ii)]
                if len(df_methylation) > 0:
                    df_methylation.loc[:, 'sequence_number'] = str(ii)
                methylation_collector.append(df_methylation)

        new_path = file.split(input_folder)[1].split('.fasta')[0]

        curr_output_path = output_path+new_path

        Path(curr_output_path).mkdir(parents=True, exist_ok=True)

        df_recombination = concat(recombination_collector)
        df_slippage = concat(slippage_collector)


        df_recombination.to_csv(join(curr_output_path, r'recombination_sites.csv'))
        df_slippage.to_csv(join(curr_output_path, r'slippage_sites.csv'))


        if compute_methylation == True:
            df_methylation = concat(methylation_collector)
            df_methylation.to_csv(join(curr_output_path, r'methylation_sites.csv'))

    else:
        example_seq = str(data.seq)

        sites_collector = suspect_site_extractor(example_seq, compute_methylation, num_sites, methylation_sites_path)

        ### save to output
        new_path = file.split('.fasta')[0]

        curr_output_path = join(output_path, new_path)

        Path(curr_output_path).mkdir(parents=True, exist_ok=True)

        sites_collector['df_recombination'].to_csv(join(curr_output_path, r'recombination_sites.csv'))
        sites_collector['df_slippage'].to_csv(join(curr_output_path, r'slippage_sites.csv'))


        if compute_methylation == True:
            sites_collector['df_methylation'].to_csv(join(curr_output_path, r'methylation_sites.csv'))

    return




def main(input_folder = os.getcwd(), output_path = join(os.getcwd(), 'output'), compute_methylation = False, num_sites = None,
         methylation_sites_path = join(os.getcwd(), r'topEnriched.313.meme.txt'), test = False):


    """
    The function gets as input a directory. This directory and all subdirectories are copied into the output folder. Each fasta and fasta.gz
    file gets replaced by a directory of the same name, and populated by csv files, detailing recombination sites, slippage sites,
    and possibly methylation sites.


    Args:

        input_folder(str): path from which to read fasta and fasta.gz files. Default value is current path.

        output_path(str): directory into which to write csv's detailing suspect sites. Default value is 'output' within path of script.

        compute_methylation(bool): whether to calculate methylation sites. Relevant only for mammalian and insectoid cells.
        Default value is False.

        num_sites(Union[int, None]): How many values to keep per output file. If None, keep all. Default is None.

        methylation_sites_path(str): path of methylation sites file, which is an input to the methylation probability calculation.
        Default value is 'topEnriched.313.meme.txt' within script path.

        test(bool): whether this is a test run and you only need a couple of files for testing purposes. Default value: False.


    Returns:
        No variable. Saves output csv's in output_path.
    """


    files_1 = glob.glob(join(input_folder, '*', '*.fasta*'), recursive = True)
    files = glob.glob(join(input_folder, '*.fasta*'), recursive = True)

    if test == True:
        files_1 = files_1[0:1]
        files = files[0:1]

    files.extend(files_1)

    for file in files:
        with open(file, "rU") as handle:
            data = list(parse(handle, "fasta"))

        data_handler(data, file, output_path, compute_methylation, num_sites, methylation_sites_path, input_folder)

    for file in glob.glob(join(input_folder, '*', '*.fasta.gz'), recursive = True):
        with gzip.open(file, "rt") as handle:
            data = list(parse(handle, "fasta"))

        data_handler(data, file, output_path, compute_methylation, num_sites, methylation_sites_path, input_folder)

    return


# input_folder = join(os.getcwd(), 'fasta_draft_200703')
# output_path = join(os.getcwd(), 'new_output', 'output')
# compute_methylation = True
# num_sites = 100
# methylation_sites_path = join(os.getcwd(), r'topEnriched.313.meme.txt')
# test = True
#
# main(input_folder = input_folder, output_path = output_path, compute_methylation = compute_methylation, num_sites = num_sites,
#      methylation_sites_path = methylation_sites_path, test = test)
#
#
