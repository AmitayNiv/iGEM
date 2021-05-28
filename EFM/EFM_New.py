from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import pandas as pd
from textwrap import wrap
from pathlib import Path
import \
    python_codon_tables as pct  # Provides codon usage tables as dictionnaries, for Python 3+. see https://pypi.org/project/python_codon_tables/
from dnachisel import *  # a library for optimizing DNA sequences with respect to a set of constraints and optimization objectives. see https://github.com/Edinburgh-Genome-Foundry/DNAChisel
from dnachisel import reports  # to create a pdf report and anotated genebank file
from Bio import SeqIO
from os import getcwd
from glob import glob
import gzip
from os.path import join, sep
import pypugjs
import pypugjs.ext
import pypugjs.ext.jinja
import pypugjs.ext.html
import pypugjs.ext.mako
import pypugjs.ext.underscore
import zipfile
from os import remove
from sequenticon import sequenticon


def plot_sequenticons(seq, final_sequence, output_path, target):
    """Helper function. 	
    This function plots the sequenticon of the sequence before and after the optimization process. 	
    The icons are being saved in the specified output path.	
    The function has no output parameter. 	

    Parameters	
    ------------	
    seq: the original sequence (SeqRecord or string).	
    final_sequence: the optimized sequence, after applying the 'translation_optimizer' function (string). 	
    output_path: the plot will be saved in this path. 	
    target: the path to the zip report file	

    """
    ### original sequence (before optimization): extract [0:45] for convenience	
    sequenticon(seq, output_format="png", size=24, output_path=join(output_path, 'sequenticon_before.png'))
    sequenticon(final_sequence, output_format="png", size=24, output_path=join(output_path, 'sequenticon_after.png'))

    # Open a zip file at the given filepath. If it doesn't exist, create one.	
    # If the directory does not exist, it fails with FileNotFoundError	
    filepath = target
    with zipfile.ZipFile(filepath, 'a') as zipf:
        # Add a file located at the source_path to the destination within the zip file. 	
        source_path = join(output_path, 'sequenticon_before.png')  # from
        destination = 'sequenticon_before.png'  # to
        zipf.write(source_path, destination)
        remove(source_path)  # remove the file from the output_path

        source_path = join(output_path, 'sequenticon_after.png')  # from
        destination = 'sequenticon_after.png'  # to
        zipf.write(source_path, destination)
        remove(source_path)  # remove the file from the output_path


def modify_df_slippage(df_slippage):
    """Helper function.	
    The input is a dataframe containing all the identified slippage areas.	
    The function converts each row with N = num_base_units to N-1 rows of the base units to be avoided.	
    If the length of the base unit is 1, the output dataframe will only include half of the base units (with "jumps")	
    This way, each basic unit will be treated as a constraint in the optimization function,	
    since we want to avoid most of the repetitions.	
    Example: In an input df_slippage, one of the rows indicates that the sequence "TGTGTGTG" includes a base unit	
    of length L=3 with N=4 repetitions.	
         start     stop     sequence   length_base_unit  num_base_units	
    __________________________________________________________________	
    0    395        403     TGTGTGTG         2                 4	
    So the output dataframe of the current function will convert this row to N-1=three rows:	
       start stop sequence	
    __________________	
    0  395    397  TG	
    1  397    399  TG	
    2  399    401  TG	
    such that each row contains a basic unit and its indices.	
    """
    df = pd.DataFrame(columns=('start', 'end', 'sequence'))
    for j in range(0, len(df_slippage)):  # rows of df_slippage	
        NBU = df_slippage.iloc[j].num_base_units
        L = df_slippage.iloc[j].length_base_unit
        if L == 1:
            step = 2
        else:
            step = 1
        for i in range(0, NBU - 1, step):
            df = df.append({'sequence': df_slippage.iloc[j].sequence[int(i * L):int(i * L + L)],
                            'start': int(df_slippage.iloc[j].start + i * L),
                            'end': int(df_slippage.iloc[j].start + i * L + L)},
                           ignore_index=True)
    return df


def convert_df_to_constraints(df_fixed):
    """Helper function.	
    The input df has 3 columns: {start, end, and sequence}.	
    The "sequence" is a pattern, and the "start" and "end" specify the location of this pattern.	
    First, the function creates from this df a dictionary that maps each pattern constraint to its location	
    (and convert the location to a tuple).	
    The result is in following form: patterns = {"TTT":(292,292+7),"CTGCTGCTG":(673,673+9)}.	
    Then, the function converts the patterns into a constraints, to be used an input to the DNAChisel problem.	
    """
    patterns = df_fixed.set_index('sequence')[['start', 'end']].T.apply(tuple)
    pattern_constraints = [AvoidPattern(patterns.keys()[k], location=patterns.values[k]) for k in
                           range(0, len(patterns.keys()))]
    return pattern_constraints


def EFM_optimizer(seq, miniGC=0.3, maxiGC=0.7, window_size_GC=None, method='use_best_codon',
                  organism_name="not_specified",
                  df_recombination=pd.DataFrame(), df_slippage=pd.DataFrame(), df_methylation=pd.DataFrame(),
                  curr_output_path=None,
                  filename='Optimization_report_staubility.zip', with_report=False, indexes=None):
    """	
    Description	
    ----------	
    This function optimizes the input sequence (string or SeqRecord) based on the identified suspected area for	
    recombination, slippage and methylation, that are given as inputs (df_recombination, df_slippage, df_methylation).	
    The identification of those areas is based on the principles that are described in the EFM calculator web tool and	
    the article cited below.	
    Those areas contain patterns that are likely to go under recombination, slippage or methylation, thus avoiding those	
    patterns should increase the stability of the input gene.	
    The translation of the gene is kept (no changes to the amino acid sequence).	
    Citation: Benjamin R. Jack, Sean P. Leonard, Dennis M. Mishler, Brian A. Renda, Dacia Leon, Gabriel A. Suárez, and	
    Jeffrey E Barrick (2015). Predicting the genetic stability of engineered DNA sequences with the EFM Calculator.	
    ACS Synthetic Biology. Just Accepted Manuscript. DOI: 10.1021/acssynbio.5b00068	
    Other aspects that this optimizer takes into consideration are:	
    a. If the sequence is divisible by 3, the optimization parameter is "Codon usage fraction":	
    the relative frequency of the codon in the host genome (specified by the organism_name input).	
    Meaning, each codon is optimized based on its frequency in the host, in one of the methods below.	
    The frequency data is from kazusa database (http://www.kazusa.or.jp/codon/readme_codon.html).	
    Citation: Codon usage tabulated from the international DNA sequence databases, status for the year 2000.,	
    Nakamura, Y., Gojobori, T. and Ikemura, T. (2000) Nucl. Acids Res. 28, 292.	
    b. Codon Optimization method (for sequences that are divisible by 3):	
        - For method = "use_best_codon", every codon will be replaced by the "best" (i.e. most frequent) synonymous codon	
          in the target organism. This is equivalent to Codon Adaptation Index (CAI) optimization.	
        - For method = "match_codon_usage", the final sequence's codon usage will match as much as possible the codon usage	
          profile of the target species (this method is used throughout the literature,	
          see for instance Hale and Thomson 1998).	
        - For method = "harmonize_rca", Each codon will be replaced by a synonymous codon whose usage in the target organism	
          matches the usage of the original codon in its host organism (as per Claassens 2017).	
        Those methods are provided through the use of DNAchisel package for the optimization task.	
    c. GC content (optional) - the requested range (minimum and maximum) of the percentage of nitrogenous bases in the	
       sequence. the algorithm will split the sequence to windows of a specified size and on optimize each window.	
       Basically, The lower the GC content, the more stable is the sequence, so one should take that into consideration.	
    Parameters	
    ----------	
    seq: string of ACGT alphabet (copy-paste of DNA sequence), or SeqRecord (originated from a fasta file)	
    miniGC and maxiGC: a numerical value from 0 to 1, specifies the range of GC content.	
    window_size_GC: numerical. The window size (number of nucleotides) in which the requested GC content is to	
       be maintained.	
    method: optimization method.	
       This is a string from the following: {"use_best_codon", "match_codon_usage", "harmonize_rca"}.	
    organism_name: the name of the host of the gene. The codon optimization is done according to the host codon's frequency.	
        This is a string from the following: {'b_subtilis', 'c_elegans', 'd_melanogaster', 'e_coli',	
        'g_gallus', 'h_sapiens', 'm_musculus', 'm_musculus_domesticus', 's_cerevisiae','not_specified'}.	
        One can access this list (aside from 'not_specified') via "python_codon_tables.available_codon_tables_names".	
        If the organism is 'not_specified' then the codon optimization objective will not be defined.	
    df_recombination, df_slippage, df_methylation: dataframes, each containing a list of patterns and their locations that	
                                                may influence the genetic stability of the gene and thus should be avoided.	
                                                These dataframes will be saved as csv. files in a different function,	
                                                as the output of the EFM calculator.	
    indexes: a tuple that specifies the ORF indices. For the sub-sequence in those indices, the optimizer enforces	
        translation (keeps amino-acid sequence), and optimizes the codon usage. The rest of the sequence is optimized by means	
        of GC content and EFM constraints (if provided).	
    curr_output_path: a path in which the output report will be saved (see "returns" below).	
    filename: the file name (of the output report). with a ".zip" suffix.	
    with_report: a flag that indicates if the function will output a pdf report or just optimize the sequence.	
    Returns	
    ----------	
    final_sequence: the optimized sequence	
    final_record: a brief summary of the changes, includes sequence edits	
    exported file named 'Translation_report.zip' if the input sequence is a string,	
        or 'Translation_report_seqID.zip' if the input is FASTA format with an id.	
        The file will be saved in "curr_output_path" folder.	
        This file contains a report of the changes in anotated genbank format, in a pdf format and csv lists,	
        all including a detailed description of the changes from the constraints and objectives."""
    # set a default value for the window size as 1/50 of the sequence length.	
    if window_size_GC is None:
        window_size_GC = round(len(seq) / 50)
    # Match the weight table for the organism (when the organism is specified):	
    if organism_name == 'not_specified':
        obj = []
    else:
        codon_usage_table = pct.get_codons_table(organism_name).copy()
        # objective function:
        obj = [CodonOptimize(species=organism_name, location=indexes, codon_usage_table=codon_usage_table.copy(),
                             method=method)]
    ## Define area for codon optimization while keeping amino-acid translation:	
    if indexes == None:
        indexes = (0, len(seq))
        # DEFINE THE CONTSTRAINTS:	
    cnst = [
        EnforceGCContent(mini=miniGC, maxi=maxiGC, window=window_size_GC),
        # enforce GC content between 30% to 50% (default)	
        EnforceTranslation(location=indexes)  # Enforce a specific amino-acid sequence translation.	
    ]
    ### EFM constraints (if given):	
    # recombination:	
    if not df_recombination.empty:
        # change column names to the same name in each df and then send to a function that produces the patterns dictionary:	
        df_rec = df_recombination[0:10].copy()[['start_1', 'end_1', 'sequence']].rename(
            columns={'start_1': 'start', 'end_1': 'end'})
        # convert df to a list of constraints to be used as input for the optimization problem:	
        cnst_rec = convert_df_to_constraints(df_rec)
        # add to the constraints:	
        cnst.extend(cnst_rec)
    # slippage:	
    if not df_slippage.empty:
        # change column names to the same name in each df and then send to a function that produces the patterns dictionary:	
        # convert the slippage dataframe to the basic repeated units for the constrains:	
        df_slip = df_slippage[0:10].copy()
        df_slip = df_slip.loc[df_slip.log10_prob_slippage_ecoli > -9]  # only 'severe' constraints.	
        df_slip = modify_df_slippage(df_slip)
        # convert df to a list of constraints to be used as input for the optimization problem:	
        cnst_slip = convert_df_to_constraints(df_slip)
        # add to the constraints:	
        cnst.extend(cnst_slip)
    # methylation:	
    if not df_methylation.empty:
        # change column names to the same name in each df and then send to a function that produces the patterns dictionary:	
        df_meth = df_methylation[0:10].copy()[['start_index', 'end_index', 'actual_site']].rename(
            columns={'start_index': 'start', 'end_index': 'end', 'actual_site': 'sequence'})
        df_meth.loc[:, 'start'] = df_meth['start'].astype(int)
        df_meth.loc[:, 'end'] = df_meth['end'].astype(int)
        # convert df to a list of constraints to be used as input for the optimization problem:	
        cnst_meth = convert_df_to_constraints(df_meth)
        # add to the constraints:	
        cnst.extend(cnst_meth)
    # DEFINE THE OPTIMIZATION PROBLEM	
    flag = 1
    while flag < 30:  # while there are less than 10 hard constraints that were not satisfied	
        problem = DnaOptimizationProblem(
            sequence=seq,
            constraints=cnst,
            objectives=obj
        )
        # SOLVE THE CONSTRAINTS, OPTIMIZE WITH RESPECT TO THE OBJECTIVE AND PRODUCE A REPORT	
        try:
            problem.resolve_constraints()
            flag = 0  # all constraints passed.	
            break
        except NoSolutionError as e:
            cnst.remove(e.constraint)  # remove the problematic contstraint	
            # print("Warning "+ str(flag) + ": The constraint " +str(e.constraint)+" has been failed. trying to solve the problem without it.")	
            flag = flag + 1
    else:
        raise NoSolutionError(
            "Unfortunately, more than 30 hard constraints were not satistied." + str(flag),
            problem=problem
        )
    ## OPTIMIZE OBJECTIVE FUNCTION:	
    if with_report == False:
        problem.optimize()  # without a report.	
    else:
        target = join(curr_output_path, filename)  # define the exported file name (and path)	
        try:
            reports.optimization_reports.write_optimization_report(target=target, problem=problem,
                                                                   project_name="staubility_EFM_optimizer",
                                                                   plot_figure=True)
        except FileNotFoundError:
            # The sequence is too long or perhaps there are too many changes to be displayed in an annotated figure,	
            # so the report will be produced without it.	
            # The annotated sequence after changes can be obtained from the exported genebank file.	
            reports.optimization_reports.write_optimization_report(target=target, problem=problem,
                                                                   project_name="staubility_EFM_optimizer",
                                                                   plot_figure=False)
    # GET THE FINAL SEQUENCE (AS STRING OR ANNOTATED BIOPYTHON RECORDS)	
    final_sequence = problem.sequence  # string	
    if with_report == True:
        plot_sequenticons(seq, final_sequence, curr_output_path, target)  ##plot the sequenticons
    # final_record = problem.to_record(with_sequence_edits=True)	
    # if isinstance(seq, SeqRecord):  # if the original sequence is a fasta file:	
    #     final_record.id = seq.id	
    #     final_record.name = seq.name	
    #     final_record.description = seq.description	
    return final_sequence


### calculate part of sequence based on start and end indexes
def genome_cutter(start, end, seq):
    return seq[start:end]


def find_recombination_sites(example_seq, num_sites):
    ### count all sequences of length 16	
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(16, 16))
    counter = vectorizer.fit_transform([example_seq]).toarray()
    ### find those that appear more than once	
    sites_recombination = list(np.where(counter > 1)[1])
    if len(sites_recombination) == 0:
        return pd.DataFrame(
            columns=['start_1', 'end_1', 'sequence', 'start_2', 'end_2', 'location_delta', 'site_length',
                     'log10_prob_recombination_ecoli', 'sequence_number'])
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
    df_recombination = pd.DataFrame(suspect_recombination, columns=['start', 'end'])
    ### when we have matches larger than 16, they will turn into subsequent matches of 16. The following script is meant	
    ### to join them together back to one larger region	
    ### find the difference between current start and previous start, and between next end to current end. For ranges in	
    ### the middle of a larger region, these will be 1 and 1. For our larger region, we only need the edges, so we get rid	
    ### of the rest.	
    df_recombination.loc[:, 'start_delta'] = df_recombination['start'] - df_recombination['start'].shift()
    df_recombination.loc[:, 'end_delta'] = df_recombination['end'].shift(-1) - df_recombination['end']
    df_recombination = df_recombination[(df_recombination.start_delta != 1.0) | (df_recombination.end_delta != 1.0)]
    ### starts of region will have end_delta==1, while ends of region will have start_delta = 1. We'll want to backpropagate	
    ### the true end coordinate to the true start coordinate. So, we delete the end values in region starts, and backfill the end coordinates	
    ### afterwards, we will keep only the region starts, which now have both coordinates correct	
    df_recombination.loc[(df_recombination.end_delta == 1.0), 'end'] = None
    df_recombination.loc[:, 'end'] = df_recombination.loc[:, 'end'].fillna(method='bfill').astype(int) - 1
    df_recombination = df_recombination[df_recombination.start_delta != 1.0][['start', 'end']]
    ### attach the segment of the genetic sequence marked by these coordinates	
    df_recombination.loc[:, 'sequence'] = df_recombination.apply(
        lambda x: genome_cutter(x['start'], x['end'], example_seq), axis=1)
    ### merge same sequences, so can easily see where the duplicates are	
    df_recombination = df_recombination.merge(df_recombination, on='sequence', suffixes=('_1', '_2'))
    ### keep them as ordered matches - also gets rid of duplicates	
    df_recombination = df_recombination[df_recombination.end_1 < df_recombination.start_2]
    ### find length of site and distance between sites	
    df_recombination.loc[:, 'location_delta'] = df_recombination.start_2 - df_recombination.end_1
    df_recombination.loc[:, 'site_length'] = df_recombination.end_1 - df_recombination.start_1
    ### insert empirical formula for mutation probability from paper.	
    A = 5.8
    B = 1465.6
    C = 0
    alpha = 29
    df_recombination.loc[:, 'log10_prob_recombination_ecoli_1'] = (A + df_recombination['location_delta'])
    df_recombination.loc[:, 'log10_prob_recombination_ecoli_2'] = (-1 * alpha / df_recombination['site_length'])
    df_recombination.loc[:, 'log10_prob_recombination_ecoli_3'] = (df_recombination['site_length']) / (
                1 + B * df_recombination['site_length']
                + C * df_recombination['location_delta'])
    df_recombination.loc[:, 'log10_prob_recombination_ecoli'] = ((df_recombination[
        'log10_prob_recombination_ecoli_1']) **
                                                                 (df_recombination[
                                                                     'log10_prob_recombination_ecoli_2'])) * \
                                                                (df_recombination['log10_prob_recombination_ecoli_3'])
    df_recombination.loc[:, 'log10_prob_recombination_ecoli'] = df_recombination[
        'log10_prob_recombination_ecoli'].apply(lambda x: np.log10(x))
    del df_recombination['log10_prob_recombination_ecoli_1']
    del df_recombination['log10_prob_recombination_ecoli_2']
    del df_recombination['log10_prob_recombination_ecoli_3']
    ### sort from mostly likely to mutate	
    df_recombination = df_recombination.sort_values('log10_prob_recombination_ecoli', ascending=False)
    if num_sites < np.inf:
        df_recombination = df_recombination.head(num_sites)
    for col in df_recombination:
        if df_recombination[col].isnull().all():
            del df_recombination[col]
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
        end_of_range = len(curr_seq_split) - 2
        if L == 1:
            end_of_range -= 1
        for ii in range(end_of_range):
            ### in case of L>1, this expression is true when current sequence is equal to the next two. this is to mark	
            ### the site that is prone to polymerase slippage	
            is_followed2 = ((curr_seq_split[ii] == curr_seq_split[ii + 1]) and (
                        curr_seq_split[ii] == curr_seq_split[ii + 2]) and L > 1)
            ### relevant expression for L=1	
            is_followed1 = ((curr_seq_split[ii] == curr_seq_split[ii + 1]) and (
                        curr_seq_split[ii] == curr_seq_split[ii + 2]) and
                            (curr_seq_split[ii] == curr_seq_split[ii + 3]) and L == 1)
            ### save index of start and end of region, for L>1 and L = 1	
            if is_followed2:
                curr_start = frameshift + ii * L
                curr_end = frameshift + L * (ii + 3)
                slippage_sites.append((curr_start, curr_end))
            if is_followed1:
                curr_start = ii
                curr_end = ii + 4
                slippage_sites.append((curr_start, curr_end))
    ### if no regions found, return empty dataframe	
    if len(slippage_sites) == 0:
        return pd.DataFrame(columns=['start', 'end', 'sequence', 'length_base_unit'])
    df_slippage = pd.DataFrame(sorted(slippage_sites), columns=['start', 'end'])
    ### once again, we have larger suspect regions represented as a sequence of small suspect regions. As before, we find	
    ### the delta to nearby start and end indices. We get rid of middle regions, and from the edges find once again the	
    ### site's coordinates	
    df_slippage.loc[:, 'start_delta'] = df_slippage['start'] - df_slippage['start'].shift()
    df_slippage.loc[:, 'end_delta'] = df_slippage['end'].shift(-1) - df_slippage['end']
    df_slippage = df_slippage[(df_slippage.start_delta != 1.0) | (df_slippage.end_delta != 1.0)]
    df_slippage.loc[(df_slippage.end_delta == 1.0), 'end'] = None
    df_slippage.loc[:, 'end'] = df_slippage.loc[:, 'end'].fillna(method='bfill').astype(int)
    df_slippage = df_slippage[df_slippage.start_delta != 1.0][['start', 'end']]
    ### round down end index to have complete number of units	
    df_slippage.loc[:, 'end'] = (((df_slippage['end'] - df_slippage['start']) / L).astype(int) * L) + df_slippage[
        'start']
    ### add sequence found, and length of base unit	
    df_slippage.loc[:, 'sequence'] = df_slippage.apply(lambda x: genome_cutter(x['start'], x['end'], sequence), axis=1)
    df_slippage.loc[:, 'length_base_unit'] = L
    return df_slippage


def find_slippage_sites(seq, num_sites):
    ### create df of slippage sites for all base unit lengths up to 15	
    slippage_sites_list = []
    for ii in range(1, 16):
        slippage_sites_list.append(find_slippage_sites_length_L(seq, ii))
    df_slippage = pd.concat(slippage_sites_list, ignore_index=True)[['start', 'end', 'length_base_unit', 'sequence']]
    ### find nmber of repeats per site, and calculate mutation rate from empirical formula	
    df_slippage.loc[:, 'num_base_units'] = (
                df_slippage.sequence.apply(lambda x: len(x)) / df_slippage.length_base_unit).astype(int)
    df_slippage.loc[:, 'log10_prob_slippage_ecoli'] = -4.749 + 0.063 * df_slippage['num_base_units']
    df_slippage.loc[df_slippage.length_base_unit == 1, 'log10_prob_slippage_ecoli'] = -12.9 + 0.729 * df_slippage[
        'num_base_units']
    ### return slippage sites, sorted by risk and limited in number of sites	
    df_slippage = df_slippage.sort_values(['log10_prob_slippage_ecoli', 'length_base_unit'], ascending=[False, False])
    if num_sites < np.inf:
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
        table_probs = [x for x in row.split("\n")[2:] if x != '']
        ### extract num of nucleotides and probability of nucleotide per index	
        motif_probs_curr = {x: {} for x in range(num_nucleotides)}
        motif_probs_curr['num_nucleotides'] = num_nucleotides
        for ii, prob_row in enumerate(table_probs):
            row_table_split = [float(x) for x in prob_row.split('\t')]
            motif_probs_curr[ii]['A'] = row_table_split[0]
            motif_probs_curr[ii]['C'] = row_table_split[1]
            motif_probs_curr[ii]['G'] = row_table_split[2]
            motif_probs_curr[ii]['T'] = row_table_split[3]
        motif_probs[motif_name] = motif_probs_curr
    ### define and sort a summarizing dataframe for later merge	
    df_site_probs = pd.DataFrame.from_dict(motif_probs).T.reset_index().rename(columns={'index': 'matching_motif'})
    cols = sorted([x for x in list(df_site_probs) if type(x) == int])
    df_site_probs = df_site_probs[['matching_motif', 'num_nucleotides'] + cols]
    return motif_probs, df_site_probs


def site_motif_grader(start_index, motif, example_seq, curr_prob):
    ### find a similarity measure between the sequence starting in current index, and current motif	
    conjugate_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    num_nucleotides = motif['num_nucleotides']
    ### calculate the current sequence, and the reverse cojugate ending in current index	
    curr_seq = example_seq[start_index:(start_index + num_nucleotides)]
    curr_seq_conjugate = ''.join([conjugate_dict[x] for x in curr_seq[::-1]])
    ### if we're already in the end, return probability 0.	
    if len(curr_seq) < num_nucleotides:
        return num_nucleotides, -np.inf
    sum_prob_log10 = 0
    sum_prob_conj = 0
    ### translate current sequence to log of match score with current motif	
    for ii, x in enumerate(curr_seq):
        enzyme_prob = np.log10(motif[ii][x])
        sum_prob_log10 += enzyme_prob
        if sum_prob_log10 < curr_prob:
            break
    for ii, x in enumerate(curr_seq_conjugate):
        enzyme_prob = np.log10(motif[ii][x])
        sum_prob_conj += enzyme_prob
        if sum_prob_conj < curr_prob:
            break
    ### take higher match between forward and conjugate	
    sum_prob_log10 = max(sum_prob_log10, sum_prob_conj)
    return num_nucleotides, sum_prob_log10


def calc_max_site(start_index, example_seq, motif_probs, limit_output, min_score):
    ### find highest matching site per index, and its descriptive parameters	
    ### initiate all parameters	
    log10_site_match = -np.inf
    if limit_output:
        log10_site_match = min_score
    curr_prob_log10 = log10_site_match
    end_index = start_index
    matching_motif = ''
    actual_site = ''
    actual_site_rev_conj = ''
    conjugate_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    for motif in motif_probs:
        ### for the current motif, calculate site name, number pf nucleotides, and score	
        site_name = motif
        num_nucleotides, curr_prob_log10 = site_motif_grader(start_index, motif_probs[motif], example_seq,
                                                             curr_prob_log10)
        ### if the current score is the best, replace current values	
        if curr_prob_log10 > log10_site_match:
            log10_site_match = curr_prob_log10
            matching_motif = site_name
            end_index = start_index + num_nucleotides + 1
            actual_site = example_seq[start_index:(end_index)]
            actual_site_rev_conj = ''.join([conjugate_dict[x] for x in actual_site[::-1]])
        else:
            curr_prob_log10 = log10_site_match
    if log10_site_match == min_score:
        log10_site_match = -np.inf
        return (), log10_site_match
    return (actual_site, actual_site_rev_conj, matching_motif, start_index, end_index,
            log10_site_match), log10_site_match


def site_ranker(example_seq, effective_num_sites, motif_probs):
    ### per start index, extract best matching site and score, arrange in dataframe, sort and keep highest scores	
    site_ranking_list = []
    scores_list = [-np.inf]
    min_score = -np.inf
    for ii in range(len(example_seq)):
        if ii < effective_num_sites:
            site_ranking_curr, new_score = calc_max_site(ii, example_seq, motif_probs, False, -np.inf)
        else:
            site_ranking_curr, new_score = calc_max_site(ii, example_seq, motif_probs, True, min_score)
        site_ranking_list.append(site_ranking_curr)
        if effective_num_sites < np.inf:
            if new_score > scores_list[-1]:
                scores_list.append(new_score)
                scores_list = sorted(scores_list, reverse=True)[:effective_num_sites]
                # print('sort_time: ' + str(time()-start_sort))
                min_score = scores_list[-1]
    df_methylation = pd.DataFrame(site_ranking_list,
                                  columns=['actual_site', 'actual_site_rev_conj', 'matching_motif', 'start_index',
                                           'end_index', 'log10_site_match'])
    df_methylation = df_methylation.sort_values('log10_site_match', ascending=False)
    if effective_num_sites < np.inf:
        df_methylation = df_methylation.head(effective_num_sites)
    return df_methylation


def suspect_site_extractor(example_seq, compute_methylation, num_sites, methylation_sites_path, extension=''):
    sites_collector = {}
    df_recombination = find_recombination_sites(example_seq, num_sites)
    print('finished finding RMD sites')
    df_slippage = find_slippage_sites(example_seq, num_sites)
    print('finished finding SSR sites')
    sites_collector['df_recombination' + extension] = df_recombination
    sites_collector['df_slippage' + extension] = df_slippage
    ### do methylation only if requested	
    if compute_methylation == True:
        motif_probs, df_site_probs = motif_prob_extractor(methylation_sites_path)
        df_methylation = site_ranker(example_seq, num_sites, motif_probs)
        df_methylation = df_methylation.merge(df_site_probs, on='matching_motif', how='left')
        print('finished finding methylation sites')
        sites_collector['df_methylation' + extension] = df_methylation
    return sites_collector


def data_handler(data, file, output_path, compute_methylation, num_sites, methylation_sites_path, input_folder,
                 optimize, miniGC, maxiGC, method, organism_name, indexes):
    print('starting the following file: ' + file)
    recombination_collector = []
    slippage_collector = []
    methylation_collector = []
    filename_indexes = file.split(input_folder + sep)[1].split('.fasta')[0]
    # define a new output folder inside "output_path", with name according to the 'file' name.	
    new_path = file.split(input_folder)[1].split('.fasta')[0]
    curr_output_path = output_path + new_path
    Path(curr_output_path).mkdir(parents=True, exist_ok=True)
    for ii, record in enumerate(data):  # the loop iterates on sequences inside a single FASTA file.	
        print('____________________________')
        example_seq = str(record.seq).upper()
        seq_indexes = str(ii)
        print('starting sequence number ' + seq_indexes)
        ## FIRST OPTIMIZATION - optimize codon usage and GC content, without EFM constraints.	
        if indexes == None:
            relevant_indexes = None
        else:
            # seq_indexes is the sequence identifier, not the start-stop.	
            relevant_indexes = indexes[(filename_indexes, seq_indexes)]  ##list
            relevant_indexes[0] = relevant_indexes[0] - 1  # convert to python	
            relevant_indexes = tuple(relevant_indexes)  # tuple (for the optimizer)	
        if optimize:
            example_seq = EFM_optimizer(example_seq, window_size_GC=None, curr_output_path=output_path,
                                        filename='EFM_without_report_' + str(ii) + '.zip', miniGC=miniGC,
                                        maxiGC=maxiGC, method=method, organism_name=organism_name, with_report=False,
                                        indexes=relevant_indexes)
            print('finished first optimization')
        ####################	
        ## extract suspected sites according to EFM criteria:	
        curr_sites_collector = suspect_site_extractor(example_seq, compute_methylation, num_sites,
                                                      methylation_sites_path, extension='_' + str(ii))
        df_recombination = curr_sites_collector['df_recombination_' + str(ii)]
        if len(df_recombination) > 0:
            df_recombination.loc[:, 'sequence_number'] = str(ii)
        recombination_collector.append(df_recombination)
        df_slippage = curr_sites_collector['df_slippage_' + str(ii)]
        if len(df_slippage) > 0:
            df_slippage.loc[:, 'sequence_number'] = str(ii)
        slippage_collector.append(df_slippage)
        if compute_methylation == True:
            df_methylation = curr_sites_collector['df_methylation_' + str(ii)]
            if len(df_methylation) > 0:
                df_methylation.loc[:, 'sequence_number'] = str(ii)
            methylation_collector.append(df_methylation)
        else:
            df_methylation = pd.DataFrame()
        ## SECOND OPTIMIZATION - optimize codon usage and GC content, WITH EFM constraints.	
        if optimize:
            _ = EFM_optimizer(example_seq, window_size_GC=None, df_recombination=df_recombination,
                              df_slippage=df_slippage, df_methylation=df_methylation, miniGC=miniGC,
                              maxiGC=maxiGC, method=method, organism_name=organism_name,
                              curr_output_path=curr_output_path, filename='Optimization_report_' + str(ii) + '.zip',
                              with_report=True, indexes=relevant_indexes)
            print('finished final optimization')
        ###########	
    # combine the dfs of all the sequences together to a single csv. file.	
    df_recombination = pd.concat(recombination_collector, ignore_index=True)
    if len(df_recombination) == 0:
        df_recombination = pd.DataFrame()
    df_slippage = pd.concat(slippage_collector, ignore_index=True)
    df_recombination.to_csv(join(curr_output_path, r'recombination_sites.csv'), index=False)
    df_slippage.to_csv(join(curr_output_path, r'slippage_sites.csv'), index=False)
    if compute_methylation == True:
        df_methylation = pd.concat(methylation_collector, ignore_index=True)
        df_methylation.to_csv(join(curr_output_path, r'methylation_sites.csv'), index=False)
    print('finished saving results')
    return


def advanced_data_handler(data, file, input_folder):
    list_sequences = []
    new_path = file.split(input_folder + sep)[1].split('.fasta')[0]
    for ii, record in enumerate(data):
        seq_id = str(ii)
        len_seq = len(record)
        list_sequences.append([new_path, seq_id, len_seq])
    return list_sequences


def advenced_function(input_folder=getcwd()):
    """	
    The function gets as input a directory. It returns a list of lists,	
    where each list is of the form (filename, seq_id, seq_length).	
    This is used in order to be able to select subsequences to optimize.	
    Args:	
        input_folder(str): path from which to read fasta and fasta.gz files. Default value is current path.	
    Returns:	
        sequence_ids(list of lists). The convention:	
        sequence_ids = [ [file name, seq_index, seq_length], [file name, seq_index, seq_length], ...]	
    """
    sequences_ids = []
    files_fas_1 = glob(join(input_folder, '*', '*.fasta'), recursive=True)
    files_fas = glob(join(input_folder, '*.fasta'), recursive=True)
    files_fasgz_1 = glob(join(input_folder, '*', '*.fasta.gz'), recursive=True)
    files_fasgz = glob(join(input_folder, '*.fasta.gz'), recursive=True)
    files_fas.extend(files_fas_1)
    files_fasgz.extend(files_fasgz_1)
    for file in files_fas:
        with open(file, "rU") as handle:
            data = list(SeqIO.parse(handle, "fasta"))
        curr_sequences = advanced_data_handler(data, file, input_folder)
        sequences_ids.extend(curr_sequences)
    for file in files_fasgz:
        with gzip.open(file, "rt") as handle:
            data = list(SeqIO.parse(handle, "fasta"))
        curr_sequences = advanced_data_handler(data, file, input_folder)
        sequences_ids.extend(curr_sequences)
    return sequences_ids


def test_input(miniGC, maxiGC, indexes):
    if miniGC < 0.0:
        return 'The minimum GC content must be over 0!'
    if maxiGC > 1.0:
        return 'The maximum GC content must be less than 1!'
    if miniGC >= maxiGC:
        return 'The minimum GC content must be less than the maximum!'
    if indexes != None:
        index_values = list(indexes.values())
        for index_pair in index_values:
            if (type(index_pair[0]) != int) or (type(index_pair[1]) != int):
                return 'Indexes must be integers!'
            if index_pair[0] < 1:
                return 'Start index must be greater than 0!'
            if index_pair[0] >= index_pair[1]:
                return 'Start index must be smaller than end index!'
            if (index_pair[1] - index_pair[0] + 1) % 3 != 0:
                return 'Indexes must describe sequence length divisible by 3!'
    return 'Success!'


def main(input_folder=getcwd(), output_path=join(getcwd(), 'output'), compute_methylation=False, num_sites=np.inf,
         methylation_sites_path=join(getcwd(), r'topEnriched.313.meme.txt'), test=False, optimize=False, miniGC=0.3,
         maxiGC=0.5, method='use_best_codon', organism_name='not_specified', indexes=None):
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
        optimize(bool): whether to use karin's optimization mechanism. Default is False.	
        miniGC(float): minimal GC fraction in optimized sequence. Default value is 0.3	
        maxiGC(float): maximal GC fraction in optimized sequence. Default value is 0.5	
        method(string): which optimization method to use. Default is 'use_best_codon'. Possible values are: 'use_best_codon',	
        'match_codon_usage', 'harmonize_rca'.	
        organism_name(string): the name of the host of the gene, used for the optimization algorithms. Default value is 'not_specified'.	
        possible values are {'b_subtilis', 'c_elegans', 'd_melanogaster', 'e_coli',	
        'g_gallus', 'h_sapiens', 'm_musculus', 'm_musculus_domesticus', 's_cerevisiae','not_specified}.	
        If the organism is 'not_specified' then the codon optimization objective will not be defined.	
        indexes(dict): a dictionary of tuples (filename,seq_index), where each value is a list containing the	
        sequence start and stop indexes of the ORF. It is defined within the advanced page in GUI, and defaults to None if	
        the user does not specify an input.	
        the convention for indexes is:	
            `{	
            (filename, seq_index(str)): [start(int),stop(int)] ,	
            (filename, seq_index(str)): [start(int),stop(int)] , ...	
            }`	
    Returns:	
        No variable. Saves output csv's in output_path.	
    """
    print('_____________________________________________________________________')
    if type(num_sites) == str:
        num_sites = np.inf
    message = test_input(miniGC, maxiGC, indexes)
    if message != 'Success!':
        print('illegal inputs')
        return message
    print('legal inputs')
    files_fas_1 = glob(join(input_folder, '*', '*.fasta'), recursive=True)
    files_fas = glob(join(input_folder, '*.fasta'), recursive=True)
    files_fasgz_1 = glob(join(input_folder, '*', '*.fasta.gz'), recursive=True)
    files_fasgz = glob(join(input_folder, '*.fasta.gz'), recursive=True)
    files_fas.extend(files_fas_1)
    files_fasgz.extend(files_fasgz_1)
    if test == True:
        files_fas = files_fas[0:2]
        files_fasgz = files_fasgz[0:2]
    for file in files_fas:
        with open(file, "rU") as handle:
            data = list(SeqIO.parse(handle, "fasta"))
        data_handler(data, file, output_path, compute_methylation, num_sites, methylation_sites_path, input_folder,
                     optimize=optimize,
                     miniGC=miniGC, maxiGC=maxiGC, method=method, organism_name=organism_name, indexes=indexes)
    for file in files_fasgz:
        with gzip.open(file, "rt") as handle:
            data = list(SeqIO.parse(handle, "fasta"))
        data_handler(data, file, output_path, compute_methylation, num_sites, methylation_sites_path, input_folder,
                     optimize=optimize,
                     miniGC=miniGC, maxiGC=maxiGC, method=method, organism_name=organism_name, indexes=indexes)
    return message