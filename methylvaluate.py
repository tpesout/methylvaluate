#!/usr/bin/env python3
from __future__ import print_function
import argparse
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pysam
import collections
import numpy as np
import os
import seaborn as sns
import pandas as pd
import math
from scipy.stats import pearsonr
from enum import Enum
from datetime import datetime


FWD_STRAND = "+"
REV_STRAND = "-"

METHYL_Q = "methyl_query"
METHYL_T = "methyl_truth"
DEPTH_Q = "depth_query"
DEPTH_T = "depth_truth"


class MethylBedType(Enum):
    UCSC_GENOME_BROWSER_BISULFITE = "UCSC_GENOME_BROWSER_BISULFITE"
    STRANDED_RATIO_SINGLE_BP = "STRANDED_RATIO_SINGLE_BP"
    UNSTRANDED_RATIO_DOUBLE_BP = "UNSTRANDED_RATIO_DOUBLE_BP"
    PEPPER_OUTPUT = "PEPPER_OUTPUT"
    SIMPLE_BED = "SIMPLE_BED"
    BISMARK = "BISMARK"
    @staticmethod
    def from_string(id):
        for mbt in MethylBedType:
            if str(id).upper() == mbt.value:
                return mbt
        return None
    @staticmethod
    def has_confidence_value(id):
        if not type(id) == MethylBedType:
            id = MethylBedType.from_string(id)
        return id in [MethylBedType.PEPPER_OUTPUT]


class ComparisonType(Enum):
    RECORD_OVERLAP = "RECORD_OVERLAP"
    BOOLEAN_CLASSIFICATION_BY_THRESHOLDS = "BOOLEAN_CLASSIFICATION_BY_THRESHOLDS"
    BUCKETED_METHLYATION_LEVEL = "BUCKETED_METHLYATION_LEVEL"
    METHYLATION_LEVEL_DIFFERENCE = "METHYLATION_LEVEL_DIFFERENCE"
    @staticmethod
    def from_string(id):
        for mbt in ComparisonType:
            if str(id).upper() == mbt.value:
                return mbt
        return None


class RecordClassification(Enum):
    TP="TP"
    FP="FP"
    FN="FN"
    TN="TN"


class MethylLocus:
    def __init__(self, line, bed_type):
        self.chr = None
        self.start_pos = None
        self.end_pos = None
        self.strand = None
        self.coverage = None
        self.methyl_count = None
        self.methyl_ratio = None
        self.call_confidence = None
        self.record_classification = None
        self.paired_record = None
        self.type = bed_type

        if bed_type == MethylBedType.UCSC_GENOME_BROWSER_BISULFITE:
            parts = line.split()
            if len(parts) < 11: raise Exception("Badly formatted {} record: {}".format(bed_type, line))
            self.chr = parts[0]
            self.start_pos = int(parts[1])
            self.end_pos = int(parts[2])
            self.strand = parts[5]
            self.coverage = int(parts[9])
            self.methyl_ratio = float(parts[10]) / 100.0
            self.methyl_count = round(self.coverage * self.methyl_ratio)
        elif bed_type == MethylBedType.STRANDED_RATIO_SINGLE_BP:
            parts = line.split()
            if len(parts) < 6: raise Exception("Badly formatted {} record: {}".format(bed_type, line))
            self.chr = parts[0]
            self.start_pos = int(parts[1])
            self.end_pos = int(parts[2])
            self.strand = parts[3]
            self.methyl_count = int(parts[4])
            self.coverage = int(parts[5])
            self.methyl_ratio = 1.0 * self.methyl_count / self.coverage
        elif bed_type == MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP:
            if type(line) == str:
                parts = line.split()
                if len(parts) < 5: raise Exception("Badly formatted {} record: {}".format(bed_type, line))
                self.chr = parts[0]
                self.start_pos = int(parts[1])
                self.end_pos = int(parts[2])
                self.coverage = int(parts[3])
                self.methyl_ratio = float(parts[4])
                self.methyl_count = round(self.coverage * self.methyl_ratio)
            elif type(line) == tuple:
                prev, curr = line
                if type(prev) != MethylLocus or type(curr) != MethylLocus or not prev < curr:
                    raise Exception("Unexpected input to {} constructor: {}".format(MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP, line))
                self.chr = prev.chr
                self.start_pos = prev.start_pos
                self.end_pos = curr.end_pos
                self.coverage = prev.coverage + curr.coverage
                self.methyl_count = prev.methyl_count + curr.methyl_count
                self.methyl_ratio = 1.0 * self.methyl_count / self.coverage
        elif bed_type == MethylBedType.PEPPER_OUTPUT:
            parts = line.split()
            if len(parts) <= 5: raise Exception("Badly formatted {} record: {}".format(bed_type, line))
            self.chr = parts[0]
            self.start_pos = int(parts[1])
            self.end_pos = int(parts[2])
            self.strand = parts[4]
            self.call_confidence = float(parts[5])
            self.methyl_ratio = 1.0
        elif bed_type == MethylBedType.SIMPLE_BED:
            parts = line.split()
            if len(parts) <= 4: raise Exception("Badly formatted {} record: {}".format(bed_type, line))
            self.chr = parts[0]
            self.start_pos = int(parts[1])
            self.end_pos = int(parts[2])
            self.strand = parts[4]
            self.methyl_ratio = 1.0
        elif bed_type == MethylBedType.BISMARK:
            parts = line.split()
            if len(parts) <= 4: raise Exception("Badly formatted {} record: {}".format(bed_type, line))
            self.chr = parts[0]
            self.start_pos = int(parts[1])
            self.end_pos = int(parts[2])
            self.methyl_count = int(parts[4])
            self.coverage = int(parts[5]) + self.methyl_count
            self.methyl_ratio = 0.0 if self.coverage == 0 else self.methyl_count / self.coverage
        else:
            raise Exception("Unknown MethylBedType: {}".format(bed_type))

        # sanity
        assert self.start_pos <= self.end_pos
        assert self.strand in (FWD_STRAND, REV_STRAND, None)

    def __str__(self):
        return "MethylRecord({}:{}{} {} {})".format(self.chr, self.start_pos,
            "-{}".format(self.end_pos) if self.strand is None else " {}".format(self.strand),
            self.methyl_ratio, "" if self.call_confidence is None else self.call_confidence)

    def __lt__(self, other):
        if self.chr != other.chr:
            return self.chr < other.chr
        if self.start_pos == other.start_pos:
            return self.strand == FWD_STRAND and other.strand == REV_STRAND
        return self.start_pos < other.start_pos

    def __eq__(self, other):
        return self.chr == other.chr and self.start_pos == other.start_pos and self.strand == other.strand



class BedRecord:
    def __init__(self, line):
        parts = line.split()
        if len(parts) < 3: raise Exception("Badly formatted BED record: {}".format(line))
        self.chr = parts[0]
        self.start_pos = int(parts[1])
        self.end_pos = int(parts[2])

    def __str__(self):
        return "BedRecord({}:{}-{})".format(self.chr, self.start_pos, self.end_pos)


def parse_args():
    parser = argparse.ArgumentParser("Produce stats and refactoring for methylation bed file based from bisulfite data")

    # input files and formats
    parser.add_argument('--truth', '-t', dest='truth', required=True, type=str,
                       help='Truth methylation BED file')
    parser.add_argument('--query', '-q', dest='query', required=True, type=str,
                       help='Query methylation BED file')
    parser.add_argument('--confidence_bed', '-c', dest='confidence_bed', required=False, default=None, type=str,
                       help='If set, results will be restricted to calls in this BED')
    parser.add_argument('--truth_format', '-T', dest='truth_format', required=False,
                        default=MethylBedType.UCSC_GENOME_BROWSER_BISULFITE, type=MethylBedType.from_string,
                        help='Truth methylation BED file format (default: UCSC_GENOME_BROWSER_BISULFITE, possible values: {})'.format([x.value for x in MethylBedType]))
    parser.add_argument('--query_format', '-Q', dest='query_format', required=False,
                        default=MethylBedType.STRANDED_RATIO_SINGLE_BP, type=MethylBedType.from_string,
                        help='Query methylation BED file format (default: STRANDED_SINGLE_BP, possible values: {})'.format([x.value for x in MethylBedType]))

    # how to classify
    parser.add_argument('--comparison_type', '-y', dest='comparison_type', required=False,
                        default=ComparisonType.BUCKETED_METHLYATION_LEVEL, type=ComparisonType.from_string,
                        help='Truth methylation BED file format (default: BUCKETED_METHLYATION_LEVEL, possible values: {})'.format([x.value for x in ComparisonType]))
    parser.add_argument('--truth_boolean_methyl_threshold', '-P', dest='truth_boolean_methyl_threshold', required=False, default=.9, type=float,
                       help='Threshold used to quantify boolean methylation state in truth')
    parser.add_argument('--query_boolean_methyl_threshold', '-p', dest='query_boolean_methyl_threshold', required=False, default=.9, type=float,
                       help='Threshold used to quantify boolean methylation state in query')
    parser.add_argument('--bucket_count', '-b', dest='bucket_count', required=False, default=5, type=int,
                       help='Number of buckets to use during BUCKETED_METHLYATION_LEVEL comparison, ie "3" results in buckets of 0-.33,.33-.66,.66-1.0')
    parser.add_argument('--acceptable_methyl_difference', '-d', dest='acceptable_methyl_difference', required=False, default=.1, type=float,
                       help='Difference in methyl ratio to count as TP, ie ".1" results in a TP for T:0.7,Q:0.65, FP for T:0.2,Q:0.6, FN for T:0.4,Q:0.1')
    parser.add_argument('--only_count_matched_sites', '-m', dest='only_count_matched_sites', required=False, action='store_true', default=False,
                        help="Only count matched sites during analyses (does not apply to RECORD_OVERLAP)")
    parser.add_argument('--min_depth', '-D', dest='min_depth', required=False, default=0, type=int,
                       help='Calls with depth below this value will not be considered')
    parser.add_argument('--normalize_type', '-n', dest='normalize_type', required=False, action='store_true', default=False,
                        help="Convert incoming records to {} type".format(MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP))
    # output
    parser.add_argument('--output_base', '-o', dest='output_base', required=False, type=str, default=None,
                        help="Base output filenames on this parameter.  If set, will write annotated BED files")
    parser.add_argument('--output_from_filename', '-O', dest='output_from_filename', required=False, action='store_true', default=False,
                        help="Base output filenames on input filenames.  If set, will write annotated BED files")
    parser.add_argument('--plot', '-l', dest='plot', required=False, action='store_true', default=False,
                        help="Produce plots")


    return parser.parse_args()


def log(msg, log_time=True):
    print("{}{}".format("" if not log_time else "[{}] ".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        msg), file=sys.stderr, flush=True)


def convert_to_unstranded_double_bp_methyl_locus(methyl_records_in):
    methyl_records_out = dict()
    unconvertable_records = 0
    for chr in methyl_records_in.keys():
        methyl_records_out[chr] = list()
        prev_methyl_record = None
        for methyl_record in methyl_records_in[chr]:
            # iterate if we don't have a prev to specify
            if prev_methyl_record is None:
                prev_methyl_record = methyl_record
                continue

            # determine if this is a pair
            if prev_methyl_record.start_pos == methyl_record.start_pos - 1 and prev_methyl_record.strand == FWD_STRAND and methyl_record.strand == REV_STRAND:
                new_methyl_record = MethylLocus((prev_methyl_record, methyl_record), MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP)
                methyl_records_out[chr].append(new_methyl_record)
                prev_methyl_record = None
            else:
                if unconvertable_records == 0:
                    log("Unable to convert records {} and {} to {} type".format(prev_methyl_record, methyl_record, MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP))
                unconvertable_records += 1
                prev_methyl_record = methyl_record
    log("Failed to convert {} ({}%) records to {} type".format(unconvertable_records,
        int(unconvertable_records / sum(list(map(len, methyl_records_in.values())))),
        MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP))
    return methyl_records_out



def filter_methyls_by_confidence(methyl_records, confidence_records, contigs_to_analyze, methyl_identifier):
    skipped_methyl_record_count = 0
    kept_methyl_record_count = 0
    for contig in contigs_to_analyze:
        kept_methyl_records = list()
        conf_iter = iter(confidence_records[contig])
        methyl_iter = iter(methyl_records[contig])
        curr_conf = next(conf_iter, None)
        curr_methyl = next(methyl_iter, None)
        while curr_conf is not None and curr_methyl is not None:
            if curr_conf.start_pos <= curr_methyl.start_pos and curr_methyl.end_pos <= curr_conf.end_pos:
                kept_methyl_records.append(curr_methyl)
                kept_methyl_record_count += 1
                curr_methyl = next(methyl_iter, None)
            elif curr_methyl.start_pos <= curr_conf.start_pos:
                skipped_methyl_record_count += 1
                curr_methyl = next(methyl_iter, None)
            elif curr_conf.end_pos <= curr_methyl.end_pos:
                curr_conf = next(conf_iter, None)
            else:
                raise Exception("Programmer Error: encountered edge case filtering "
                                "{} records for methyl '{}' and confidence '{}'".format(
                                    methyl_identifier, curr_methyl, curr_conf))
        methyl_records[contig] = kept_methyl_records
    log("Kept {} and discarded {} {} records based on confidence BED".format(
        kept_methyl_record_count, skipped_methyl_record_count, methyl_identifier))


def get_output_base(args, filename=None):
    if args.output_base is not None:
        return args.output_base
    if args.output_from_filename:
        return strip_suffixes(os.path.basename(filename if filename is not None else args.query), ".bed")
    return None


def strip_suffixes(string, suffixes):
    if type(suffixes) == str:
        suffixes = [suffixes]
    for suffix in suffixes:
        if not suffix.startswith("."):
            suffix = "." + suffix
        if string.endswith(suffix):
            string = string[:(-1 * len(suffix))]
    return string

def get_number_string(num):
    if num < 1000: return str(num)
    if num < 1000000: return str(int(num/1000))+"k"
    if num < 1000000000: return str(int(num/1000000))+"M"
    return str(int(num/1000000000))+"G"


def write_output_files(args, truth_records, query_records, contigs_to_analyze):
    # names
    truth_output_filename = "{}.TRUTH_RESULTS.bed".format(get_output_base(args, args.truth))
    query_output_filename = "{}.QUERY_RESULTS.bed".format(get_output_base(args, args.query))
    log("Writing truth output to {}".format(truth_output_filename))
    log("Writing query output to {}".format(query_output_filename))

    # open files
    truth_out = None
    query_out = None
    full_out = None
    try:
        truth_out = open(truth_output_filename, "w")
        query_out = open(query_output_filename, "w")
        # full_out = open(full_output_filename, "w")

        # write truth
        columns = ["chr", "start_pos", "end_pos", "strand", "classification", "methyl_ratio", "query_methyl_ratio"]
        truth_out.write("#" + "\t".join(columns) + "\n")
        for contig in contigs_to_analyze:
            for record in truth_records[contig]:
                out_record = [
                    record.chr,
                    record.start_pos,
                    record.end_pos,
                    record.strand,
                    "." if record.record_classification is None else record.record_classification.value,
                    record.methyl_ratio,
                    "." if record.paired_record is None else record.paired_record.methyl_ratio
                ]
                truth_out.write("\t".join(list(map(str, out_record))) + "\n")

        # write query
        columns = ["chr", "start_pos", "end_pos", "strand", "classification", "confidence", "methyl_ratio", "truth_methyl_ratio"]
        query_out.write("#" + "\t".join(columns) + "\n")
        for contig in contigs_to_analyze:
            for record in query_records[contig]:
                out_record = [
                    record.chr,
                    record.start_pos,
                    record.end_pos,
                    record.strand,
                    "." if record.record_classification is None else record.record_classification.value,
                    "." if record.call_confidence is None else record.call_confidence,
                    record.methyl_ratio,
                    "." if record.paired_record is None else record.paired_record.methyl_ratio
                ]
                query_out.write("\t".join(list(map(str, out_record))) + "\n")

    except:
        if truth_out is not None: truth_out.close()
        if query_out is not None: query_out.close()
        if full_out is not None: full_out.close()


def plot_roc(args, truth_records, query_records, contigs_to_analyze, bucket_count=32):
    # sanity check
    get_confidence = lambda x: x.call_confidence
    if not MethylBedType.has_confidence_value(args.query_format):
        log("Query format {} does not have confidence values".format(args.query_format))
        if args.query_format in [MethylBedType.STRANDED_RATIO_SINGLE_BP, MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP]:
            log("Using depth as proxy for confidence")
            get_confidence = lambda x: x.coverage
        else:
            return


    # get confidence range
    min_confidence = sys.maxsize
    max_confidence = 0
    for contig in contigs_to_analyze:
        for record in query_records[contig]:
            if get_confidence(record) is None:
                log("Query record {} had no confidence value, cannot produce ROC plot".format(record))
                return
            min_confidence = min(get_confidence(record), min_confidence)
            max_confidence = max(get_confidence(record), max_confidence)
    if max_confidence <= min_confidence:
        log("Got unusable confidence range {}-{}, cannot produce ROC plot".format(min_confidence, max_confidence))
        return

    # confidence buckets
    confidence_range = max_confidence - min_confidence
    def get_confidence_bucket(val):
        return int((val - min_confidence) / confidence_range * (bucket_count - 1)) #bc-1 means all max conf values have a single bucket
    classification_by_confidence = collections.defaultdict(lambda : [0, 0, 0, 0])
    def get_classification_idx(val):
        return {RecordClassification.TP:0,RecordClassification.FP:1,RecordClassification.FN:2,RecordClassification.TN:3}[val]
    idx_to_annotation = {
        0: min_confidence,
        int(bucket_count / 4): min_confidence + confidence_range / 4,
        int(bucket_count / 2): min_confidence + confidence_range / 2,
        int(bucket_count * 3 / 4): min_confidence + confidence_range * 3 / 4,
        bucket_count - 1: max_confidence,
    }

    # classify query records by confidence
    for contig in contigs_to_analyze:
        for record in query_records[contig]:
            bucket = get_confidence_bucket(get_confidence(record))
            if type(record.record_classification) is not RecordClassification:
                log("Unexpected record classification {} for {}".format(record.record_classification, record))
                continue
            idx = get_classification_idx(record.record_classification)
            classification_by_confidence[bucket][idx] += 1

    # get missing query calls (count all FNs from missing )
    truth_only_fn_count = 0
    for contig in contigs_to_analyze:
        for record in truth_records[contig]:
            if record.paired_record is None and record.record_classification == RecordClassification.FN:
                truth_only_fn_count += 1


    # get data for plotting
    def get_precision(tp, fp): return 0 if tp + fp == 0 else tp / (tp + fp)
    def get_recall(tp, fn): return 0 if tp + fn == 0 else tp / (tp + fn)
    at_value_precision = list()
    at_value_recall = list()
    at_or_above_value_precision = list()
    at_or_above_value_recall = list()
    total_tp = 0
    total_fp = 0
    total_fn = truth_only_fn_count
    for x in range(bucket_count):
        b = bucket_count - x - 1
        tp = classification_by_confidence[b][0]
        fp = classification_by_confidence[b][1]
        fn = classification_by_confidence[b][2]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        fn += truth_only_fn_count
        at_value_recall.append(get_recall(tp, fn))
        at_value_precision.append(get_precision(tp, fp))
        at_or_above_value_recall.append(get_recall(total_tp, total_fn))
        at_or_above_value_precision.append(get_precision(total_tp, total_fp))

    # plot it
    plt.plot(at_value_precision, at_value_recall)
    plt.scatter(at_value_precision, at_value_recall)
    for idx in idx_to_annotation.keys():
        annotation = idx_to_annotation[idx]
        idx = bucket_count - idx - 1
        plt.annotate(annotation, xy=(at_value_precision[idx], at_value_recall[idx]))
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("ROC \"At Value\"")
    plt.savefig("{}.ROC.at_value.png".format(get_output_base(args)))
    plt.show()
    plt.close()

    # plot it
    plt.plot(at_or_above_value_precision, at_or_above_value_recall)
    plt.scatter(at_or_above_value_precision, at_or_above_value_recall)
    for idx in idx_to_annotation.keys():
        annotation = idx_to_annotation[idx]
        idx = bucket_count - idx - 1
        plt.annotate(annotation, xy=(at_or_above_value_precision[idx], at_or_above_value_recall[idx]))
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("ROC \"At or Above Value\"")
    plt.savefig("{}.ROC.at_or_above_value.png".format(get_output_base(args)))
    plt.show()
    plt.close()


def plot_bucketed_heatmap(buckets, args, fig_size=(8,8)):
    # sanity check
    if len(buckets) == 0:
        log("No bucketed Truth/Query comparisons")
        return

    # printing text
    print()
    labels = ["{:.2f}-{:.2f}".format(x/args.bucket_count, (x+1)/args.bucket_count) for x in range(args.bucket_count)]
    max_size = max(max(list(map(len, labels))), max(list(map(lambda y: max([len(str(buckets[y][x])) for x in range(args.bucket_count)]), buckets))))
    element_format_str = " {:>" + str(max_size) + "s} "
    print(element_format_str.format("T=y, Q=x"), end="")
    for label in labels:
        print(element_format_str.format(label), end="")
    print()
    for i in range(args.bucket_count):
        print(element_format_str.format(labels[i]), end="")
        for m in range(args.bucket_count):
            print(element_format_str.format(str(buckets[i][m])), end="")
        print()
    print("", flush=True)


    # plot it
    use_log = True
    # use_log = False
    fig, ax = plt.subplots(figsize=fig_size)
    pair_maps = [[buckets[y][x] for x in range(args.bucket_count)] for y in range(args.bucket_count -1, -1, -1)]
    log_pair_maps = [[0 if buckets[y][x] == 0 else math.log10(buckets[y][x]) for x in range(args.bucket_count)] for y in range(args.bucket_count -1, -1, -1)]
    if use_log:
        im = ax.imshow(log_pair_maps)
    else:
        im = ax.imshow(pair_maps)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    if use_log:
        pass
        # cbar.ax.set_yticklabels([int(10 ** x) for x in cbar.ax.get_yticks()])
    else:
        cbar.ax.set_yticklabels([x for x in cbar.ax.get_yticks()])

    # label ticks
    ax.set_xticks([x for x in range(args.bucket_count)])
    ax.set_yticks([x for x in range(args.bucket_count)])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(reversed(labels))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # add labels
    for i in range(args.bucket_count):
        for j in range(args.bucket_count):
            text = ax.text(j, i, get_number_string(pair_maps[i][j]), ha="center", va="center", color="w", size=6)
            # text = ax.text(j, i, "{:.2f}".format(log_pair_maps[i][j]) if use_log else pair_maps[i][j], ha="center", va="center", color="w", size=8)

    plt.suptitle("Methyl Ratio Comparison", y=1)
    ax.set_ylabel("Truth")
    ax.set_xlabel("Query")
    fig.tight_layout()
    output_base = get_output_base(args)
    if output_base is not None:
        filename = "{}.bucketed_methyl_heatmap_{}.png".format(output_base, args.bucket_count)
        log("Saving heatmap to {}".format(filename))
        plt.savefig(filename)
    plt.show()
    plt.close()


def plot_heatmap(query, truth, args, fig_size=10):
    # get pearsonr
    r, p = pearsonr(query, truth)
    log("Pearson R correlation:\n\tR: {}\n\tP: {}".format(r, p))

    # plot joint and clear scatter
    ax1 = sns.jointplot(x=query, y=truth, height=fig_size, marginal_kws=dict(bins=args.bucket_count))
    ax1.ax_marg_x.text(1.05, 1, "Pearson's R: {:.3f}\n".format(r), ha="left", va="bottom", color="black", size=10)
    ax1.ax_joint.cla()
    plt.sca(ax1.ax_joint)

    # plot scatter + colobar
    plt.hist2d(query,truth,args.bucket_count,norm=colors.LogNorm(),cmap=sns.color_palette("viridis", as_cmap=True))
    cbar_ax = ax1.fig.add_axes([1, 0.1, .03, .7])
    cb = plt.colorbar(cax=cbar_ax)
    cb.set_label(r'$\log_{10}$ density',fontsize=13)

    # put colorbar in the right spot
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = ax1.ax_joint.get_position()
    pos_marg_x_ax = ax1.ax_marg_x.get_position()
    # reposition the joint ax so it has the same width as the marginal x ax
    ax1.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    # reposition the colorbar using new x positions and y positions of the joint ax
    ax1.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .04, pos_joint_ax.height])

    # labels
    plt.suptitle("Methyl Ratio Comparison", y=1)
    ax1.ax_joint.set_ylabel("Truth")
    ax1.ax_joint.set_xlabel("Query")

    # save
    output_base = get_output_base(args)
    if output_base is not None:
        filename = "{}.jointplot_heatmap_{}.png".format(output_base, args.bucket_count)
        log("Saving heatmap to {}".format(filename))
        plt.savefig(filename)
    plt.show()
    plt.close()



def plot_differential_methylation(differences, args):
    if len(differences) == 0:
        log("No compared differences")
        return

    print()
    print("Made {} comparisons".format(len(differences)))
    print("\tAvg difference:     {:.5f}".format(np.mean(differences)))
    print("\tAvg abs difference: {:.5f}".format(np.mean(list(map(abs,differences)))))
    print("", flush=True)

    difference_size = args.acceptable_methyl_difference
    differences_factors = collections.defaultdict(lambda: 0)

    for d in differences:
        idx = int(d / difference_size)
        differences_factors[idx] += 1

    xs = []
    ys = []
    x0 = min(differences_factors.keys())
    xn = max(differences_factors.keys())
    x = x0
    while x <= xn:
        xs.append(x)
        ys.append(differences_factors[x])
        x += 1

    plt.bar(xs, ys)

    plt.xticks(xs, ["{}x".format(x) for x in xs])

    plt.title("Methylation Difference Factors")
    plt.xlabel("Difference Factor\n(based on acceptable difference of {}%)".format(int(100 * difference_size)))
    plt.ylabel("Count")
    plt.tight_layout()
    output_base = get_output_base(args)
    if output_base is not None:
        filename = "{}.methyl_difference_factor_{}.png".format(output_base, int(100 * difference_size))
        log("Saving heatmap to {}".format(filename))
        plt.savefig(filename)
    plt.show()
    plt.close()


def main():
    args = parse_args()

    # sanity check
    if (args.truth_format == MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP) != (args.query_format == MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP):
        raise Exception("Cannot compare formats: truth {}, query {}".format(args.truth_format.value, args.query_format.value))

    # inital logging
    if args.comparison_type == ComparisonType.RECORD_OVERLAP:
        log("Comparing using {}".format(ComparisonType.RECORD_OVERLAP.value))
    elif args.comparison_type == ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS:
        log("Comparing using {} with truth threshold {} and query threshold {}".format(
            ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS, args.truth_boolean_methyl_threshold,
            args.query_boolean_methyl_threshold))
    elif args.comparison_type == ComparisonType.BUCKETED_METHLYATION_LEVEL:
        log ("Comparing using {} with bucket count {}".format(ComparisonType.BUCKETED_METHLYATION_LEVEL,
                                                              args.bucket_count))
        assert(args.bucket_count > 0)
    elif args.comparison_type == ComparisonType.METHYLATION_LEVEL_DIFFERENCE:
        log ("Comparing using {} with bucket count {}".format(ComparisonType.METHYLATION_LEVEL_DIFFERENCE,
                                                              args.acceptable_methyl_difference))
        assert(args.acceptable_methyl_difference > 0.0 and args.acceptable_methyl_difference < 1.0)
    else:
        raise Exception("Unhandled comparison type: {}".format(args.comparison_type))

    # data we want
    truth_records = collections.defaultdict(lambda: list())
    query_records = collections.defaultdict(lambda: list())

    # read truth file
    log("Reading truth records from {} with format {}".format(args.truth, args.truth_format.value))
    with open(args.truth) as fin:
        for linenr, line in enumerate(fin):
            try:
                if line.startswith("#") or len(line.strip()) == 0: continue
                ml = MethylLocus(line, args.truth_format)
                truth_records[ml.chr].append(ml)
            except Exception as e:
                log("Exception at line {}: {}".format(linenr, line))
                raise e
    for methyl_list in truth_records.values():
        methyl_list.sort(key=lambda x: x.start_pos)
    record_count = sum(list(map(len, truth_records.values())))
    log("Got {} truth methyl records over {} contigs".format(record_count, len(truth_records)))
    if args.normalize_type and args.truth_format != MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP:
        truth_records = convert_to_unstranded_double_bp_methyl_locus(truth_records)

    # read query file
    log("Reading query records from {} with format {}".format(args.query, args.query_format.value))
    with open(args.query) as fin:
        for linenr, line in enumerate(fin):
            try:
                if line.startswith("#") or len(line.strip()) == 0: continue
                ml = MethylLocus(line, args.query_format)
                query_records[ml.chr].append(ml)
            except Exception as e:
                log("Exception at line {}: {}".format(linenr, line))
                raise e
    for methyl_list in query_records.values():
        methyl_list.sort(key=lambda x: x.start_pos)
    record_count = sum(list(map(len, query_records.values())))
    log("Got {} query methyl records over {} contigs".format(record_count, len(query_records)))
    if args.normalize_type and args.query_format != MethylBedType.UNSTRANDED_RATIO_DOUBLE_BP:
        query_records = convert_to_unstranded_double_bp_methyl_locus(query_records)

    #TODO convert single base loci to combined multi-base locus

    # get intersection and loggit
    contigs_to_analyze = list(set(truth_records.keys()).intersection(set(query_records.keys())))
    log("Found {} shared contigs between truth and query, excluding {} truth and {} query contigs".format(
        len(contigs_to_analyze), len(list(filter(lambda x: x not in contigs_to_analyze, list(truth_records.keys())))),
        len(list(filter(lambda x: x not in contigs_to_analyze, list(query_records.keys()))))))

    # handle high conf (if applicable)
    if args.confidence_bed is not None:
        # read file
        confidence_regions = collections.defaultdict(lambda: list())
        with open(args.confidence_bed) as fin:
            for linenr, line in enumerate(fin):
                try:
                    if line.startswith("#") or len(line.strip()) == 0: continue
                    br = BedRecord(line)
                    confidence_regions[br.chr].append(br)
                except Exception as e:
                    log("Exception at line {}: {}".format(linenr, line))
                    raise e
        for methyl_list in query_records.values():
            methyl_list.sort(key=lambda x: x.start_pos)
        record_count = sum(list(map(len, confidence_regions.values())))
        log("Got {} confidence BED records over {} contigs".format(record_count, len(confidence_regions)))

        # update contigs to analyze
        conf_contig_intersect = set(contigs_to_analyze).intersection(set(confidence_regions.keys()))
        log("Found {} shared contigs in confidence bed, excluding {} from confidence and {} from truth/query".format(
            len(conf_contig_intersect),
            len(list(filter(lambda x: x not in conf_contig_intersect, list(confidence_regions.keys())))),
            len(list(filter(lambda x: x not in conf_contig_intersect, contigs_to_analyze)))))
        contigs_to_analyze = list(conf_contig_intersect)

        # update to only keep records in high conf
        filter_methyls_by_confidence(truth_records, confidence_regions, contigs_to_analyze, "truth")
        filter_methyls_by_confidence(query_records, confidence_regions, contigs_to_analyze, "query")
    else:
        log("No confidence BED specified")

    # prep for analysis
    tp_records = list()
    fp_records = list()
    fn_records = list()
    tn_records = list()
    contigs_to_analyze.sort()

    # prep for bucketed analysis
    bucket_counter = collections.defaultdict(lambda : collections.defaultdict( lambda : int(0)))
    get_methyl_bucket = lambda x: min(args.bucket_count - 1, int(x * args.bucket_count))
    sufficient_depth = lambda t,q: (t is None or t.coverage >= args.min_depth) and (q is None or q.coverage >= args.min_depth)
    insufficient_depth_count = 0

    # for seaborn plotting
    raw_matched_entries_methyl_query = list()
    raw_matched_entries_methyl_truth = list()

    # prep for differential analysis
    methyl_differences = list()

    # iterate over contigs
    for contig in contigs_to_analyze:
        # iterate over each pair, advancing one (or both) at each step
        truth_iter = iter(truth_records[contig])
        query_iter = iter(query_records[contig])
        curr_truth = next(truth_iter, None)
        curr_query = next(query_iter, None)
        while curr_truth is not None and curr_query is not None:

            # we have the same locus (true positive candidate)
            if curr_truth == curr_query:
                if not sufficient_depth(curr_truth, curr_query):
                    insufficient_depth_count += 1
                else:
                    raw_matched_entries_methyl_query.append(curr_query.methyl_ratio)
                    raw_matched_entries_methyl_truth.append(curr_truth.methyl_ratio)
                    if args.comparison_type == ComparisonType.RECORD_OVERLAP:
                        tp_records.append((curr_truth, curr_query))
                        curr_truth.record_classification = RecordClassification.TP
                        curr_query.record_classification = RecordClassification.TP
                    elif args.comparison_type == ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS:
                        truth_is_methyl = curr_truth.methyl_ratio >= args.truth_boolean_methyl_threshold
                        query_is_methyl = curr_query.methyl_ratio >= args.query_boolean_methyl_threshold
                        if truth_is_methyl and query_is_methyl:
                            tp_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.TP
                            curr_query.record_classification = RecordClassification.TP
                        elif truth_is_methyl:
                            fn_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.FN
                            curr_query.record_classification = RecordClassification.FN
                        elif query_is_methyl:
                            fp_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.FP
                            curr_query.record_classification = RecordClassification.FP
                        else:
                            tn_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.TN
                            curr_query.record_classification = RecordClassification.TN
                    elif args.comparison_type == ComparisonType.BUCKETED_METHLYATION_LEVEL:
                        truth_bucket = get_methyl_bucket(curr_truth.methyl_ratio)
                        query_bucket = get_methyl_bucket(curr_query.methyl_ratio)
                        bucket_counter[truth_bucket][query_bucket] += 1
                        if truth_bucket == query_bucket:
                            tp_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.TP
                            curr_query.record_classification = RecordClassification.TP
                        elif query_bucket > truth_bucket:
                            fp_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.FP
                            curr_query.record_classification = RecordClassification.FP
                        else:
                            fn_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.FN
                            curr_query.record_classification = RecordClassification.FN
                    elif args.comparison_type == ComparisonType.METHYLATION_LEVEL_DIFFERENCE:
                        methyl_difference = curr_truth.methyl_ratio - curr_query.methyl_ratio
                        methyl_differences.append(methyl_difference)
                        if abs(methyl_difference) <= args.acceptable_methyl_difference:
                            tp_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.TP
                            curr_query.record_classification = RecordClassification.TP
                        elif curr_truth.methyl_ratio < curr_query.methyl_ratio:
                            fp_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.FP
                            curr_query.record_classification = RecordClassification.FP
                        else:
                            fn_records.append((curr_truth, curr_query))
                            curr_truth.record_classification = RecordClassification.FN
                            curr_query.record_classification = RecordClassification.FN

                # iterate
                curr_truth.paired_record = curr_query
                curr_query.paired_record = curr_truth
                curr_truth = next(truth_iter, None)
                curr_query = next(query_iter, None)

            # only a truth record at this locus (false negative)
            elif curr_truth < curr_query:
                if not sufficient_depth(curr_truth, curr_query):
                    insufficient_depth_count += 1
                else:
                    if args.comparison_type == ComparisonType.RECORD_OVERLAP:
                        fn_records.append((curr_truth, None))
                        curr_truth.record_classification = RecordClassification.FN
                    elif args.only_count_matched_sites:
                        pass
                    elif args.comparison_type == ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS:
                        truth_is_methyl = curr_truth.methyl_ratio >= args.truth_boolean_methyl_threshold
                        if truth_is_methyl:
                            fn_records.append((curr_truth, None))
                            curr_truth.record_classification = RecordClassification.FN
                        else:
                            tn_records.append((curr_truth, None))
                            curr_truth.record_classification = RecordClassification.TN
                    elif args.comparison_type == ComparisonType.BUCKETED_METHLYATION_LEVEL:
                        fn_records.append((curr_truth, None))
                        curr_truth.record_classification = RecordClassification.FN
                    elif args.comparison_type == ComparisonType.METHYLATION_LEVEL_DIFFERENCE:
                        methyl_difference = curr_truth.methyl_ratio
                        methyl_differences.append(methyl_difference)
                        if methyl_difference <= args.acceptable_methyl_difference:
                            tp_records.append((curr_truth, None))
                            curr_truth.record_classification = RecordClassification.TP
                        else:
                            fn_records.append((curr_truth, None))
                            curr_truth.record_classification = RecordClassification.FN

                # iterate
                curr_truth = next(truth_iter, None)

            # only a query record at this locus (false positive)
            elif curr_query < curr_truth:
                if not sufficient_depth(curr_truth, curr_query):
                    insufficient_depth_count += 1
                else:
                    if args.comparison_type == ComparisonType.RECORD_OVERLAP:
                        fp_records.append((None, curr_query))
                        curr_query.record_classification = RecordClassification.FP
                    elif args.only_count_matched_sites:
                        pass
                    elif args.comparison_type == ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS:
                        query_is_methyl = curr_query.methyl_ratio >= args.query_boolean_methyl_threshold
                        if query_is_methyl:
                            fp_records.append((None, curr_query))
                            curr_query.record_classification = RecordClassification.FP
                        else:
                            tn_records.append((None, curr_query))
                            curr_query.record_classification = RecordClassification.TN
                    elif args.comparison_type == ComparisonType.BUCKETED_METHLYATION_LEVEL:
                        fp_records.append((None, curr_query))
                        curr_query.record_classification = RecordClassification.FP
                    elif args.comparison_type == ComparisonType.METHYLATION_LEVEL_DIFFERENCE:
                        methyl_difference = curr_query.methyl_ratio
                        methyl_differences.append( -1 * methyl_difference)
                        if methyl_difference <= args.acceptable_methyl_difference:
                            tp_records.append((None, curr_query))
                            curr_truth.record_classification = RecordClassification.TP
                        else:
                            fp_records.append((None, curr_query))
                            curr_truth.record_classification = RecordClassification.FP

                # iterate
                curr_query = next(query_iter, None)

            # should not happen
            else:
                raise Exception("Programmer Error: encountered unexpected edge case analyzing "
                                "truth '{}' and query '{}'".format(curr_truth, curr_query))

    log("Excluded {} sites because of insufficient depth".format(insufficient_depth_count))
    # actual stats
    tp = len(tp_records)
    fp = len(fp_records)
    fn = len(fn_records)
    precision = tp / max(1, tp+fp)
    recall = tp / max(1, tp+fn)
    f1 = 2 * (precision * recall) / (1 if precision + recall == 0 else precision + recall)
    log("", log_time=False)
    log("Accuracy Stats:", log_time=False)
    log("\tTP: {}".format(tp), log_time=False)
    log("\tFP: {}".format(fp), log_time=False)
    log("\tFN: {}".format(fn), log_time=False)
    log("\tPrecision: {}".format(precision), log_time=False)
    log("\tRecall:    {}".format(recall), log_time=False)
    log("\tF1:        {}".format(f1), log_time=False)
    log("", log_time=False)

    # output files
    if args.output_base is not None or args.output_from_filename:
        write_output_files(args, truth_records, query_records, contigs_to_analyze)

    # plot
    if args.plot:
        log("Plotting")
        plot_heatmap(raw_matched_entries_methyl_query, raw_matched_entries_methyl_truth, args)
        if args.comparison_type == ComparisonType.RECORD_OVERLAP or args.comparison_type == ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS:
            plot_roc(args, truth_records, query_records, contigs_to_analyze)
        if args.comparison_type == ComparisonType.BUCKETED_METHLYATION_LEVEL:
            plot_bucketed_heatmap(bucket_counter, args)
        if args.comparison_type == ComparisonType.METHYLATION_LEVEL_DIFFERENCE:
            plot_differential_methylation(methyl_differences, args)

    log("Fin.")


if __name__ == "__main__":
    main()
