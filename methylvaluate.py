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
import pandas
import math
from scipy.stats import pearsonr
from enum import Enum
from datetime import datetime


FWD_STRAND = "+"
REV_STRAND = "-"


class MethylBedType(Enum):
    UCSC_GENOME_BROWSER_BISULFITE = "UCSC_GENOME_BROWSER_BISULFITE"
    STRANDED_SINGLE_BP = "STRANDED_SINGLE_BP"
    UNSTRANDED_DOUBLE_BP = "UNSTRANDED_DOUBLE_BP"
    @staticmethod
    def from_string(id):
        for mbt in MethylBedType:
            if str(id).upper() == mbt.value:
                return mbt
        return None


class ComparisonType(Enum):
    RECORD_OVERLAP = "RECORD_OVERLAP"
    BOOLEAN_CLASSIFICATION_BY_THRESHOLDS = "BOOLEAN_CLASSIFICATION_BY_THRESHOLDS"
    @staticmethod
    def from_string(id):
        for mbt in ComparisonType:
            if str(id).upper() == mbt.value:
                return mbt
        return None


class MethylLocus:
    def __init__(self, line, type):
        self.chr = None
        self.start_pos = None
        self.end_pos = None
        self.strand = None
        self.coverage = None
        self.methyl_count = None
        self.methyl_ratio = None
        self.type = type

        if type == MethylBedType.UCSC_GENOME_BROWSER_BISULFITE:
            parts = line.split()
            if len(parts) < 11: raise Exception("Badly formatted {} record: {}".format(type, line))
            self.chr = parts[0]
            self.start_pos = int(parts[1])
            self.end_pos = int(parts[2])
            self.strand = parts[5]
            self.coverage = int(parts[9])
            self.methyl_ratio = int(parts[10]) / 100.0
            self.methyl_count = round(self.coverage * self.methyl_ratio)
        elif type == MethylBedType.STRANDED_SINGLE_BP:
            parts = line.split()
            if len(parts) < 6: raise Exception("Badly formatted {} record: {}".format(type, line))
            self.chr = parts[0]
            self.start_pos = int(parts[1])
            self.end_pos = int(parts[2])
            self.strand = parts[3]
            self.methyl_count = int(parts[4])
            self.coverage = int(parts[5])
            self.methyl_ratio = 1.0 * self.methyl_count / self.coverage
        elif type == MethylBedType.UNSTRANDED_DOUBLE_BP:
            parts = line.split()
            if len(parts) < 5: raise Exception("Badly formatted {} record: {}".format(type, line))
            self.chr = parts[0]
            self.start_pos = int(parts[1])
            self.end_pos = int(parts[2])
            self.coverage = int(parts[3])
            self.methyl_ratio = float(parts[4])
            self.methyl_count = round(self.coverage * self.methyl_ratio)
        else:
            raise Exception("Unknown MethylBedType: {}".format(type))

        # sanity
        assert self.start_pos <= self.end_pos
        assert self.strand in (FWD_STRAND, REV_STRAND, None)

    def __str__(self):
        return "MethylRecord({}:{}{} {}/{})".format(self.chr, self.start_pos,
            "-{}".format(self.end_pos) if self.strand is None else " {}".format(self.strand),
            self.methyl_count, self.coverage)

    def __lt__(self, other):
        if self.start_pos == other.start_pos:
            return self.strand == FWD_STRAND and other.strand == REV_STRAND
        return self.start_pos < other.start_pos

    def __eq__(self, other):
        return self.start_pos == other.start_pos and self.strand == other.strand


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
                        default=MethylBedType.STRANDED_SINGLE_BP, type=MethylBedType.from_string,
                        help='Query methylation BED file format (default: STRANDED_SINGLE_BP, possible values: {})'.format([x.value for x in MethylBedType]))

    # how to classify
    parser.add_argument('--comparison_type', '-y', dest='comparison_type', required=False,
                        default=ComparisonType.RECORD_OVERLAP, type=ComparisonType.from_string,
                        help='Truth methylation BED file format (default: RECORD_OVERLAP, possible values: {})'.format([x.value for x in ComparisonType]))
    parser.add_argument('--truth_boolean_methyl_threshold', '-P', dest='truth_boolean_methyl_threshold', required=False, default=.9, type=float,
                       help='Threshold used to quantify boolean methylation state in truth')
    parser.add_argument('--query_boolean_methyl_threshold', '-p', dest='query_boolean_methyl_threshold', required=False, default=.9, type=float,
                       help='Threshold used to quantify boolean methylation state in query')

    return parser.parse_args()


def log(msg, log_time=True):
    print("{}{}".format("" if not log_time else "[{}] ".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        msg), file=sys.stderr)


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


def main():
    args = parse_args()

    # sanity check
    if (args.truth_format == MethylBedType.UNSTRANDED_DOUBLE_BP) != (args.query_format == MethylBedType.UNSTRANDED_DOUBLE_BP):
        raise Exception("Cannot compare formats: truth {}, query {}".format(args.truth_format.value, args.query_format.value))

    # inital logging
    if args.comparison_type == ComparisonType.RECORD_OVERLAP:
        log("Comparing using {}".format(ComparisonType.RECORD_OVERLAP.value))
    elif args.comparison_type == ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS:
        log("Comparing using {} with truth threshold {} and query threshold {}".format(
            ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS, args.truth_boolean_methyl_threshold,
            args.query_boolean_methyl_threshold))
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
                if args.comparison_type == ComparisonType.RECORD_OVERLAP:
                    tp_records.append((curr_truth, curr_query))
                elif args.comparison_type == ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS:
                    truth_is_methyl = curr_truth.methyl_ratio >= args.truth_boolean_methyl_threshold
                    query_is_methyl = curr_query.methyl_ratio >= args.query_boolean_methyl_threshold
                    if truth_is_methyl and query_is_methyl:
                        tp_records.append((curr_truth, curr_query))
                    elif truth_is_methyl:
                        fn_records.append((curr_truth, curr_query))
                    elif query_is_methyl:
                        fp_records.append((curr_truth, curr_query))
                    else:
                        tn_records.append((curr_truth, curr_query))
                curr_truth = next(truth_iter, None)
                curr_query = next(query_iter, None)

            # only a truth record at this locus (false negative)
            elif curr_truth < curr_query:
                if args.comparison_type == ComparisonType.RECORD_OVERLAP:
                    fn_records.append((curr_truth, None))
                elif args.comparison_type == ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS:
                    truth_is_methyl = curr_truth.methyl_ratio >= args.truth_boolean_methyl_threshold
                    if truth_is_methyl:
                        fn_records.append((curr_truth, None))
                    else:
                        tn_records.append((curr_truth, None))
                curr_truth = next(truth_iter, None)

            # only a query record at this locus (false positive)
            elif curr_query < curr_truth:
                if args.comparison_type == ComparisonType.RECORD_OVERLAP:
                    fp_records.append((None, curr_query))
                elif args.comparison_type == ComparisonType.BOOLEAN_CLASSIFICATION_BY_THRESHOLDS:
                    query_is_methyl = curr_query.methyl_ratio >= args.query_boolean_methyl_threshold
                    if query_is_methyl:
                        fp_records.append((None, curr_query))
                    else:
                        tn_records.append((None, curr_query))
                curr_query = next(query_iter, None)

            # should not happen
            else:
                raise Exception("Programmer Error: encountered unexpected edge case analyzing "
                                "truth '{}' and query '{}'".format(curr_truth, curr_query))

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

    log("Fin.")


if __name__ == "__main__":
    main()