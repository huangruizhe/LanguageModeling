#!/usr/bin/env python3

# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2018  Johns Hopkins University (Author: Ruizhe Huang)
# Apache 2.0.

import sys
import math
import random
import argparse
from collections import Counter, defaultdict


parser = argparse.ArgumentParser(description="")
parser.add_argument("--ngram-order", type=int, default=4, choices=[2, 3, 4, 5, 6, 7], help="Order of n-gram")
parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help="Verbose level")
args = parser.parse_args()


class CountsForHistory:
    ## This class (which is more like a struct) stores the counts seen in a
    ## particular history-state.  It is used inside class NgramCounts.
    ## It really does the job of a dict from int to float, but it also
    ## keeps track of the total count.
    def __init__(self):
        # The 'lambda: defaultdict(float)' is an anonymous function taking no
        # arguments that returns a new defaultdict(float).
        self.word_to_count = defaultdict(int)
        self.word_to_context = defaultdict(set)
        self.total_count = 0

    def words(self):
        return self.word_to_count.keys()

    def __str__(self):
        # e.g. returns ' total=12: 3->4, 4->6, -1->2'
        return ' total={0}: {1}'.format(
            str(self.total_count),
            ', '.join(['{0} -> {1}'.format(word, count)
                      for word, count in self.word_to_count.items()]))

    def add_count(self, predicted_word, context_word, count):
        assert count >= 0

        self.total_count += count
        self.word_to_count[predicted_word] += count
        if context_word is not None:
            self.word_to_context[predicted_word].add(context_word)


class NgramCounts:
    ## A note on data-structure.  Firstly, all words are represented as
    ## integers.  We store n-gram counts as an array, indexed by (history-length
    ## == n-gram order minus one) (note: python calls arrays "lists") of dicts
    ## from histories to counts, where histories are arrays of integers and
    ## "counts" are dicts from integer to float.  For instance, when
    ## accumulating the 4-gram count for the '8' in the sequence '5 6 7 8', we'd
    ## do as follows: self.counts[3][[5,6,7]][8] += 1.0 where the [3] indexes an
    ## array, the [[5,6,7]] indexes a dict, and the [8] indexes a dict.
    def __init__(self, ngram_order, bos_symbol='<s>', eos_symbol='</s>'):
        assert ngram_order >= 2

        self.ngram_order = ngram_order
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        self.backoff_symbol = -1
        # self.total_num_words = 0  # count includes EOS but not BOS.

        self.counts = []
        for n in range(ngram_order):
            self.counts.append(defaultdict(lambda: CountsForHistory()))

    # adds a raw count (called while processing input data).
    # Suppose we see the sequence '6 7 8 9' and ngram_order=4, 'history'
    # would be (6,7,8) and 'predicted_word' would be 9; 'count' would be
    # 1.
    def add_count(self, history, predicted_word, context_word, count):
        self.counts[len(history)][history].add_count(predicted_word, context_word, count)

    # 'line' is a string containing a sequence of integer word-ids.
    # This function adds the un-smoothed counts from this line of text.
    def add_raw_counts_from_line(self, line):
        words = [self.bos_symbol] + line.split() + [self.eos_symbol]

        for i in range(len(words)):
            for n in range(1, self.ngram_order+1):
                if i + n > len(words):
                    break

                ngram = words[i: i + n]
                predicted_word = ngram[-1]
                history = tuple(ngram[: -1])
                if i == 0 or n == self.ngram_order:
                    context_word = None
                else:
                    context_word = words[i-1]

                self.add_count(history, predicted_word, context_word, 1)

    def add_raw_counts_from_standard_input(self):
        lines_processed = 0
        while True:
            line = sys.stdin.readline()
            line = line.strip()
            if line == '':
                break
            self.add_raw_counts_from_line(line)
            lines_processed += 1
        if lines_processed == 0 or args.verbose > 0:
            print("make_phone_lm.py: processed {0} lines of input".format(lines_processed), file=sys.stderr)

    def add_raw_counts_from_file(self, filename):
        lines_processed = 0
        with open(filename) as fp:
            for line in fp:
                line = line.strip()
                if line == '':
                    break
                self.add_raw_counts_from_line(line)
                lines_processed += 1
        if lines_processed == 0 or args.verbose > 0:
            print("make_phone_lm.py: processed {0} lines of input".format(lines_processed), file=sys.stderr)

    def print_raw_counts(self, info_string):
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    res.append("{0}\t{1}".format(ngram.strip(), counts_for_hist.word_to_count[w]))
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_modified_counts(self, info_string):
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    modified_count = len(counts_for_hist.word_to_context[w])
                    raw_count = counts_for_hist.word_to_count[w]
                    if modified_count == 0:
                        res.append("{0}\t{1}".format(ngram.strip(), raw_count))
                    else:
                        res.append("{0}\t{1}".format(ngram.strip(), modified_count))
        res.sort(reverse=True)
        for r in res:
            print(r)


if __name__ == "__main__":

    ngram_counts = NgramCounts(args.ngram_order)
    # ngram_counts.add_raw_counts_from_standard_input()
    ngram_counts.add_raw_counts_from_file("data/c5.txt")
    ngram_counts.print_raw_counts("Raw counts:")
    ngram_counts.print_modified_counts("Modified counts:")
