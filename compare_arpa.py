#!/usr/bin/env python3

# Copyright 2018  Johns Hopkins University (Author: Ruizhe Huang)

import sys
import math

if __name__ == "__main__":
    # This script compares whether two arpa files are the same
    arpa1_filename = sys.argv[1]
    arpa2_filename = sys.argv[2]

    threshold = 0.001

    arpa1 = dict()
    with open(arpa1_filename) as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                continue

            if line.startswith("\\") or line.startswith("n"):
                continue

            fields = line.split("\t")

            word = fields[1]
            prob = float(fields[0])
            if len(fields) > 2:
                bow = float(fields[2])
            else:
                bow = 0
            arpa1[word] = (prob, bow)

    warning_count = 0
    line_count = 0

    with open(arpa1_filename) as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                continue

            if line.startswith("\\") or line.startswith("n"):
                continue

            line_count += 1
            fields = line.split("\t")

            word = fields[1]
            prob = float(fields[0])
            if len(fields) > 2:
                bow = float(fields[2])
            else:
                bow = 0

            prob_diff = abs((prob - arpa1[word][0]) / prob)
            if prob_diff >= threshold:
                print("warning [prob_diff=" + str(prob_diff) + "]: " + line)

            if bow == 0:
                bow_diff = abs((bow - arpa1[word][1]))
            else:
                bow_diff = abs((bow - arpa1[word][1]) / bow)

            if bow_diff >= threshold:
                warning_count += 1
                print("warning [bow_diff=" + str(bow_diff) + "]: " + line)

    print("finish: line_count=%d, warning_count=%d" % (line_count, warning_count))
