#!/usr/bin/env python3
"""Convert the SNAP signed networks to the linear-bias Ising .txt format of datasets/ (see the README).

On the SNAP files, it gives the Epinions, Slashdot and WikiElec instances of the paper.

Inputs (https://snap.stanford.edu/data/):
  soc-sign-epinions.txt.gz, soc-sign-Slashdot090221.txt.gz: directed arcs "from to sign" (sign +1/-1), node ids 0-based;
  wikiElec.ElecBs3.txt.gz: elections; each vote "V <vote> <voter> ..." is an arc voter -> candidate (U line) with
  vote +1 (support) or -1 (oppose), user ids 1-based; neutral votes (0) are skipped.

Rule:
  - self-loops are dropped;
  - the arcs of an unordered pair {i, j} (both directions, every occurrence) are summed, and the edge gets
    J_ij = sign(sum): +1 or -1; a pair whose sum is 0 (as many +1 as -1 arcs) gets no edge;
  - node k of a 0-based file becomes row k + 1; ids of a 1-based file are kept; n is the largest row;
  - fields h_i are 0; edges are written once, smaller row first, in the order each pair first appears in the input.

Output format: "n m", then n lines h_i, then m lines "i j J_ij" (1-indexed).

Usage: python snap_signed_to_txt.py {edges|wikielec} <input.gz> <output.txt>
"""
import gzip
import sys


def arcs_from_edges(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            a, b, s = line.split()
            yield int(a), int(b), int(s)


def arcs_from_wikielec(path):
    candidate = None
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] == "U":
                candidate = int(p[1])
            elif p[0] == "V" and int(p[1]) != 0:     # neutral votes are skipped (they would add 0)
                yield int(p[2]), candidate, int(p[1])


def convert(arcs, out_path):
    arcs = list(arcs)
    shift = 1 if min(min(a, b) for a, b, _ in arcs) == 0 else 0
    n = max(max(a, b) for a, b, _ in arcs) + shift
    total = {}                                   # insertion order = first appearance of the pair
    for a, b, s in arcs:
        if a == b:
            continue
        key = (min(a, b) + shift, max(a, b) + shift)
        total[key] = total.get(key, 0) + s
    edges = [(i, j, 1 if v > 0 else -1) for (i, j), v in total.items() if v != 0]
    with open(out_path, "w") as f:
        f.write(f"{n} {len(edges)}\n")
        f.write("0\n" * n)
        for i, j, w in edges:
            f.write(f"{i} {j} {w}\n")
    return n, len(edges)


if __name__ == "__main__":
    kind, src, dst = sys.argv[1:4]
    arcs = arcs_from_wikielec(src) if kind == "wikielec" else arcs_from_edges(src)
    print(convert(arcs, dst))
