import re
import sys
import numpy as np
from matplotlib import pyplot as plt

is_marvin = False

mutation_re = re.compile(r'Num features added: +(\d+).*')
generation_re = re.compile(r'Experiment successful: +(\d+)')
feature_re = re.compile(r'Feature types: *(\d+) (\d+) (\d+)')
failed_re = re.compile(r'failed')


def process(header):
    filename = "../output/logs/" + header + "master.log"

    with open(filename, "r") as f:
        log = f.readlines()
    mutations = []
    generations = []
    perm, static, dynamic = [], [], []
    num_failed = 0

    for line in log:
        mutation_match = mutation_re.search(line)
        if mutation_match is not None:
            mutations.append(int(mutation_match.group(1)))
        generation_match = generation_re.search(line)
        if generation_match is not None:
            generations.append(int(generation_match.group(1)))
        feature_match = feature_re.search(line)
        if is_marvin and feature_match is not None:
            perm.append(int(feature_match.group(1)))
            static.append(int(feature_match.group(2)))
            dynamic.append(int(feature_match.group(3)))
        failed_match = failed_re.search(line)
        if failed_match is not None:
            num_failed += 1

    print(len(mutations))
    print("Mutations", np.mean(mutations))
    print("Generations", np.mean(generations))
    if is_marvin:
        print("Perm", np.mean(perm))
        print("Static", np.mean(static))
        print("Dynamic", np.mean(dynamic))

    sorted_mutations = sorted(mutations)
    sorted_generations = sorted(generations)
    return sorted_mutations, sorted_generations


if __name__ == "__main__":
    if sys.argv[1] == 'marvin':
        is_marvin = True

    if is_marvin:
        all_mut, all_gen = process(sys.argv[2] + "A_")
        static_mut, static_gen = process(sys.argv[2] + "S_")
        dynamic_mut, dynamic_gen = process(sys.argv[2] + "D_")
    else:
        all_mut, all_gen = process(sys.argv[2])

    plt.plot(all_mut, range(1, len(all_mut) + 1), 'b', label='All')
    plt.title("Evasion Progress by Mutations")
    plt.xlabel('Number of Mutations')
    plt.ylabel('Evasive Variants Found')

    if is_marvin:
        plt.plot(static_mut, range(1, len(static_mut)+1), 'r', label='Static')
        plt.plot(dynamic_mut, range(1, len(dynamic_mut)+1), 'g', label='Dynamic')
        plt.legend()
        plt.savefig('../output/figures/test/marvin_mutations.png')
    else:
        plt.savefig('../output/figures/test/cnn_mutations.png')
    plt.clf()

    plt.plot(all_gen, range(1, len(all_gen) + 1), 'b', label='All')
    plt.title("Evasion Progress over Generations")
    plt.xlabel('Number of Generations')
    plt.ylabel('Evasive Variants Found')

    if is_marvin:
        plt.plot(static_gen, range(1, len(static_gen)+1), 'r', label='Static')
        plt.plot(dynamic_gen, range(1, len(dynamic_gen)+1), 'g', label='Dynamic')
        plt.legend()
        plt.savefig('../output/figures/test/marvin_generations.png')
    else:
        plt.savefig('../output/figures/test/cnn_generations.png')
    plt.clf()
