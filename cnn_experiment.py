import os
import random
import torch

nop = ""
max_generation = 0
population_size = 0
fitness_rate = 0.0
fitness_boundary  = fitness_rate*population_size


for seed_file in os.listdir(seed_dir):

    #load malicious seed
    with open(seed_file, "r") as f:
        seed_opseq = f.readlines()

    population = [seed_opseq]

    #iterate over generations
    for generation in range(max_generation):

        for member_index in range(fitness_boundary, population_size):    

            #modify existing unfit opseq by inserting one more
            #TODO: verify mutation method of unfit variants
            opseq = population[member_index]

            insertion_index = random.randrange(len(opseq))
            opseq.insert(insertion_index, nop)

            population[member_index] = opseq

        #save mutated variants, if necessary

        
        #run model on variants in gen


        #evaluate fitness
        confidence_scores = []

        confidence_scores.sort()