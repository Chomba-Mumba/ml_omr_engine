import tensorflow as tf

import numpy as np
import random
import os
import gc

from train import fit
from model import OMREnginePatchGan, OMREngineUNet
from data_loader import DataLoader

def randomise_hyperparams():
    #TODO - different parameters for gneerator and discriminator?
    return {
        'learning_rate': random.uniform(1e-5, 1e-3),
        'dropout_rate': random.uniform(0.2, 0.5),
        'num_filters': random.choice([32, 64, 128]),
        'batch_size': random.choice([8, 16, 32]),
        'epochs': random.choice([5, 10, 15])
    }

class Individual:
    def __init__(self):
        #generator and discrimnator chromosomes
        self.gen_chromosomes = randomise_hyperparams()
        self.disc_chromosomes = randomise_hyperparams()

        self.fitness = None
        self.train_ds, self.test_ds = None, None

        #randomise hyper parameters
        self.generator = OMREngineUNet(**self.gen_chromosomes)

        self.discriminator = OMREnginePatchGan(**self.disc_chromosomes)

    def cal_fitness(self, num_epochs):
        self.fitness = fit(self.generator, self.discriminator, self.train_ds, self.test_ds, num_epochs)
        return self.fitness

class Population:
    def __init__(self, pop_size, mut_rate):
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.population = [Individual() for _ in range(len(pop_size))]

    def selection(self):
        selected = []

        for _ in range(self.pop_size)//2:
            #tournament selection with 3 individuals
            contenders = random.sample(self.population, 3)
            winner = max(contenders, key=lambda ind: ind.fitness)
            selected.append(winner)

        return selected
            
    def crossover(self, p1, p2):
        child = Individual()

        #select hyperparameters between parents for child
        for key in child.gen_chromosomes:

            #discriminator
            child.disc_chromosomes[key] = random.choice([p1.disc_chromosomes[key], p2.disc_chromosomes[key]])

            #generator
            child.gen_chromosomes[key] = random.choice([p1.gen_chromosomes[key], p2.gen_chromosomes[key]])

        return child

    def mutate(self, individual):
        for key in individual.gen_chromosomes:
            if random.random() < self.mut_rate:
                individual.gen_chromosomes[key] = randomise_hyperparams()[key]
                individual.disc_chromosomes[key] = randomise_hyperparams()[key]
    
    def evolve(self, train_ds, test_ds, num_epochs):
        #provide data
        self.train_ds = train_ds
        self.test_ds = test_ds

        #assign fitness values to population
        for ind in self.population:
            ind.cal_fitness(num_epochs)

        selected = self.selection()
        new_population = []

        while len(new_population) < self.pop_size:
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)

            #replace generator and discriminator
            del child.generator
            del child.discriminator
            gc.collect() #remove deleted models from memory

            child.generator = OMREngineUNet(**child.gen_chromosomes)
            child.discriminator = OMREnginePatchGan(**child.disc_chromosomes)
            new_population.append(child)

        self.population = new_population

def run_geneteic_algorithm():
    pop_size = os.environ.get("POP_SIZE")
    mut_rate = os.environ.get("MUT_RATE")
    generations = os.environ.get("GENERATIONS")
    num_epochs = os.environ.get("NUM_EPOCHS")

    #load data 
    loader = DataLoader(1,2)
    base_dir = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(base_dir, "data", "input")
    tar_path = os.path.join(base_dir, "data", "target")

    data = loader.get_dataset(src_path, tar_path)
    train_ds, test_ds, val_ds = loader.split(data)

    pop = Population(pop_size, mut_rate)

    for gen in range(generations):
        print(f"=== Generation {gen} ===")
        pop.evolve(train_ds, test_ds, num_epochs)
        best = max(pop.population, key=lambda ind: ind.fitness)
        print(f"Best fitness: {best.fitness}")