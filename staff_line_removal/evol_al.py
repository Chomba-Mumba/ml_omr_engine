import tensorflow as tf

import numpy as np
import random
import os

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
        self.gen_chrom = randomise_hyperparams()
        self.disc_chrom = randomise_hyperparams()
        self.fitness = None
        self.train_ds, self.test_ds = None, None

        #randomise hyper parameters
        self.generator = OMREngineUNet(**self.chrom)

        self.discriminator = OMREnginePatchGan(**self.chrom)

    def cal_fitness(self, num_epochs):
        self.fitness = fit(self.generator, self.discriminator, self.train_ds, self.test_ds, num_epochs)
        return self.fitness

class Population:
    def __init__(self, pop_size, mut_rate):
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.population = [Individual() for _ in range(len(pop_size))]

    def selection(self):#TODO - which individuals are slected during selection?
        selected = []

        for _ in range(self.pop_size):
            #tournament selection with 3 individuals
            contenders = random.sample(self.population, 3)
            winner = max(contenders, key=lambda ind: ind.fitness)
            selected.append(winner)

        return selected
            
    def crossover(self, p1, p2):
        child = Individual()

        #select hyperparameters between parents for child
        for key in child.chrom:
            child.chrom[key] = random.choice([p1.chrom[key], p2.chrom[key]])

        return child

    def mutate(self, individual):
        for key in individual.chrom:
            if random.random() < self.mut_rate:
                individual.chrom[key] = randomise_hyperparams()[key]
    
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
            child.generator = OMREngineUNet(**child.gen_chrom)
            child.discriminator = OMREnginePatchGan(**child.disc_chrom)
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