'''
CS 441 Section 001
Jared Collums (jcollums@pdx.edu)
Programming Assignment #2: Solving the 8-Queens Problem with a Genetic Algorithm
'''

from random import seed, randint, random, choices
from time import perf_counter
import json

# Constants
COLS = 8
ROWS = 8
QUEENS = 8
optimalFitness = (QUEENS * (QUEENS - 1)) / 2 # 8 choose 2

# Execute genetic algorithm with population size / mutation probability combinations
def main():
    seed(perf_counter())

    sizes = [10, 50, 100, 500, 1000]
    probs = [0.1, 0.2, 0.3, 0.4]
    trials = range(1)
    limit = 100

    # For summary
    log = {}
    for populationSize in sizes:
        log[populationSize] = {}
        for mutationProb in probs:
            log[populationSize][mutationProb] = []

    # Start the loop!
    for populationSize in sizes:
        for mutationProb in probs:
            for trial in trials:
                print("Population Size: " + str(populationSize))
                print("Mutation Probability: " + str(mutationProb))
                # print("Trial #" + str(trial))

                [mostFit, averageFitness] = genetic_algorithm(populationSize, mutationProb, limit, True)
                
                f = fitness(mostFit)
                if mostFit != "":
                    display(mostFit)
                    print("Representation: " + mostFit)
                    print("Fitness: " + str(f))
                    print("Interations: " + str(len(averageFitness)))
                else:
                    print("Failed to solve")
                
                log[populationSize][mutationProb].append([mostFit, f, len(averageFitness), averageFitness])

    # Print summary of runs
    filename = "8queens_summary.txt"
    summary = "Population\tMutation\tGeneration\tAverage Fitness\n"
    for s in sizes:
        for p in probs:
            for g in range(0, len(log[s][p][0][3])):
                summary += str(s) + "\t" + str(p) + "\t" + str(g) + "\t" + str(log[s][p][0][3][g]) + "\n"
    writeToFile(filename, summary)
    print("Summary written to " + filename)

# Execute algorithm
def genetic_algorithm(populationSize, mutationProb, limit, verbose = False):
    '''
    Each fitness value is a key in the population dictionary, 
    containing an array of individiuals with that fitness value
    (This helps greatly for selection efficiency purposes,
    since fitness values are a small number of discrete values)
    '''
    population = {}
    
    # Random starting population
    generatePopulation(population, populationSize)
    
    # Repeat until offspring is perfectly fit
    mostFit = ""
    iterations = 0
    timer = 0
    
    averageFitness = []
    while fitness(mostFit) < optimalFitness:
        iterations += 1

        # Return indicating failure
        if iterations >= limit:
            return ["", averageFitness]

        # Print progress roughly every second
        if verbose:
            if iterations == 1 or perf_counter() - timer > 1:
                print("Starting generation #" + str(iterations))
                timer = perf_counter()

        population = nextGeneration(population, populationSize, mutationProb)
        mostFit = findMostFit(population)

        averageFitness.append(getAverageFitness(population))

    return [mostFit, averageFitness]

# Create a new generation from the current population
def nextGeneration(population, populationSize, mutationProb):
    distribution = weightedDistribution(population, populationSize)

    new_population = {}
    for _ in range(0, populationSize):
        # Random weighted selection based on fitness
        state1 = selection(population, distribution)
        state2 = selection(population, distribution)

        # Do crossover
        newState = reproduce(state1, state2)
        # print(state1 + " + " + state2 + " = " + newState)

        # Small chance of mutation
        if random() < mutationProb:
            newState = mutate(newState)
            # print("Mutate -> " + newState)

        # Create new key for fitness value if it doesn't exist
        f = fitness(newState)
        if not f in new_population.keys():
            new_population[f] = []
        new_population[f].append(newState)

    return new_population

# Return number of non-attacking queen pairs
def fitness(state):
    # Uninitialized state
    if state == "": 
        return -1

    result = 0
    for x in range(COLS):
        for y in range(COLS):
            if (x != y):
                sameRow = (state[x] == state[y])
                sameDiagonal = (abs(x - y) == abs(int(state[x])) - abs(int(state[y])))

                if not (sameRow or sameDiagonal):
                    result += 1

    return int(result / 2)

# Crossover at random spot
def reproduce(state1, state2):
    i = randint(0, COLS)
    child1 = state1[:i] + state2[i:]
    child2 = state2[:i] + state1[i:]

    # Return the most fit of the two offspring
    if (fitness(child1) > fitness(child2)):
        return child1
    return child2
    
# Swap two random pieces with each other
def mutate(state):
    i1 = randint(0, COLS - 1)
    i2 = randint(0, COLS - 1)

    # Turn it into an array so we can more easily swap
    arr = list(state)
    temp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = temp

    return "".join(arr)

# Select a random individual weighted by fitness
def selection(population, distribution):
    fitValue = distribution[randint(0, len(distribution)-1)]
    individual = population[fitValue][randint(0, len(population[fitValue])-1)]

    return individual

# Generate a random state
def generateState():
    state = ""
    for _ in range(COLS):
        state += str(randint(0, ROWS - 1))
    return state

# Generate a random population
def generatePopulation(new_population, populationSize):
    new_population.clear()
    for _ in range(populationSize):
        newState = generateState()
        f = fitness(newState)

        if not f in new_population.keys():
            new_population[f] = []
        new_population[f].append(newState)

# Find most fit individual in the population
def findMostFit(population):
    # Return the first individual from the highest fitness key
    keys = list(population.keys())
    keys.sort()

    maxKey = keys[len(keys)-1]
    return population[maxKey][0]

# Calculate average fitness of a population
def getAverageFitness(population):
    sum = 0
    count = 0
    for k in population.keys():
        sum += int(k) * len(population[k])
        count += len(population[k])

    return sum / count

# Creates a distribution of fitness values weighted by the most fit
def weightedDistribution(population, populationSize):
    weights = []
    total = 0

    # Calculate sum of fitness values
    for i in population.keys():
        total += i

    # Determine weight for each fitness value
    for k in population.keys():
        weights.append(k / total)

    # Create a distribution of fitness values weighted by the most fit
    distribution = choices(list(population.keys()), weights = weights, k = populationSize)
    return distribution

# Disply the board in a grid with Q representing the Queens
def display(state):
    for y in range(ROWS):
        for x in range(COLS):
            if x == 0:
                print("| ", end = "")
            if state[x] == str(y):
                print('Q', end = ' | ')
            else:
                print(' ', end = ' | ')

        print('\n')

# Write to file
def writeToFile(filename, contents):
    f = open(filename, 'w')
    f.write(contents)
    f.close()

main()