from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from noise import snoise2

parameters = { 
    'mu': 0.1, # mutation rate
    'alpha': 1, # growth rate
    'beta': -0.1,         
    'fecundity_m': 2, # mean fecundity
    'fecundity_sd': 1, # standard deviation of fecundity
    'dispersal': 0.1, # dispersal rate
    'offspring_vitality': 1, # offspring vitality
    'resource_preference': 0.5 # resource preference

    
}

pop = np.empty((20, 20), dtype=object)

res = np.zeros((pop.shape[0], pop.shape[0]))

tile_type = np.zeros((pop.shape[0], pop.shape[1]))

demographic_history = []

resource_history = []

offspring_count_list = []

trait_matching_list = []

vitality_list = []

pop_distributions = []

preference_distribution = []

class Individual:
    def __init__(self, traits=parameters,
                sex="female",
                pos_x=0,
                pos_y=0,
                vitality=2,
                age=0):
        
        self.traits = traits.copy()
        self.sex    = sex
        self.pos_x  = pos_x
        self.pos_y  = pos_y
        self.vitality = vitality
        self.age = age
        self.growth_rate = 0
        self.offspring_count = 0
        self.vitality_curve_list = []
        self.res_curve_list = []
        for key, value in self.traits.items():
            m = random.normal(0, abs(self.traits['mu']))
            self.traits[key] = value + m
    

        
    def reproduce(self, other, fecundity):
        """Reproduce with another Individual."""
        for i in range(int(fecundity)):
            self.offspring_count += 1
            self.vitality -= abs(self.traits['offspring_vitality'])
            child_traits = self.traits.copy()
            for key in child_traits:
                child_traits[key] = (child_traits[key] + other.traits[key]) / 2
            
            
            child = Individual(traits=child_traits,
                                sex = ["female", "male"][random.randint(0, 2)],
                                pos_x = self.pos_x,
                                pos_y = self.pos_y,
                                vitality = abs(self.traits['offspring_vitality']))
            pop[self.pos_x, self.pos_y].append(child)

    def move(self):
        """Move the Individual based on its dispersal traits."""
        if random.random() < self.traits['dispersal']:
            pop[self.pos_x, self.pos_y].remove(self)
            move_x = round(random.normal(0, 1))
            move_y = round(random.normal(0, 1))
            if self.pos_x + move_x >= 0 and self.pos_x + move_x < pop.shape[0]:
                self.pos_x += move_x

            if self.pos_y + move_y >= 0 and self.pos_y + move_y < pop.shape[1]:
                self.pos_y += move_y
            pop[self.pos_x, self.pos_y].append(self)

    def grow(self):
        """Age the Individual, reducing vitality."""
        self.vitality += abs(self.traits['alpha'])/(self.age/20+1)  * (1 - (abs(self.traits['resource_preference'] - tile_type[self.pos_x, self.pos_y]))) 
        if res[self.pos_x, self.pos_y] > 3*abs(self.traits['alpha'])/(1 - (abs(self.traits['resource_preference'] - tile_type[self.pos_x, self.pos_y]))):
            res[self.pos_x, self.pos_y] -= abs(self.traits['alpha'])
        else:
            self.vitality -= abs(self.traits['alpha']) 
        self.vitality -= abs(self.traits["beta"]) * self.age
        
        self.age += 1
        self.res_curve_list = res[self.pos_x, self.pos_y]
        self.vitality_curve_list.append(self.vitality)
        if self.vitality <= 0:
            self.die()

    def die(self):
        """Remove the Individual from the population."""
        global pop
        pop[self.pos_x, self.pos_y].remove(self)
        offspring_count_list.append(self.offspring_count)
        trait_matching_list.append((1 - (abs(self.traits['resource_preference'] - tile_type[self.pos_x, self.pos_y]))) )
        del self
    
    def step(self):
        """Perform a step in the simulation."""
        self.move()
        self.grow()
        if self.sex == "female":
            fecundity = random.normal(self.traits['fecundity_m'], abs(self.traits['fecundity_sd']))
            if self.vitality > fecundity * abs(self.traits['offspring_vitality']):
                for other in pop[self.pos_x, self.pos_y]:
                    if other != self and other.sex == "male":
                        self.reproduce(other, fecundity)
                        break



def simulate(steps=10, starting_population=2,
             starting_resource=5, resource_growth_rate=1.5,
             resource_k=100):
    """Run the simulation for a given number of steps."""
    global pop
    global tile_type
    global res
    global pop_distributions

    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            pop[i, j] = []

            # Initialize population with 10 individuals
            for q in range(starting_population):
                individual = Individual(pos_x=i, pos_y=j,
                                        sex=["female", "male"][random.randint(0, 2)])
                pop[i, j].append(individual)

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = starting_resource
    #tile_type = np.random.rand(tiles.shape[0], tiles.shape[1])  # Random tile quality for each tile

    tile_type = np.zeros((pop.shape[0], pop.shape[1]))  # Initialize tile types
    for i in range(int(pop.shape[0]/2)):
        for j in range(pop.shape[1]):
            tile_type[j, pop.shape[0]-i-1] = 1

    pop_dist = np.zeros((pop.shape[0], pop.shape[1]), dtype=int)
    mean_preferences = np.zeros((pop.shape[0], pop.shape[1]))
    for step in range(steps):
        # Update demographic history
        demographic_history.append(len([ind for cell in pop.flat for ind in cell]))
        resource_history.append(np.mean(res))
        for i in range(pop.shape[0]):
            for j in range(pop.shape[1]):
                res[i, j] += res[i, j] * (1-res[i, j]/resource_k) * resource_growth_rate
                if res[i, j] < 0:
                    res[i, j] = 0
                pop_dist[i, j] = len(pop[i, j])
                preferences = [ind.traits['resource_preference'] for ind in pop[i, j]]
                if preferences:
                    mean_preferences[i, j] = np.mean(preferences)
                else:
                    mean_preferences[i, j] = 0.5
                for individual in pop[i, j]:
                    individual.step()
        pop_distributions.append(pop_dist.copy())
        preference_distribution.append(mean_preferences.copy())


        # Remove dead individuals
        for i in range(pop.shape[0]):
            for j in range(pop.shape[1]):
                pop[i, j] = [ind for ind in pop[i, j] if ind.vitality > 0]


        if step % 100 == 0 or step < 100:
            print(f"Step {step}: Population size: {demographic_history[len(demographic_history)-1]}")
            print(f"Resource level: {resource_history[len(resource_history)-1]}")

    print([ind for cell in pop.flat for ind in cell][0].traits)
    print([ind for cell in pop.flat for ind in cell][int(demographic_history[len(demographic_history)-1]/2)].traits)
    #print([ind for cell in pop.flat for ind in cell][demographic_history[len(demographic_history)-1]-1].traits)
    print(f"Simulation completed after {steps} steps. Population size: {demographic_history[len(demographic_history)-1]}")
    

def plot_population_distribution():
    """Plot the population distribution on the grid."""
    counts = np.zeros_like(pop, dtype=int)

    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            counts[i, j] = len(pop[i, j])
    
    plt.imshow(counts, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Population Density')
    plt.title('Population Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def plot_tile_type():
    """Plot the population distribution on the grid."""
    plt.imshow(tile_type, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Tile Type')
    plt.title('Population Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def plot_demographic_history():
    """Plot the demographic history of the population."""
    plt.plot(demographic_history)
    plt.title('Demographic History')
    plt.xlabel('Time Steps')
    plt.ylabel('Population Size')
    plt.ylim(0, max(demographic_history) + 100)
    plt.grid()
    plt.show()

def plot_resource_history():
    """Plot the resource history of the population."""
    plt.plot(resource_history)
    plt.title('Resource History')
    plt.xlabel('Time Steps')
    plt.ylabel('Resource Level')
    plt.ylim(0, max(resource_history) + 100)
    plt.grid()
    plt.show()

def plot_preference():
    """Plot the resource preference of the population."""
    preferences = [ind.traits["resource_preference"] for cell in pop.flat for ind in cell if ind is not None]
    plt.hist(preferences, bins=20, alpha=0.7)
    plt.title('Resource Preference Distribution')
    plt.xlabel('Resource Preference')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()  

def plot_mean_preference_for_tiles():
    """Plot the mean resource preference for each tile type."""

    mean_preferences = np.zeros((pop.shape[0], pop.shape[1]))
    
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            preferences = [ind.traits['resource_preference'] for ind in pop[i, j]]
            if preferences:
                mean_preferences[i, j] = np.mean(preferences)
            else:
                mean_preferences[i, j] = 0.5
    

    plt.imshow(mean_preferences, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Mean Preferences')
    plt.title('Population Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


def plot_offspring_count_to_trait_matching():
    """Plot the relationship between offspring count and trait matching."""
    plt.scatter(offspring_count_list, trait_matching_list, alpha=0.5)
    plt.title('Offspring Count vs Trait Matching')
    plt.xlabel('Offspring Count')
    plt.ylabel('Trait Matching (1 - |Preference - Tile Type|)')
    plt.grid()
    plt.show()

def plot_vitality_curve():
    """Plot the vitality curve of the population."""
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            for individual in pop[i, j]:
                plt.plot(individual.vitality_curve_list, label=f'Individual ({i}, {j})')
    
    plt.title('Vitality Curve of Individuals')
    plt.xlabel('Time Steps')
    plt.ylabel('Vitality')
    plt.grid()
    plt.show()


    
def plot_resource_distribution():
    """Plot the resource distribution on the grid."""
    plt.imshow(res, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Resource Level')
    plt.title('Resource Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def animate_population_distribution():
    """Animate the population distribution over time."""
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros_like(pop_distributions[0], dtype=int), vmin=0, vmax=np.max(pop_distributions),
                   cmap='hot', interpolation='nearest')

    def update(frame):
        im.set_data(pop_distributions[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, interval = 10, frames=range(len(pop_distributions)))
    plt.show()
    
def animate_mean_preferences():
    """Animate the mean preferences over time."""
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros_like(preference_distribution[0], dtype=int), vmin=0, vmax=np.max(preference_distribution),
                   cmap='hot', interpolation='nearest')

    def update(frame):
        im.set_data(preference_distribution[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, interval = 10, frames=range(len(preference_distribution)))
    plt.show()

def plot_phase_diagram_of_resource_and_population():
    """Plot a phase diagram of resource and population."""
    # plot line plot
    plt.scatter(resource_history, demographic_history, c='blue', alpha=0.5)
    plt.plot(resource_history, demographic_history, color='red', alpha=0.5)

    plt.title('Phase Diagram of Resource and Population')
    plt.ylabel('Population Size')
    plt.xlabel('Resource Level')
    plt.grid()
    plt.show()

def sample_parameter_space(fidelity = 10, length = 10):
    deltas = np.zeros((fidelity, fidelity))
    q = 0
    for i in range(fidelity):
        for j in range(fidelity):
            print(q)
            q += 1
            simulate(steps=length, resource_growth_rate=1+i/50, resource_k=10+j*50, starting_resource=10+j*50/2)
            deltas[i, j] = np.cov(demographic_history)
            
    plt.imshow(deltas, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Population Change')
    plt.title('Phase Diagram of Resource and Population')
    plt.xlabel('growth rate')     
    plt.ylabel('resource K')
    plt.show()
    

#simulate(steps=100)
simulate(steps=2000, resource_growth_rate=1+1/50, resource_k=10+1*50, starting_resource=10+1*50/2)

plot_population_distribution()
plot_demographic_history()
plot_resource_history()
plot_preference()
plot_mean_preference_for_tiles()
plot_vitality_curve()
plot_resource_distribution()

animate_population_distribution()
animate_mean_preferences()
plot_phase_diagram_of_resource_and_population()

#sample_parameter_space(length=1000, fidelity=5)