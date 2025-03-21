import tensorflow as tf
import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
import math
from datetime import date, datetime
from scipy import special
from tensorflow.keras import Model

np.random.seed(0)

# Function to create a matrix of zeros with a given batch size and N
def create_matrix(batch_size, N):
    matrix = np.zeros((batch_size, N), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.arange(0, N, 1)
    return matrix

# Create a pilot matrix with a given batch size and Np
def create_pilot(batch_size, Np):
    matrix = np.zeros((batch_size, Np), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.array([1, N - 2])
    return matrix

# Function to create a pilot value matrix with a given batch size and Np
def pilot_value(batch_size, Np):
    matrix = np.zeros((batch_size, Np), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.array([-1, 1])
    return matrix

# Initialize constants and parameters
NUM_BITS_PER_SYMBOL = 2 
N = 32  # Number of OFDM subcarriers
batch_size = 1000  # Generate data for 1000 times N
Np = 2  # Number of pilot subcarriers
BLOCK_LENGTH = (N - Np) * NUM_BITS_PER_SYMBOL

# Create a QAM constellation for mapping
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)

# Initialize a binary source
binary_source = sn.utils.BinarySource()

# Create a mapper for symbol mapping
mapper = sn.mapping.Mapper(constellation=constellation)

# Generate random binary data
bits = binary_source([batch_size, BLOCK_LENGTH])

# Map binary data to symbols
x = mapper(bits)

# Create matrices for all subcarriers, pilot subcarriers, and data subcarriers
allCarriers = create_matrix(batch_size, N)
pilotCarriers = create_pilot(batch_size, Np)
dataCarriers = np.delete(allCarriers, pilotCarriers, axis=1)

# Allocate bits to dataCarriers and pilotCarriers in the symbol matrix
symbol = np.zeros((batch_size, N), dtype=complex)
symbol[:, pilotCarriers] = pilot_value(batch_size, Np)
symbol[np.arange(batch_size)[:, None], dataCarriers] = x

# Generate the OFDM time-domain signal
OFDM_time = np.sqrt(N) * np.fft.ifft(symbol)

# Find PAPR (Peak-to-Average Power Ratio) for each signal in the batch
idx = np.arange(0, 1000) ##mark 3 //PAPR
PAPR = np.zeros(len(idx))

for i in idx:
    var = np.var(OFDM_time[i])
    peakValue = np.max(abs(OFDM_time[i]) ** 2)
    PAPR[i] = peakValue / var

# Convert PAPR to dB
PAPR_dB = 10 * np.log10(PAPR)   

# Find CCDF (Complementary Cumulative Distribution Function)

# Define the range of PAPR values
PAPR_Total = len(PAPR_dB)
ma = max(PAPR_dB)
mi = min(PAPR_dB)
eixo_x = np.arange(mi, ma, 0.1)
##eixo_x_values.append(eixo_x)

# Initialize a list to store CCDF values
y = []

# Calculate CCDF values for each PAPR threshold
for j in eixo_x:
    A = len(np.where(PAPR_dB > j)[0]) / PAPR_Total
    y.append(A)  # Append A to the list y

CCDF = y #plot the relationship between the PAPR and its probability of occurrence in the given signal.

#capture PAPR vs. CCDF from the og signal
eixo_xOG=eixo_x
CCDFOG=CCDF

# Define GA parameters
population_size = 50  # Number of chromosomes in the population
chromosome_length = N  # Length of each chromosome (same as the subcarriers)
mutation_rate = 0.1  # Rate of mutation for genetic diversity

# Initialize an empty population
population = []

# Generate random chromosomes and add them to the population
for _ in range(population_size):
    # Randomly initialize a chromosome with values -1, 0, or 1
    chromosome = np.random.choice([-1, 0, 1], size=chromosome_length)
    population.append(chromosome)

# Fitness function to evaluate the PAPR reduction performance of a chromosome
def fitnessAval(chromosome):
    # Apply the chromosome to the OFDM signal
    ## original with error modified_symbol = symbol * chromosome[:, np.newaxis] 
    ##New one possibily fixed
    modified_symbol = symbol * chromosome.T ##is used to align the dimensions of the two variables for proper multiplication.
    modified_OFDM_time = np.sqrt(N) * np.fft.ifft(modified_symbol)
    
    # Calculate PAPR for each modified signal
    PAPR = np.max(np.abs(modified_OFDM_time) ** 2, axis=1) / np.var(modified_OFDM_time, axis=1)
    
    # Calculate the fitness as the inverse of the mean PAPR (lower is better)
    fitness = 1.0 / np.mean(PAPR)
    
    return fitness

def complexity(ind_complex):
    # Define a function named "complexity"

    vetor1 = np.array(ind_complex).astype(int)
 
    # Create NumPy arrays "vetor1" and "vetor2" from the input arrays "ind_complex" and "ind_complex_no"
    
    vetor1 = vetor1.flatten()
 
    # Flatten the arrays to one-dimensional arrays

    pmf1 = np.bincount(vetor1) / len(vetor1)

    # Calculate the Probability Mass Function (PMF) for "vetor1" and "vetor2"

    fig, ax = plt.subplots(figsize=(10, 8))
    # Create a figure and axis for the plot

    plt.stem(np.arange(len(pmf1)), pmf1, linefmt='b-', markerfmt='bo', basefmt='', label='WithMemory')
    plt.stem(4, 1, linefmt='g-', markerfmt='g^', basefmt='', label='PTS')### mark
    # Plot stems for PMF1, PMF2, and a specific point (4, 1)

    ax.set_xlabel('Number of Iterations', fontsize=17, fontweight='bold')
    ax.set_ylabel('Probability Mass Function', fontsize=17, fontweight='bold')
    # Set axis labels

    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='black')
    ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='black')
    ax.grid(axis='both', linestyle='--', alpha=0.7, color='black')
    # Add grid lines to the plot

    ax.set_facecolor('white')
    # Set the background color of the plot

    ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')
    # Add a legend to the plot

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    # Customize the appearance of the plot's spines (borders)

    ax.tick_params(axis='both', which='major', labelsize=17)
    # Set tick parameters (font size) for major ticks

    filename = 'Complexity' + datetime.now().strftime('%d-%m-%Y--%H:%M:%S') + '.png'
    # Create a filename based on the current date and time

    # Save the figure with the updated filename
    plt.savefig(filename)

def slm(signal, num_candidates):
    # Generate phase sequences
    phase_sequences = [np.exp(1j * 2 * np.pi * np.random.rand(N)) for _ in range(num_candidates)]
    
    # Apply phase sequences to signal and calculate PAPR for each candidate
    papr_values = []
    for phase_sequence in phase_sequences:
        candidate_signal = signal * phase_sequence
        papr = np.max(np.abs(candidate_signal) ** 2) / np.var(candidate_signal)
        papr_values.append(papr)
    
    # Select candidate with minimum PAPR
    best_candidate = phase_sequences[np.argmin(papr_values)]
    return signal * best_candidate

# Number of generations
num_generations = 50

best_fitness = 0
eixo_x_values = []
CCDF_values = []
papr_values_slm = []
ccdf_values_slm = []
ind_complex = 0

for x in range(pow(10,3)): ## mark inicial

    # Main GA loop  mark GA
    for generation in range(num_generations):
        fitness_scores = []

        # Shuffle population for tournament selection
        np.random.shuffle(population)

        # Evaluate the fitness of each chromosome
        for chromosome in population:
            fitness_scores.append(fitnessAval(chromosome))

        # Perform tournament selection to choose parents
        tournament_size = 3
        selected_parents = []
        
        for _ in range(population_size // 2):
            tournament_indices = np.random.choice(range(len(population)), size=tournament_size, replace=False)
            tournament_scores = [fitness_scores[i] for i in tournament_indices]
            #ind_complex++
            winner_index = tournament_indices[np.argmax(tournament_scores)]
            selected_parents.append(population[winner_index])

        # Recombination process
        next_population = []
        for parent1, parent2 in zip(selected_parents[::2], selected_parents[1::2]):
            # Perform uniform crossover
            child1 = np.empty_like(parent1)
            child2 = np.empty_like(parent2)
            
            for i in range(chromosome_length):
                if np.random.random() < 0.5:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
                else:
                    child1[i] = parent2[i]
                    child2[i] = parent1[i]

            # Perform mutation
            if np.random.random() < mutation_rate:
                mutation_point = np.random.randint(0, chromosome_length)
                child1[mutation_point] = np.random.choice([-1, 0, 1])
            if np.random.random() < mutation_rate:
                mutation_point = np.random.randint(0, chromosome_length)
                child2[mutation_point] = np.random.choice([-1, 0, 1])

            next_population.extend([child1, child2])

    # Update the population with the next generation
        population = next_population

    # Find the new best fitness score in the current generation
        new_best_fitness = np.max(fitness_scores)

    # Calculate PAPR and CCDF for the best chromosome in the current generation
        best_chromosome = population[np.argmax(fitness_scores)]
        modified_symbol = symbol * best_chromosome.T
        modified_OFDM_time = np.sqrt(N) * np.fft.ifft(modified_symbol)
        PAPR = np.max(np.abs(modified_OFDM_time) ** 2) / np.var(modified_OFDM_time)
        CCDF = np.exp(-np.abs(PAPR))

        # Store PAPR and CCDF values
        eixo_x_values.append(PAPR)
        CCDF_values.append(CCDF)

    # Check if the new best fitness is better than the previous best fitness
        if new_best_fitness > best_fitness:
            # Update the best fitness value
            best_fitness = new_best_fitness

            # Print the best fitness score in the current generation
            print(f"Generation {generation+1}, Best fitness: {best_fitness}")
                  
            # Terminate the loop if the best fitness is below a threshold
        if best_fitness < 0.001:
            print('Best fitness of 0.001 was achieved!')
            print('Breaking.....')
            break;
    
    print(' ')
    print('Grupos: ' + str(x+1) )
    print(' ')

# Apply SLM to the original signal
slm_symbol = slm(symbol, num_candidates=10)

# Calculate PAPR and CCDF for the SLM-modified signal
modified_OFDM_time = np.sqrt(N) * np.fft.ifft(slm_symbol)
PAPR = np.max(np.abs(modified_OFDM_time) ** 2) / np.var(modified_OFDM_time)
CCDF = np.exp(-np.abs(PAPR))

# Store PAPR and CCDF values for SLM
papr_values_slm.append(PAPR)
ccdf_values_slm.append(CCDF)

complexity(ind_complex) 

# Plot PAPR vs. CCDF for the general proccess
plt.semilogy(eixo_x_values, CCDF_values, 'x-', linewidth=2.5, label=f"GA")
plt.semilogy(eixo_xOG, CCDFOG, 'x-', linewidth=2.5, label=f"Original Signal")
plt.semilogy(papr_values_slm, ccdf_values_slm, 'o-', linewidth=2.5, label='SLM')
##mark 3 Plot the SLM here
plt.legend(loc="lower left")
plt.xlabel('PAPR (dB)')
plt.ylabel('CCDF')
plt.grid()
plt.ylim([1e-2, 1])
plt.title("PAPR vs. CCDF genaral graph",fontsize=25)
filename = 'PAPR_X_CDF_General' + datetime.now().strftime('%d-%m-%Y--%H:%M:%S') + '.png'

# Save the figure with the updated filename
plt.savefig(filename)
    
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class UncodedSystemAWGN(Model): 
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):

        super().__init__() 

        # Initialization of variables and objects
        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        
    def __call__(self, batch_size, ebno_db):
        
        global OFDM_time    
        global bits
        
        # Calculate noise variance based on Eb/No
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
    
        # Create pilot carriers
        pilotCarriers = create_pilot(batch_size, Np)
        
        # Channel
        h = np.array([1])
        self.H = np.fft.fft(h,self.N)        
        self.OFDM_RX_FD = OFDM_time
        
        # Add AWGN noise to the signal
        y = self.awgn_channel([self.OFDM_RX_FD, no])
        y= np.fft.fft(y)
        
        # Remove pilot carriers from received signal
        OFDM_demod = np.delete(y, pilotCarriers,axis=1)
        
        # Demapping to obtain LLR values
        llr = self.demapper([OFDM_demod,no])
        
        # Store original output and print LLR values
        self.Out_Ori = y
        print(llr)
        
        return bits, llr


class UncodedSystemAWGN_GA(Model): 
    def __init__(self, num_bits_per_symbol, block_length, Subcarriers):

        # Call the parent class constructor
        super().__init__() 

        # Initialize class variables
        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH  # Assuming BLOCK_LENGTH is a defined constant or variable
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        
    def __call__(self, batch_size, ebno_db):
        
        # Define global variables that are being used or modified
        global modified_OFDM_time    
        global bits
        
        # Convert EBN0 to noise power
        no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=1.0)
        
        # Generate pilot carriers
        pilotCarriers = create_pilot(batch_size, Np)  # Assuming Np is defined
        
        # Channel modulation and demodulation
        self.OFDM_RX_FD = modified_OFDM_time
        y_ = self.awgn_channel([self.OFDM_RX_FD, no])
        y_ = np.fft.fft(y_)
        OFDM_demod_ = np.delete(y_, pilotCarriers, axis=1)
        
        # Compute LLR values
        llr_GA = self.demapper([OFDM_demod_, no])
        self.Out_GA = y_
        
        return bits, llr_GA

   


#%%
  
# Instanciando o modelo
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)
model_uncoded_awgn_GA = UncodedSystemAWGN_GA(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)

#%%

# Sionna provides a utility to easily compute and plot the bit error rate (BER).

batch_size = batch_size
SNR = np.arange(0, 20)

EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations

# Original Simulation:

ber_plots = sn.utils.PlotBER()
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = batch_size,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM = np.array(ber_plots.ber).ravel()

# GA Simulation:
 # Create an instance of the PlotBER class
ber_plots_GA = sn.utils.PlotBER()

# Simulate the Bit Error Rate (BER) for an uncoded communication system
ber_plots_GA.simulate(model_uncoded_awgn_GA,
                      ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                      batch_size=batch_size,
                      num_target_block_errors=100,  # Simulate until 100 block errors occur
                      legend="Uncoded",
                      soft_estimates=True,
                      max_mc_iter=100,  # Run 100 Monte-Carlo simulations (each with batch_size samples)
                      show_fig=False)

# Extract the BER simulation results and flatten them into a 1D array
BER_SIM_GA = np.array(ber_plots_GA.ber).ravel()


#%% Theoretical:

# Define some parameters
M = 2**(NUM_BITS_PER_SYMBOL)  # Number of symbols
L = np.sqrt(M)  # Constellation size
mu = 4 * (L - 1) / L  # Average number of neighbors
Es = 3 / (L ** 2 - 1)  # Constellation adjustment factor

# Create an array of Eb/N0 values
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)

# Initialize an array for theoretical Bit Error Rate (BER) values
BER_THEO = np.zeros((len(ebno_dbs)))

# Calculate theoretical BER for each Eb/N0 value
i = 0
for idx in ebno_dbs:
    BER_THEO[i] = (mu / (2 * N * NUM_BITS_PER_SYMBOL)) * np.sum(
        special.erfc(np.sqrt(np.abs(model_uncoded_awgn.H) ** 2 ##mark V What's special?!
                             * Es * NUM_BITS_PER_SYMBOL * 10**(idx / 10)) / np.sqrt(2)))
    i += 1

# Create a plot
# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot simulated BER data points using circles ('o') in color C1 (green)
ax.plot(ebno_dbs, BER_SIM, 'o', markersize=5, color='C1', label='Simulation Signal')

# Plot GA (genetic algorithm) simulated BER data points using crosses ('x') in color C2 (orange)
ax.plot(ebno_dbs, BER_SIM_GA, 'x', markersize=5, color='C2', label='GA Signal')

# Plot theoretical BER data using a dashed line ('-.') in color C0 (blue) with a thicker linewidth
ax.plot(ebno_dbs, BER_THEO, '-.', color='C0', label='Theoretical', linewidth=2)

# Set the y-axis label and font properties
ax.set_ylabel('Bit Error Rate (BER)', fontsize=16, fontweight='bold')

# Set the x-axis label and font properties
ax.set_xlabel('Eb/N0 (dB)', fontsize=16, fontweight='bold')

# Customize major tick labels on both axes
ax.tick_params(axis='both', which='major', labelsize=17)

# Add gridlines for both major and minor ticks on the y-axis
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='black')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='black')

# Add gridlines for both axes
ax.grid(axis='both', linestyle='--', alpha=0.7, color='black')

# Set the background color of the plot area to white
ax.set_facecolor('white')

# Add a legend to the upper right corner of the plot
ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')

# Customize the color and linewidth of the plot's spines (borders)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Set the x-axis limits based on predefined minimum and maximum Eb/N0 values
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])

# Set the y-axis limits for a logarithmic scale (from 1e-5 to 1)
ax.set_ylim([1e-5, 1])

# Set the y-axis scale to logarithmic
ax.set_yscale('log')

filename = 'BER' + datetime.now().strftime('%d-%m-%Y--%H:%M:%S') + '.png'
# Create a filename based on the current date and time

# Save the figure with the updated filename
plt.savefig(filename)


