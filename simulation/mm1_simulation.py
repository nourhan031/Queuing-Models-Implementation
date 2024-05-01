import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

def MM_1(prob_of_IAT, prob_of_ST, pop_size):
    # generate IAT using exponential distribution
    inter_arrival_time = list(np.random.exponential(scale= prob_of_IAT, size=pop_size))
    # generate random ST for each cust
    service_time = list(np.random.exponential(scale= prob_of_ST, size=pop_size))

    avg_of_IAT = sum(inter_arrival_time)/pop_size
    arrival_rate = 1 / avg_of_IAT

    avg_of_ST = sum(service_time)/pop_size
    service_rate = 1 / avg_of_ST

    # system utilization
    rho = arrival_rate / service_rate
    if rho >= 1:
        rho = 0.9

    # avg num of cust in queue
    L_q = rho ** 2 / (1 - rho)

    # avg wt cust spends in queue
    W_q = L_q / arrival_rate

    # avg num of cust in system
    L = rho / (1 - rho)

    # avg wt a cust spends in system
    W = 1 / (service_rate - arrival_rate)

    # prob that all servers are busy
    P_w = arrival_rate / service_rate

    # prob of 0 cust in system
    P_0 = 1 - rho

    # n = int(input("Enter the number of customers for which you want to calculate the probability: "))
    P_n = (1 - rho) * (rho ** n)

    return [arrival_rate, service_rate, rho, W_q, L, L_q, W, P_w, P_0, P_n]

# prob_of_IAT = 3
# prob_of_ST = 2

pop_size = 1000
n = 10

def generate_random_probabilities():
    # Randomly select from a broader range using uniform distribution
    prob_of_IAT = np.random.uniform(1, 10)  # Scale is arbitrary but broad
    prob_of_ST = np.random.uniform(1, 10)  # Ditto
    return prob_of_IAT, prob_of_ST

# Run the simulation 500 times
num_simulations = 500
results = []
for _ in range(num_simulations):
    prob_of_IAT, prob_of_ST = generate_random_probabilities()
    results.append(MM_1(prob_of_IAT, prob_of_ST, pop_size))

# Convert the results into a DataFrame
df = pd.DataFrame(results, columns=['Lambda', 'Mu', 'rho', 'W_q', 'L', 'L_q', 'W', 'P_w', 'P_0', 'P_n'])

numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].mean())

# Create a figure and a set of subplots
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

# Remove the extra subplot
fig.delaxes(axs[2,3])

# Plot histograms for each column
for ax, column in zip(axs.flatten(), df.columns):
    ax.hist(df[column], bins=50, color=np.random.rand(3,), edgecolor='black')
    ax.set_title(column)
    # ax.set_xlabel(column)
    ax.set_ylabel("time")

# Display the plot
plt.tight_layout()
plt.show()