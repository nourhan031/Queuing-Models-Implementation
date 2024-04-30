import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# lecture 5
def MM_1(prob_of_IAT, prob_of_ST, pop_size):
    model_type = "M/M/1"

    # generate IAT using exponential distribution
    inter_arrival_time = list(np.random.exponential(scale= prob_of_IAT, size=pop_size))
    # generate random ST for each cust
    service_time = list(np.random.exponential(scale= prob_of_ST, size=pop_size))



    avg_of_IAT = sum(inter_arrival_time)/pop_size
    arrival_rate = 1 / avg_of_IAT


    avg_of_ST = sum(service_time)/pop_size
    # service_rate
    service_rate = 1 / avg_of_ST

    # system utilization
    rho = arrival_rate / service_rate
    # avg wt cust spends in queue
    W_q = arrival_rate / (service_rate * (service_rate - arrival_rate))

    # avg num of cust in system
    L = arrival_rate / (service_rate - arrival_rate)

    # avg num of cust in queue
    L_q = arrival_rate ** 2 / (service_rate * (service_rate - arrival_rate))

    # avg wt a cust spends in system
    W = 1 / (service_rate - arrival_rate)

    # prob that all servers are busy
    P_w = arrival_rate / service_rate

    # prob of 0 cust in system
    P_0 = 1 - P_w

    n = input("Enter no of customers you want to get probability of: ")
    P_n = (1 - rho) * (rho ** n)

    print(f"Model Type: {model_type}")
    print(f"Lambda: {arrival_rate}")
    print(f"Mu: {service_rate}")
    print(f"System utilization (rho): {rho}")
    print(f"Average waiting time in the queue (W_q): {W_q}")
    print(f"Average number of customers in the system (L): {L}")
    print(f"Average number of customers in the queue (L_q): {L_q}")
    print(f"Average waiting time in the system (W): {W}")
    print(f"Probability that all servers are busy (P_w): {P_w}")
    print(f"Probability of having no customers in the system (P_0): {P_0}")
    print(f"Probability of having n customers in the system (P_n): {P_n}")

# M/M/k
def MM_k(prob_of_IAT, prob_of_ST, num_servers, pop_size):
    # M: Poisson arrival process
    # M: exponential service time distribution
    # k: number of servers
    # Discipline: FCFS

    # type of queuing model
    model_type = "M/M/k"
    # generate IAT using exponential distribution
    inter_arrival_time = list(np.random.exponential(scale= prob_of_IAT, size=pop_size))
    # generate random ST for each cust
    service_time = list(np.random.exponential(scale= prob_of_ST, size=pop_size))


    avg_of_IAT = sum(inter_arrival_time)/pop_size
    arrival_rate = 1 / avg_of_IAT

    avg_of_ST = sum(service_time)/pop_size
    # service_rate
    service_rate = 1 / avg_of_ST
    k = num_servers

    # system utilization
    rho = arrival_rate / (num_servers * service_rate)

    # prob of 0 cust in system
    P_0_denominator_term1 = sum([(1 / math.factorial(n)) * (arrival_rate / service_rate) ** n for n in range(k-1)])
    P_0_denominator_term2 = (1 / math.factorial(k)) * ((arrival_rate / service_rate) ** k) * ((k * service_rate) / (k * service_rate - arrival_rate))
    P_0 = 1 / (P_0_denominator_term1 + P_0_denominator_term2)

    # avg wt a cust spends in system
    W_numerator = ((arrival_rate / service_rate) ** k) * service_rate
    W_denominator = (math.factorial(k - 1)) * ((k * service_rate) - arrival_rate) ** 2
    W = ((W_numerator / W_denominator) * P_0) + (1 / service_rate)

    # avg wt cust spends in queue
    W_q = (W_numerator / W_denominator) * P_0

    # avg num of cust in system
    L_numerator = ((arrival_rate / service_rate) ** k) * arrival_rate * service_rate
    L_denominator = math.factorial(k-1) * ((k * service_rate) - arrival_rate) ** 2
    L = ((L_numerator / L_denominator) * P_0) + (arrival_rate / service_rate)

    # avg num of cust in queue
    L_q = (L_numerator / L_denominator) * P_0

    # prob that all servers are busy
    P_w = (1 / (math.factorial(k))) * (arrival_rate / service_rate) ** k * ((k * service_rate) / ((k * service_rate) - arrival_rate)) * P_0

    # Get user input for system capacity
    n = int(input("Enter the no of customers you want to get probability of: "))

    # prob of n cust in system

    if n <= k:
        P_n = (((arrival_rate / service_rate) ** n) / math.factorial(n)) * P_0
    else:
        P_n = (((arrival_rate / service_rate) ** n) / (math.factorial(k) * k ** (n - k))) * P_0



    print(f"Model Type: {model_type}")
    print(f"Lambda: {arrival_rate}")
    print(f"Mu: {service_rate}")
    print(f"System utilization (rho): {rho}")
    print(f"Average waiting time in the queue (W_q): {W_q}")
    print(f"Average number of customers in the system (L): {L}")
    print(f"Average number of customers in the queue (L_q): {L_q}")
    print(f"Average waiting time in the system (W): {W}")
    print(f"Probability that all servers are busy (P_w): {P_w}")
    print(f"Probability of having no customers in the system (P_0): {P_0}")
    print(f"Probability of having n customers in the system (P_n): {P_n}")

# lecture 6
def MM_infinity(IAT, service_rate, n):
    # M: Memoryless arrivals -> poisson process
    # M: Memoryless service times -> Exponential process
    # infinity -> infinte no. of servers
    # lambdas are equal for all states
    # mu = number of state it's coming from * mu

    arrival_rate = 1 / IAT

    # mu_k and lambda_k
    lambda_k = arrival_rate

    # Calculate Poisson parameter (mean λ/μ)
    poisson_mean = arrival_rate / service_rate

    # probability of having n customers in the system
    if n > 0:
      P_n_numerator = ((arrival_rate / service_rate) ** n) * (math.exp(1) ** (arrival_rate/service_rate))
      P_n_denominator = math.factorial(n)
      P_n = P_n_numerator / P_n_denominator
    else:
        print("invalid value of servers.")

    # expected system size is the mean of the poisson distribution
    L = arrival_rate / service_rate

    # avg WT = avg service time -> also has the same dist as it does (Exponential)
    W = 1 / service_rate

    print("MM-infinity Queuing Model")

# lecture 7
def MM1_k(): # finite storage

    print("MM1k Queuing Model")

def MM1_m(arrival_rate, service_rate, pop_size): # finite cutomer population
    # 1: single server
    # service time follows an exponential distribution with mean 1/mu
    # m: total potential customers

    model_type = "M/M/1//m"
    # M = pop_size
    # mu_k and lambda_k
    mu_k = service_rate
    """
    if 0 <= k <= pop_size:
       lambda_k = lambda(pop_size - k)
       
    else:
       lambda_k = 0
    """

    # utilization

    # prob of 0 cust in system
    P_0_denominator = sum(
        [(arrival_rate / service_rate) ** k * math.factorial(pop_size) / math.factorial(pop_size - k) for
         k in range(pop_size + 1)])
    P_0 = 1 / P_0_denominator

    # prob of k cust in system
    P_k = [(P_0 * (arrival_rate / service_rate) ** k * math.factorial(pop_size) / math.factorial(
        pop_size - k)) for k in range(pop_size + 1)]

    # avg num of cust in system
    L = sum([k * P_k[k] for k in range(pop_size + 1)])

def main():
    # Get user input for all characteristics of the queuing model
    prob_of_IAT = float(input("Enter the probability distribution of interarrival time : "))
    prob_of_ST = float(input("Enter the probability distribution of service time: "))
    num_servers = int(input("Enter the number of servers: "))
    system_capacity = int(input("Enter the system capacity (queue size): "))
    pop_size = int(input("Enter the population size: "))
    queue_discipline = input("Enter the queuing discipline (e.g., FIFO, FCFS): ")


    """
    IAT EXPONENTIAL:
       Service Time EXPONENTIAL:
         SINGLE SERVER:
           FINITE queue length:
              M/M/1/k
           ELSE (INFINITE queue length):
             FINITE customer population:
               M/M/1//m
             ELSE (INFINITE customer population):
               M/M/1
         ELSE (MORE THAN A SINGLE SERVER):
             FINITE queue length:
                M/M/K
    """
# metba'y ne3raf el condition bta3 M/M/infinity

    if num_servers == 1 and queue_discipline == "FCFS":
        # check for pop capacity, if finite -> MM1, else: MM1m
        MM_1(prob_of_IAT, prob_of_ST, pop_size)

    elif num_servers > 1 and queue_discipline == "FCFS":
        MM_k(prob_of_IAT, prob_of_ST, num_servers,pop_size)

if __name__ == "__main__":
    main()
