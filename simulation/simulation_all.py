import math
import numpy as np

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

    n = int(input("Enter the number of customers for which you want to calculate the probability: "))
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
    model_type = "M/M/k"

    # generate IAT using exponential distribution
    inter_arrival_time = list(np.random.exponential(scale= prob_of_IAT, size=pop_size))
    # generate random ST for each cust
    service_time = list(np.random.exponential(scale= prob_of_ST, size=pop_size))

    avg_of_IAT = sum(inter_arrival_time)/pop_size
    arrival_rate = 1 / avg_of_IAT

    avg_of_ST = sum(service_time)/pop_size
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
    n = int(input("Enter the number of customers for which you want to calculate the probability: "))

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
def MM_infinity(prob_of_IAT, prob_of_ST, pop_size):
    model_type = "M/M/infinity"

    # generate IAT using exponential distribution
    inter_arrival_time = list(np.random.exponential(scale=prob_of_IAT, size=pop_size))
    # generate random ST for each cust
    service_time = list(np.random.exponential(scale=prob_of_ST, size=pop_size))

    avg_of_IAT = sum(inter_arrival_time) / pop_size
    arrival_rate = 1 / avg_of_IAT

    avg_of_ST = sum(service_time) / pop_size
    service_rate = 1 / avg_of_ST

    # system utilization
    rho = arrival_rate / service_rate

    n = int(input("Enter the number of customers for which you want to calculate the probability: "))

    # probability of having n customers in the system
    P_n_numerator = ((arrival_rate / service_rate) ** n) * (math.exp(1) ** (arrival_rate/service_rate))
    P_n_denominator = math.factorial(n)
    P_n = P_n_numerator / P_n_denominator

    # expected system size is the mean of the poisson distribution
    L = arrival_rate / service_rate

    # avg WT = avg service time -> also has the same dist as it does (Exponential)
    W = 1 / service_rate

    Mu_n = n * service_rate

    # avg WT in queue is always 0 bc there are infinite servers so no cust has to wait
    W_q = 0

    # same goes to L_q
    L_q = 0

    # prob that all servers are busy is undefined bc there are infinte servers
    P_w = 'undefined'

    # prob of having no cust in system is the complement of the sum of the probabilities of having n customers for n=1 to infinity
    P_0 = 1 - sum([(arrival_rate / service_rate) ** i / math.factorial(i) for i in range(1, pop_size+1)])

    print(f"Model Type: {model_type}")
    print(f"Lambda: {arrival_rate}")
    print(f"Mu_k: {Mu_n}")
    print(f"System utilization (rho): {rho}")
    print(f"Average waiting time in the queue (W_q): {W_q}")
    print(f"Average number of customers in the system (L): {L}")
    print(f"Average number of customers in the queue (L_q): {L_q}")
    print(f"Average waiting time in the system (W): {W}")
    print(f"Probability that all servers are busy (P_w): {P_w}")
    print(f"Probability of having no customers in the system (P_0): {P_0}")
    print(f"Probability of having n customers in the system (P_n): {P_n}")

# lecture 7
def MM1_k(prob_of_IAT, prob_of_ST, num_servers, pop_size, system_capacity): # finite system capacity
    model_type = "M/M/1/k"

    # generate IAT using exponential distribution
    inter_arrival_time = list(np.random.exponential(scale=prob_of_IAT, size=pop_size))
    # generate random ST for each cust
    service_time = list(np.random.exponential(scale=prob_of_ST, size=pop_size))

    avg_of_IAT = sum(inter_arrival_time) / pop_size
    # arrival_rate = 1 / avg_of_IAT

    avg_of_ST = sum(service_time) / pop_size
    service_rate = 1 / avg_of_ST

    k = int(input("Enter the number of customers for which you want to calculate the probability: "))
    if k < system_capacity:
        arrival_rate = 1 / avg_of_IAT
    else:
        arrival_rate = 0

    # system utilization
    rho = arrival_rate / service_rate

    # avg wt cust spends in queue
    W_q = arrival_rate / (service_rate * (service_rate - arrival_rate))

    # avg wt a cust spends in system
    W = 1 / (service_rate - arrival_rate)

    # avg num of cust in system
    L = arrival_rate * W

    # avg num of cust in queue
    L_q = arrival_rate * W_q

    # prob that all servers are busy
    P_w = arrival_rate / service_rate

    # prob of 0 cust in system
    P_0 = (1 - rho) / (1 - (rho) ** (system_capacity+1))
    # P_0_denominator = 1 + sum(((arrival_rate / service_rate) ** k) / math.factorial(k) for k in range(k-1))
    # P_0 = 1 / P_0_denominator

    # prob of having n customers in the system
    if 0 <= k < system_capacity:
        P_n = P_0 * (arrival_rate / service_rate) ** k
    else:
        P_n = 0

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


def MM1_m(prob_of_IAT, prob_of_ST, pop_size): # finite customer population
    model_type = "M/M/1//m"

    # generate IAT using exponential distribution
    inter_arrival_time = list(np.random.exponential(scale=prob_of_IAT, size=pop_size))
    # generate random ST for each cust
    service_time = list(np.random.exponential(scale=prob_of_ST, size=pop_size))

    avg_of_IAT = sum(inter_arrival_time) / pop_size
    arrival_rate = 1 / avg_of_IAT

    avg_of_ST = sum(service_time) / pop_size
    service_rate = 1 / avg_of_ST

    m = int(input("Enter the number of pop in the system: "))
    n = int(input("Enter the number of customers for which you want to calculate the probability: "))

    # prob of 0 cust in system
    P_0_denominator = sum((math.factorial(m) / (math.factorial(m - n))) * ((arrival_rate / service_rate) ** n))
    P_0 = 1 / P_0_denominator

    # system utilization
    rho = 1 - P_0

    # prob of having n customers in the system
    P_n = (math.factorial(m) / (math.factorial(m - n))) * ((arrival_rate / service_rate) ** n) * P_0

    # avg num of cust in system
    L = m - (service_rate / arrival_rate) * (1 - P_0)

    # avg num of cust in queue
    L_q = m - ((arrival_rate + service_rate) / arrival_rate) * (1 - P_0)

    # prob of waiting in queue
    W_q = L_q / (arrival_rate * (m - L))

    # prob of k cust in system
    P_k = [(P_0 * (arrival_rate / service_rate) ** k * math.factorial(pop_size) / math.factorial(
        pop_size - k)) for k in range(pop_size + 1)]

    # avg num of cust in system
    L = sum([k * P_k[k] for k in range(pop_size + 1)])

    # prob that all servers are busy
    P_w = 1 - P_0

    # avg WT in queue
    W_q = L_q / (arrival_rate * (m - L))

    # avg WT in system
    W = L / (arrival_rate * (m - L))

def main():
    # Get user input for all characteristics of the queuing model
    prob_of_IAT = float(input("Enter the probability distribution of interarrival time : "))
    prob_of_ST = float(input("Enter the probability distribution of service time: "))

    num_servers = input("Enter the number of servers: ")
    if num_servers.lower() == "infinite":
        num_servers = float('inf')
    else:
        num_servers = int(num_servers)

    system_capacity = input("Enter the system capacity (queue size): ")
    if system_capacity.lower() == "infinite":
        system_capacity = float('inf')
    else:
        system_capacity = int(system_capacity)

    pop_size = int(input("Enter the population size: "))
    queue_discipline = input("Enter the queuing discipline (e.g., FIFO, FCFS): ")

    if num_servers == 1 and queue_discipline == "FCFS":
        # check for pop capacity, if finite -> MM1, else: MM1m
        MM_1(prob_of_IAT, prob_of_ST, pop_size)

    elif num_servers == 1 and queue_discipline == "FIFO":
        MM1_k(prob_of_IAT, prob_of_ST, num_servers, pop_size, system_capacity)

    # msh beyodkhol el function aslun?????
    elif num_servers == "infinite": # single queue, infinite num of servers
        # some other condition
        MM_infinity(prob_of_IAT, prob_of_ST, pop_size)

    # msh beyodkhol el function aslun?????
    elif num_servers == 1 and system_capacity == "infinite": # infinite queue length
        # some other condition
        MM1_m(prob_of_IAT, prob_of_ST, pop_size)

    elif num_servers > 1 and queue_discipline == "FCFS":
        MM_k(prob_of_IAT, prob_of_ST, num_servers, pop_size)


    # if num_servers == 1:
    #     if queue_discipline == 'FCFS' or 'fcfs':
    #         MM_1(prob_of_IAT, prob_of_ST, pop_size)
    #     elif queue_discipline == 'FIFO' or 'fifo':
    #         MM1_k(prob_of_IAT, prob_of_ST, num_servers, pop_size, system_capacity)
    #     elif num_servers == 1 and system_capacity == ('unlimited' or 'infinite'):
    #         MM_infinity(prob_of_IAT, prob_of_ST, pop_size)
    #     elif num_servers == 1 and pop_size == ('unlimited' or 'infinite'):
    #         MM1_m(prob_of_IAT, prob_of_ST, pop_size)
    #
    # elif num_servers > 1:
    #     MM_k(prob_of_IAT, prob_of_ST, num_servers,pop_size)


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



if __name__ == "__main__":
    main()
