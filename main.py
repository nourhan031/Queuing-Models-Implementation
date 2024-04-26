# import pandas as pd
import math

# lecture 5
def MM_1(arrival_rate, service_rate):
    # M: Poisson arrival process
    # M: exponential service time distribution
    # 1: one server
    # Discipline: FCFS

    # type of queuing model
    model_type = "M/M/1"

    # mu_k and lambda_k
    mu_k = service_rate
    lambda_k = arrival_rate

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
    P_0 = 1 - (arrival_rate / service_rate)

    n = int(input("Enter num of customers in system: "))
    # prob of n cust in system
    P_n = (1 - rho) * (rho ** n)

    # ha3mel function feeha el print statements di w neb'a n call it inside kol wahda instead of having to write them every single time
    print(f"Model Type: {model_type}")
    print(f"Lambda_k: {lambda_k}")
    print(f"Mu_k: {mu_k}")
    print(f"System utilization (rho): {rho}")
    print(f"Average waiting time in the queue (W_q): {W_q}")
    print(f"Average number of customers in the system (L): {L}")
    print(f"Average number of customers in the queue (L_q): {L_q}")
    print(f"Average waiting time in the system (W): {W}")
    print(f"Probability that all servers are busy (P_w): {P_w}")
    print(f"Probability of having no customers in the system (P_0): {P_0}")
    print(f"Probability of having n customers in the system (P_n): {P_n}")


# M/M/k
def MM_k(arrival_rate, service_rate, num_servers):
    # M: Poisson arrival process
    # M: exponential service time distribution
    # k: number of servers
    # Discipline: FCFS

    # type of queuing model
    model_type = "M/M/k"

    # arrival rate (lambda) and service rate (mu) are the same for all customers and servers
    mu_k = service_rate
    lambda_k = arrival_rate
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
    system_capacity = int(input("Enter the system capacity: "))

    P_n_list = []

    # prob of n cust in system
    for n in range(0, system_capacity + 1):
        if n <= k:
            P_n = (((arrival_rate / service_rate) ** n) / math.factorial(n)) * P_0
        else:
            P_n = (((arrival_rate / service_rate) ** n) / (math.factorial(k) * k ** (n - k))) * P_0

        P_n_list.append(P_n)

    print(f"Model Type: {model_type}")
    print(f"Lambda_k: {lambda_k}")
    print(f"Mu_k: {mu_k}")
    print(f"System utilization (rho): {rho}")
    print(f"Average waiting time in the queue (W_q): {W_q}")
    print(f"Average number of customers in the system (L): {L}")
    print(f"Average number of customers in the queue (L_q): {L_q}")
    print(f"Average waiting time in the system (W): {W}")
    print(f"Probability that all servers are busy (P_w): {P_w}")
    print(f"Probability of having no customers in the system (P_0): {P_0}")
    print(f"Probability of having n customers in the system (P_n): {P_n_list}")

# lecture 6
def MM_infinity():
    print("MM-infinity Queuing Model")

# lecture 7
def MM1_k(): # finite storage

    print("MM1k Queuing Model")

def MM1_m(): # finite cutomer population
    # 1: single server
    # service time follows an exponential distribution with mean 1/mu
    # m: total potential customers


    print("MM1m Queuing Model")

def main():
    # Get user input for all characteristics of the queuing model
    arrival_rate = float(input("Enter the probability distribution of arrivals (lambda): "))
    service_rate = float(input("Enter the probability distribution of service time (mu): "))
    num_servers = int(input("Enter the number of servers: "))
    system_capacity = int(input("Enter the system capacity: "))
    pop_size = int(input("Enter the population size: "))
    queue_discipline = input("Enter the queuing discipline (e.g., FIFO, FCFS): ")

    if num_servers == 1 and queue_discipline == "FCFS":
        MM_1(arrival_rate, service_rate)

    elif num_servers > 1:
        MM_k(arrival_rate, service_rate, num_servers)

if __name__ == "__main__":
    main()
