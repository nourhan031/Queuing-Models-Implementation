# import pandas as pd
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

    # prob of k cust in system
    P_k = [(1 - rho) * rho**k for k in range(11)]

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
    print(f"Probability of having k customers in the system (P_k): {P_k}")


# M/M/k

# lecture 6
def MM_infinity():
    print("MM-infinity Queuing Model")

# lecture 7
def MM1_k(): # finite storage

    print("MM1k Queuing Model")

def MM1_m(): # finite cutomer population
    print("MM1m Queuing Moddel")

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

if __name__ == "__main__":
    main()
