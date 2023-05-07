from os import listdir
from FileReader import FileParser
from time import perf_counter

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class LocalSearch:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        capacity: int,
        demand_list: np.ndarray,
        location_list: np.ndarray,
        instance: str,
        iterations: int,
        plot: bool,
    ) -> None:
        self.start_time = perf_counter()
        self.max_iter = iterations
        self.instance = instance
        self.maxcapacity = capacity
        self.distance_matrix = distance_matrix
        self.customer_demand = demand_list
        self.plot_results = plot
        self.customer_id = {id: val for id, val in enumerate(location_list)}
        # remove depo from customer list
        self.customer_id.pop(0)
        # save depo location
        self.depo_location = location_list[0]
        self.best_solution = None
        self.best_solution_cost = float("inf")

    def generate_initial_solution(self) -> list[list]:
        solution = list()
        customers_index = list(self.customer_id.keys())
        capacity_limit = self.maxcapacity

        while len(customers_index) > 1:
            route = list()
            initial_location = np.random.choice(customers_index)
            capacity_left = capacity_limit - self.customer_demand[initial_location]
            route.append(initial_location)
            customers_index.remove(initial_location)

            while len(customers_index) > 1:
                next_city = np.random.choice(customers_index)
                capacity_left = capacity_left - self.customer_demand[next_city]

                # update initial location
                initial_location = next_city
                if capacity_left > 0:
                    route.append(next_city)
                    customers_index.remove(next_city)
                else:
                    break
            if len(customers_index) == 1:
                route.append(customers_index[0])
                customers_index.remove(customers_index[0])
            solution.append(route)
        return solution

    def vns(self, route) -> list:
        # Initialize Route
        self.best_solution = route
        self.best_solution_cost = self.solution_cost(route)

        # Define neighborhood structures
        neighborhoods = [
            self.SwapNeighborhood,
            self.ReverseNeighborhood,
            self.InsertionNeighborhood,
        ]

        # Initialize variables for early exit
        stagnation_count = 0
        early_exit_iter = 0

        for iteration in range(self.max_iter):
            # Select a random neighborhood structure
            selection = np.random.choice(neighborhoods)

            # Apply the selected neighborhood structure to the current solution
            new_solution = selection()

            # check for empty list
            new_solution = list(filter(lambda lst: len(lst) != 0, new_solution))

            # Improve the new solution
            new_solution = self.two_opt(new_solution)
            new_solution_cost = self.solution_cost(new_solution)

            # Compare objective function values
            if new_solution_cost < self.best_solution_cost:
                self.best_solution = new_solution
                self.best_solution_cost = new_solution_cost
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Early exit if there is no improvement for a certain number of iterations
            if stagnation_count >= 100:
                early_exit_iter = iteration
                break

        if early_exit_iter > 0:
            iteration = early_exit_iter
        if self.plot_results:
            self.plot_solution_routes()
        return self.best_solution_cost, self.best_solution, perf_counter() - self.start_time, iteration

    def SwapNeighborhood(self) -> list[list]:
        solution = deepcopy(self.best_solution)
        choice = 1  # np.random.randint(0, 2)
        if choice == 0:
            # Inter Swap
            # select a route at random
            i = np.random.choice(range(len(solution)))
            route_i = [] + solution[i]

            if len(route_i) < 2:
                return solution

            # Select two random positions in the route
            pos_i, pos_j = np.random.choice(range(len(route_i)), 2, replace=False)

            # Swap the customers at the selected positions
            route_i[pos_i], route_i[pos_j] = route_i[pos_j], route_i[pos_i]

            if self.check_capacity(route_i):
                # Update the solution
                solution[i] = route_i
        else:
            # Intra Swap
            # Select two different routes at random
            i, j = np.random.choice(range(len(solution)), 2, replace=False)
            route_i = [] + solution[i]
            route_j = [] + solution[j]

            # Select two different customers from the selected routes at random
            customer_i = np.random.choice(route_i)
            customer_j = np.random.choice(route_j)

            # Swap the customers between the routes
            route_i[route_i.index(customer_i)] = customer_j
            route_j[route_j.index(customer_j)] = customer_i

            # Update the solution
            if self.check_capacity(route_i) and self.check_capacity(route_j):
                solution[i] = route_i
                solution[j] = route_j

        return solution

    def ReverseNeighborhood(self) -> list[list]:
        solution = deepcopy(self.best_solution)
        choice = np.random.randint(0, 2)
        if choice == 0:
            # Inter Swap
            # Select a random route
            i = np.random.randint(len(solution))
            route_i = [] + solution[i]

            if len(route_i) < 2:
                return solution

            # Select two different edges at random
            j, k = np.random.choice(range(0, len(route_i)), 2, replace=False)

            # Reverse the segment between j and k
            if j > k:
                j, k = k, j

            route_i[j : k + 1] = reversed(route_i[j : k + 1])

            if self.check_capacity(route_i):
                # Update the solution
                solution[i] = route_i
        else:
            # Intra Swap
            # Select two random route
            i, j = np.random.choice(range(0, len(solution)), 2, replace=False)

            route_i = [] + solution[i]
            route_j = [] + solution[j]

            if len(route_i) < 2 or len(route_j) < 2:
                return solution

            # Select two different edges at random
            if len(route_j) > len(route_i):
                l, k = np.random.choice(range(0, len(route_i)), 2, replace=False)
            else:
                l, k = np.random.choice(range(0, len(route_j)), 2, replace=False)

            # Reverse the segment between j and k
            if l > k:
                l, k = k, l

            Temp = [] + route_i
            route_i[l : k + 1] = reversed(route_j[l : k + 1])
            route_j[l : k + 1] = reversed(Temp[l : k + 1])

            # Update the solution
            if self.check_capacity(route_i) and self.check_capacity(route_j):
                solution[i] = route_i
                solution[j] = route_j

        return solution

    def CheckInsertion(self, route, customer):
        # Find the cheapest insertion position for the selected customer in route i
        cheapest_cost = float("inf")
        cheapest_pos = -1

        if len(route) < 2:
            route.append(customer)
            return route

        for pos in range(1, len(route)):
            cost = (
                self.get_edges(route[pos - 1], customer)
                + self.get_edges(customer, route[pos])
                - self.get_edges(route[pos - 1], route[pos])
            )
            if cost < cheapest_cost:
                cheapest_cost = cost
                cheapest_pos = pos

        # Insert the customer into route j at the cheapest position
        if cheapest_pos >= 0:
            route.insert(cheapest_pos, customer)

        return route

    def InsertionNeighborhood(self) -> list[list]:
        solution = deepcopy(self.best_solution)
        choice = 1  # np.random.randint(0, 2)
        if choice == 0:
            # Inter Swap
            # Select a routes at random
            i = np.random.choice(range(len(solution)))
            route_i = [] + solution[i]

            if len(route_i) < 2:
                return solution

            # Select a random customer from route i
            customer = np.random.choice(route_i)
            route_i.remove(customer)

            route_i = self.CheckInsertion(route_i, customer)

            # Update the solution
            if self.check_capacity(route_i):
                solution[i] = route_i
        else:
            # Intra Swap
            # Select two different routes at random
            i, j = np.random.choice(range(len(solution)), 2, replace=False)
            route_i = [] + solution[i]
            route_j = [] + solution[j]

            # Select route betwwen two routes
            choice_route = np.random.randint(0, 2)
            if choice_route == 0:
                customer = np.random.choice(route_i)
                route_i.remove(customer)
                route_j = self.CheckInsertion(route_j, customer)
            else:
                customer = np.random.choice(route_j)
                route_j.remove(customer)
                route_i = self.CheckInsertion(route_i, customer)

            # Update the solution
            if self.check_capacity(route_i) and self.check_capacity(route_j):
                solution[i] = route_i
                solution[j] = route_j

        return solution

    def two_opt(self, solution) -> list[list]:
        improved = True
        _solution = []
        while improved:
            improved = False
            for sub_route in solution:
                for i in range(1, len(sub_route) - 1):
                    for j in range(i + 1, len(sub_route)):
                        # Check if a swap would result in a valid solution
                        if j - i == 1:
                            continue
                        new_solution = sub_route[:]
                        # Swap the order of two edges and check if the new solution is an improvement
                        new_solution[i:j] = sub_route[j - 1 : i - 1 : -1]
                        # Calculate the total distance of a given route and check capacity
                        if self.route_distance(new_solution) < self.route_distance(sub_route):
                            sub_route = new_solution
                            improved = True
                _solution.append(sub_route)
            return _solution

    def route_distance(self, route) -> int:
        distance = 0
        distance += self.get_edges(0, route[0])
        for i in range(len(route) - 1):
            distance += self.get_edges(route[i], route[i + 1])
        distance += self.get_edges(0, route[-1])
        return distance

    def route_demand(self, route) -> int:
        total_demand = 0
        for id in route:
            total_demand += self.customer_demand[id]
        return total_demand

    def check_capacity(self, route) -> bool:
        capacity = 0
        for node in route:
            capacity += self.customer_demand[node]
            if capacity > self.maxcapacity:
                return False
        return True

    def get_edges(self, a, b) -> int:
        dist = self.distance_matrix[(min(a, b), max(a, b))]
        if dist != 0:
            return dist
        return 1

    def solution_cost(self, solution) -> int:
        path_cost = 0
        for path in solution:
            # add 1st location to both end and beginning of path
            path_cost += self.get_edges(0, path[0])
            for id, val in enumerate(path):
                if id + 1 < len(path):
                    path_cost += self.get_edges(val, path[id + 1])
            path_cost += self.get_edges(0, path[-1])
        return path_cost

    def plot_solution_routes(self) -> None:
        # plot depo
        plt.plot(self.depo_location[0], self.depo_location[1], "o", color="black", markersize=10)
        # plot customers
        for id, val in self.customer_id.items():
            plt.plot(val[0], val[1], "o", color="blue", markersize=7)

        for path in self.best_solution:
            path_cords = [self.depo_location] + [self.customer_id[idx] for idx in path] + [self.depo_location]
            x = [cords[0] for cords in path_cords]
            y = [cords[1] for cords in path_cords]
            plt.plot(x, y, "-->", linewidth=2)

        plt.title(f"Routing solution with total distance {round(self.best_solution_cost, 2)}")
        plt.show(block=False)
        plt.pause(3)
        plt.close("all")


if __name__ == "__main__":
    filesDir = "Set_A_Augerat_1995"
    print("Starting............")
    for filename in listdir(filesDir):
        file = filesDir + "/" + filename
        if file.endswith(".vrp"):
            # read input file
            max_capacity, locations, demands, distance_matrix = FileParser(file).parse_input()
            for i in range(5):
                np.random.seed(224961 + i * 100)
                # print(f"\nRun: {i}")
                Local_Search = LocalSearch(
                    capacity=max_capacity,
                    location_list=locations,
                    demand_list=demands,
                    distance_matrix=distance_matrix,
                    instance=filename,
                    iterations=1000,
                    plot=False,  # set it to false to disable plots
                )
                route = Local_Search.generate_initial_solution()
                cost, path, time, iterations = Local_Search.vns(route)
                print(f"Instance: {file} | Cost: {round(cost,2)} | Iterations: {iterations} | Time: {time:.2f} sec")
                with open("LS.txt", "a") as f:
                    f.writelines(
                        f"Instance: {file} | Cost: {round(cost,2)} | Iterations: {iterations} | Time: {time:.2f} sec\n"
                    )

    print("Ending............")
