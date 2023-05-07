from os import listdir
from FileReader import FileParser
import numpy as np
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Pool
from time import perf_counter

# custom setting for matplot
sns.set_theme()
plt.ion()
plt.rcParams["figure.figsize"] = (12, 8)


class AntColony:
    def __init__(
        self,
        ant_count: int = 10,
        elitist_weight: float = 1.0,
        min_scaling_factor: float = 0.001,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.1,
        pheromone_deposit_weight: float = 1.0,
        initial_pheromone: float = 1.0,
        iterations: int = 100,
        plot=False,
        Search=False,
    ) -> None:
        self.ant_count = ant_count
        self.alpha = alpha
        self.beta = beta
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.initial_pheromone = initial_pheromone
        self.maxiterations = iterations
        self.global_best_tour = None
        self.global_best_distance = float("inf")
        self.plot_results = plot
        self.enable_search = Search

    def intitlialize(
        self,
        mode: str,
        distance_matrix: np.ndarray,
        capacity: int,
        demand_list: np.ndarray,
        location_list: np.ndarray,
        instance: str,
    ) -> None:
        self.start_time = perf_counter()
        self.mode = mode
        self.instance = instance
        self.maxcapacity = capacity
        self.distance_matrix = distance_matrix
        self.pheromone = dict()
        self.customer_demand = demand_list
        self.customer_id = {id: val for id, val in enumerate(location_list)}
        # remove depo from customer list
        self.customer_id.pop(0)
        # save depo location
        self.depo_location = location_list[0]

        # initialize pheromone
        for a in range(len(location_list)):
            for b in range(len(location_list)):
                if a != b:
                    self.pheromone[(min(a, b), max(a, b))] = self.initial_pheromone

        if self.mode == "ACS":
            self.solve()
        elif self.mode == "Elitist":
            self.elitist()
        elif self.mode == "MinMax":
            self.minmax()
        else:
            print(f"Unknow Aco Type Selected {self.mode}")

        if self.plot_results:
            self.plot_solution_routes()

        self.end_time = perf_counter()
        # save details to file
        with open(f"result.txt", "a") as file:
            file.write(
                f"instance: {self.instance}, time_take: {self.end_time-self.start_time:.2f}, iterations: {self.iterations},"
                f" Best Distance Found: {(round(self.global_best_distance, 2))}, Mode Used: {self.mode}, with parameters alpha: {self.alpha}, beta: {self.beta},"
                f" rho: {self.rho}, pheromone deposit weight: {self.pheromone_deposit_weight}, elitist weight: {self.elitist_weight},"
                f" ant count: {self.ant_count}, min scaling factor: {self.min_scaling_factor}, initial_pheromone: {self.initial_pheromone},"
                f" maxiterations: {self.maxiterations}\n"
            )

    def update_pheromones(self, min_pheromone: int = None, max_pheromone: int = None) -> None:
        for i in range(len(self.customer_id)):
            for j in range(len(self.customer_id)):
                if i != j:
                    self.pheromone[(min(i, j), max(i, j))] *= 1.0 - self.rho
                    if max_pheromone is not None and self.pheromone[(min(i, j), max(i, j))] > max_pheromone:
                        self.pheromone[(min(i, j), max(i, j))] = max_pheromone
                    elif min_pheromone is not None and self.pheromone[(min(i, j), max(i, j))] < min_pheromone:
                        self.pheromone[(min(i, j), max(i, j))] = min_pheromone

    def lay_pheromone(self, solution, distance_travelled, factor: float = None) -> int:
        if factor is None:
            factor = self.pheromone_deposit_weight
        pheromone_to_add = factor / distance_travelled
        for path in solution:
            for i in range(len(path) - 1):
                city1, city2 = min(path[i], path[i + 1]), max(path[i], path[i + 1])
                current_pheromone_value = self.pheromone[(city1, city2)]
                self.pheromone[(city1, city2)] = current_pheromone_value + pheromone_to_add

    def path_generation(self) -> list[list]:
        solution = list()
        customers_index = list(self.customer_id.keys())
        capacity_limit = self.maxcapacity

        while len(customers_index) > 1:
            route = list()
            # initial_location = np.random.choice(customers_index)
            initial_location = self.nearest_neighbour(customers_index)
            capacity_left = capacity_limit - self.customer_demand[initial_location]

            route.append(initial_location)
            customers_index.remove(initial_location)

            while len(customers_index) > 1:
                probability_list = self.probability_generation(customers_index, initial_location)

                next_city = np.random.choice(customers_index, p=probability_list)
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

    def nearest_neighbour(self, customers_index) -> int:
        min_dist = np.inf
        current_id = None
        for id in customers_index:
            dist = self.get_edges(0, id)
            if dist < min_dist:
                min_dist = dist
                current_id = id
        return current_id

    def probability_generation(self, customers: list, initial_location: int) -> list[float]:
        pheromones_array = np.array(
            [self.pheromone[(min(x, initial_location), max(x, initial_location))] for x in customers]
        )

        edges_array = np.array([self.get_edges(x, initial_location) for x in customers])

        probabilities = (pheromones_array**self.alpha) * ((1 / edges_array) ** self.beta)

        # to fix really small values
        EPSILON = 1e-8
        probabilities[probabilities == 0] = EPSILON
        probabilities /= np.sum(probabilities)

        return probabilities

    def get_edges(self, a, b) -> int:
        dist = self.distance_matrix[(min(a, b), max(a, b))]
        if dist != 0:
            return dist
        return 1

    def solution_cost(self, solution) -> int:
        path_cost = 0
        for path in solution:
            # add 1st location to both end and begging of path
            path_cost += self.get_edges(0, path[0])
            for id, val in enumerate(path):
                if id + 1 < len(path):
                    path_cost += self.get_edges(val, path[id + 1])
            path_cost += self.get_edges(0, path[-1])
        return path_cost

    def solve(self) -> None:
        num_iterations_without_improvement = 0
        self.iterations = None
        for id in range(1, self.maxiterations + 1):
            has_improved = False
            for _ in range(self.ant_count):
                solution = self.path_generation()
                route_distance = self.solution_cost(solution)

                # optimize route
                if self.enable_search:
                    _route = self.two_opt(solution)
                    solution = self.two_opt_inter_route_swap(_route)
                    route_distance = self.solution_cost(solution)

                if route_distance < self.global_best_distance:
                    self.global_best_distance = route_distance
                    self.global_best_tour = solution
                    self.lay_pheromone(solution=solution, distance_travelled=route_distance)
                    has_improved = True

            self.update_pheromones()

            # early exit if results dosent improve
            if not has_improved:
                num_iterations_without_improvement += 1
            else:
                num_iterations_without_improvement = 0

            if num_iterations_without_improvement >= 100:
                self.iterations = id
                break

    def elitist(self) -> None:
        num_iterations_without_improvement = 0
        self.iterations = None
        for id in range(1, self.maxiterations + 1):
            has_improved = False
            for _ in range(self.ant_count):
                solution = self.path_generation()
                route_distance = self.solution_cost(solution)

                # optimize route
                if self.enable_search:
                    _route = self.two_opt(solution)
                    solution = self.two_opt_inter_route_swap(_route)
                    route_distance = self.solution_cost(solution)

                if route_distance < self.global_best_distance:
                    self.global_best_distance = route_distance
                    self.global_best_tour = solution

                    # lay pheromone for all ants
                    self.lay_pheromone(solution=solution, distance_travelled=route_distance)
                    has_improved = True

            # lay pheromone based on elite ant
            self.lay_pheromone(
                solution=self.global_best_tour,
                distance_travelled=self.global_best_distance,
                factor=self.elitist_weight,
            )
            self.update_pheromones()

            # early exit if results dont improve
            if not has_improved:
                num_iterations_without_improvement += 1
            else:
                num_iterations_without_improvement = 0

            if num_iterations_without_improvement >= 100:
                self.iterations = id
                break

    def minmax(self) -> None:
        num_iterations_without_improvement = 0
        self.iterations = None
        for id in range(1, self.maxiterations + 1):
            iteration_best_tour = None
            iteration_best_distance = float("inf")
            has_improved = False
            # generate multiple routes
            for _ in range(self.ant_count):
                solution = self.path_generation()
                route_distance = self.solution_cost(solution)

                # optimize route
                if self.enable_search:
                    _route = self.two_opt(solution)
                    solution = self.two_opt_inter_route_swap(_route)
                    route_distance = self.solution_cost(solution)

                if route_distance < iteration_best_distance:
                    iteration_best_distance = route_distance
                    iteration_best_tour = solution

            if iteration_best_distance < self.global_best_distance:
                self.global_best_tour = iteration_best_tour
                self.global_best_distance = iteration_best_distance
                has_improved = True

            self.lay_pheromone(self.global_best_tour, self.global_best_distance)
            max_pheromone = self.pheromone_deposit_weight / (self.global_best_distance * (1 - self.rho))
            min_pheromone = max_pheromone * self.min_scaling_factor

            self.update_pheromones(min_pheromone, max_pheromone)

            # early exit if results dont improve
            if not has_improved:
                num_iterations_without_improvement += 1
            else:
                num_iterations_without_improvement = 0

            if num_iterations_without_improvement >= 100:
                self.iterations = id
                break

    def plot_solution_routes(self) -> None:
        # plot depo
        plt.plot(self.depo_location[0], self.depo_location[1], "o", color="black", markersize=10)
        # plot customers
        for id, val in self.customer_id.items():
            plt.plot(val[0], val[1], "o", color="blue", markersize=7)

        for path in self.global_best_tour:
            path_cords = [self.depo_location] + [self.customer_id[idx] for idx in path] + [self.depo_location]
            x = [cords[0] for cords in path_cords]
            y = [cords[1] for cords in path_cords]
            plt.plot(x, y, "-->", linewidth=2)

        plt.title(f"AntSystem Routing solution with total distance {round(self.global_best_distance, 2)}")
        plt.show(block=False)
        plt.pause(3)
        plt.close("all")


def setup_instance(file: str, params: dict) -> None:
    # read input file
    max_capacity, locations, demands, distance_matrix = FileParser(file).parse_input()

    for i in range(5):
        np.random.seed(224961 + i * 100)
        solver = AntColony(**params)
        solver.intitlialize(
            mode="ACS",
            capacity=max_capacity,
            distance_matrix=distance_matrix,
            demand_list=demands,
            location_list=locations,
            instance=file,
        )

        solver.intitlialize(
            mode="Elitist",
            capacity=max_capacity,
            distance_matrix=distance_matrix,
            demand_list=demands,
            location_list=locations,
            instance=file,
        )

        solver.intitlialize(
            mode="MinMax",
            capacity=max_capacity,
            distance_matrix=distance_matrix,
            demand_list=demands,
            location_list=locations,
            instance=file,
        )


def run_instances(files: list[str], params: dict) -> None:
    pool = Pool()
    for file in files:
        if file.endswith(".vrp"):
            pool.apply_async(setup_instance, args=(join(filesDir, file), params))
    pool.close()
    pool.join()


if __name__ == "__main__":
    filesDir = "Set_A_Augerat_1995"
    filenames = listdir(filesDir)
    params = {
        "ant_count": 20,
        "elitist_weight": 2.0,
        "min_scaling_factor": 0.001,
        "alpha": 1.0,
        "beta": 4.0,
        "rho": 0.50,
        "pheromone_deposit_weight": 0.1,
        "initial_pheromone": 0.01,
        "iterations": 1000,
        "plot": False,  # set it to false to disable plots
        "Search": False,
    }
    print("Starting............")
    run_instances(filenames, params)
    print("Ending............")
