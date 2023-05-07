import re
from itertools import chain
from scipy.spatial import distance
import numpy as np


class FileParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.capacity = 0
        self.optimal = 0
        self.demand = []
        self.distance_matrix = []
        self.locations = []
        self.solution_path = []

    def parse_input(self) -> list:
        """Parses input files of .vrp type"""
        with open(self.file_path, "r") as file:
            content = file.read()

            # Extract capacity, locations and demand from the input file
            capacity_regex = re.compile(r"^\s?CAPACITY\s?: (\d+)\s?$", re.MULTILINE)
            self.capacity = int(capacity_regex.search(content).group(1))
            locations_regex = re.compile(r"^\s?(\d+) (\d+) (\d+)\s?$", re.MULTILINE)
            self.locations = [list(map(int, loc[1:])) for loc in locations_regex.findall(content)]
            demand_regex = re.compile(r"^\s?(\d+) (\d+)\s?$", re.MULTILINE)
            self.demand = np.array(
                list(chain.from_iterable([list(map(int, val[1:])) for val in demand_regex.findall(content)]))
            )

            # Compute distance matrix between locations
            self.distance_matrix = distance.cdist(self.locations, self.locations, "euclidean")

        return [self.capacity, self.locations, self.demand, self.distance_matrix]

    def parse_solution(self) -> list:
        """Parses output files of .sol type"""
        with open(self.file_path, "r") as file:
            content = file.read()

            # Extract optimal cost and solution paths from the output file
            cost_regex = re.compile(r"Cost (\d+)", re.MULTILINE)
            self.optimal_cost = int(cost_regex.search(content).group(1))
            path_regex = re.compile(r"^\s?Route #(\d+)\s?:\s?(.*)\s?$", re.MULTILINE)
            self.solution_path = [
                [int(vertex) for vertex in unparsed[1].split()] for unparsed in path_regex.findall(content)
            ]

        return self.optimal_cost
