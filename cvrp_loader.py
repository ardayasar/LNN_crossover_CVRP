"""
cvrp_loader.py

This module loads CVRP instances (both symmetric and asymmetric) from .vrp, .dat,
and optionally .sol files (for best-known solution).

It provides:
- A CVRPInstance class holding all relevant data: name, #customers, capacity, demands, distance matrix, BKS.
- Functions to parse:
  - VRP format (e.g., "E-n22-k4.vrp" from CVRPLIB style).
  - DAT format (e.g., "A034-02f.dat" from Fischetti, Toth, and Vigo style).
  - SOL files to extract best-known solution cost (if present).

Finally, load_all_instances() collects 16 benchmark instances:
  - 8 Symmetric CVRP from the SCVRP folder
  - 8 Asymmetric CVRP from the ACVRP folder
and returns a list of CVRPInstance objects.
"""

import os
import numpy as np

class CVRPInstance:
    """
    Represents a single CVRP instance.
    Attributes:
      - name: Instance name.
      - num_customers: Number of customers (excluding depot).
      - vehicle_capacity: Capacity of each vehicle.
      - demands: List of demands (index 0 is depot).
      - distance_matrix: NxN matrix of distances (N = num_customers + 1).
      - best_known_solution: Reference cost from .sol file or manual input.
    """
    def __init__(self, name, num_customers, vehicle_capacity, demands, distance_matrix, best_known_solution):
        self.name = name
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.demands = demands
        self.distance_matrix = distance_matrix
        self.best_known_solution = best_known_solution

def _parse_depot(lines):
    """
    Return the 1-based depot node id from DEPOT_SECTION, defaulting to 1.

    TSPLIB lists depots one per line after the header, terminated by -1 or EOF.
    Only the first is used; these benchmarks are all single-depot.
    """
    in_section = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("DEPOT_SECTION"):
            in_section = True
            continue
        if in_section:
            if up.startswith("EOF") or s.startswith("-1"):
                break
            if any(ch.isalpha() for ch in s):     # next section header
                break
            try:
                value = int(s.split()[0])
            except ValueError:
                break
            if value > 0:
                return value
            break
    return 1


def _move_depot_to_front(depot, demands, distance_matrix, source=""):
    """
    Put the depot at index 0.

    GA.py assumes node 0 is the depot throughout (distance_matrix[0][...],
    demands[0]), but benchmark files do not all agree. The CVRPLIB E-n*
    instances name node 1, which already maps to index 0; the Fischetti-Toth-
    Vigo ACVRP instances name node N, i.e. the *last* node. Without this
    permutation those instances load with customer 1 acting as the depot and
    the real depot treated as a customer.

    Both axes of the distance matrix are permuted, which matters for the
    asymmetric set where the matrix is not its own transpose.
    """
    n = len(demands)
    k = depot - 1                                  # DEPOT_SECTION is 1-based
    if not 0 <= k < n:
        raise ValueError(
            f"{source}: DEPOT_SECTION names node {depot}, outside 1..{n}")

    if k != 0:
        order = [k] + [i for i in range(n) if i != k]
        demands = [demands[i] for i in order]
        distance_matrix = distance_matrix[np.ix_(order, order)]

    if demands[0] != 0:
        raise ValueError(
            f"{source}: depot (node {depot}) has demand {demands[0]}, expected 0")

    return demands, distance_matrix


def load_instance_from_vrp(filepath, name, best_known_solution):
    """
    Loads a .vrp file in standard CVRPLIB style.
    Reads CAPACITY, DEMAND_SECTION, NODE_COORD_SECTION, and DEPOT_SECTION,
    then builds a Euclidean distance matrix.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    capacity = None
    demands = []
    coords = []
    section = None

    for line in lines:
        line = line.strip()
        if "CAPACITY" in line.upper():
            parts = line.split()
            capacity = int(parts[-1])
        elif "DEMAND_SECTION" in line.upper():
            section = "DEMAND"
            continue
        elif "NODE_COORD_SECTION" in line.upper():
            section = "COORD"
            continue
        elif "DEPOT_SECTION" in line.upper():
            section = "DEPOT"
            continue
        elif section == "DEMAND" and line and "EOF" not in line.upper():
            parts = line.split()
            if len(parts) == 2:
                demands.append(int(parts[1]))
        elif section == "COORD" and line and "EOF" not in line.upper():
            parts = line.split()
            if len(parts) == 3:
                x, y = float(parts[1]), float(parts[2])
                coords.append((x, y))

    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xi, yi = coords[i]
            xj, yj = coords[j]
            dist_matrix[i][j] = round(np.hypot(xi - xj, yi - yj))

    demands, dist_matrix = _move_depot_to_front(
        _parse_depot(lines), demands, dist_matrix, source=os.path.basename(filepath))

    return CVRPInstance(name=name,
                        num_customers=n-1,
                        vehicle_capacity=capacity,
                        demands=demands,
                        distance_matrix=dist_matrix,
                        best_known_solution=best_known_solution)

def load_instance_from_dat(filepath, name, best_known_solution):
    """
    Loads a .dat file in ACVRP style.
    Reads CAPACITY, DEMAND_SECTION, and EDGE_WEIGHT_SECTION (or DISTANCE_SECTION).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    demands = []
    distance_matrix = []
    capacity = None
    reading_section = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "CAPACITY" in line.upper():
            parts = line.split()
            capacity = int(parts[-1])
        elif "DEMAND_SECTION" in line.upper():
            reading_section = "DEMAND"
            continue
        elif "EDGE_WEIGHT_SECTION" in line.upper() or "DISTANCE_SECTION" in line.upper():
            reading_section = "DISTANCE"
            continue
        elif reading_section == "DEMAND":
            parts = line.split()
            if len(parts) == 2:
                demands.append(int(parts[1]))
        elif reading_section == "DISTANCE":
            row_vals = list(map(float, line.split()))
            distance_matrix.append(row_vals)

    distance_matrix = np.array(distance_matrix)
    num_nodes = len(demands)

    demands, distance_matrix = _move_depot_to_front(
        _parse_depot(lines), demands, distance_matrix,
        source=os.path.basename(filepath))

    return CVRPInstance(name=name,
                        num_customers=num_nodes - 1,
                        vehicle_capacity=capacity,
                        demands=demands,
                        distance_matrix=distance_matrix,
                        best_known_solution=best_known_solution)

def read_bks_from_sol(filepath):
    """
    Reads a .sol file to extract the best-known solution from a line starting with "Cost".
    """
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'r') as f:
        for line in f:
            if line.lower().startswith("cost"):
                try:
                    return float(line.strip().split()[-1])
                except:
                    return None
    return None

def load_all_instances():
    """
    Loads the benchmark instances present on disk: up to 8 symmetric
    (Data/SCVRP/*.vrp) and up to 8 asymmetric (Data/ACVRP/*.dat).

    Instance files that are absent are skipped with a warning instead of
    raising, so the symmetric benchmark runs even when the ACVRP data is not
    checked in. Raises FileNotFoundError only if nothing at all could be loaded.
    """
    base_dir = os.path.dirname(__file__)
    scvrp_path = os.path.join(base_dir, "Data", "SCVRP")
    acvrp_path = os.path.join(base_dir, "Data", "ACVRP")

    instances = []
    missing = []

    symmetric_instances = [
        "E-n22-k4", "E-n51-k5", "E-n76-k7", "E-n76-k8",
        "E-n76-k10", "E-n76-k14", "E-n101-k8", "E-n101-k14"
    ]
    for inst_name in symmetric_instances:
        vrp_file = os.path.join(scvrp_path, f"{inst_name}.vrp")
        if not os.path.isfile(vrp_file):
            missing.append(vrp_file)
            continue
        sol_file = os.path.join(scvrp_path, f"{inst_name}.sol")
        bks = read_bks_from_sol(sol_file)
        instances.append(load_instance_from_vrp(vrp_file, inst_name, bks))

    asymmetric_instances = [
        ("A034-02f", 322),
        ("A036-03f", 341),
        ("A039-03f", 375),
        ("A045-03f", 414),
        ("A048-03f", 453),
        ("A056-03f", 495),
        ("A065-03f", 512),
        ("A071-03f", 548)
    ]
    for inst_name, bks_value in asymmetric_instances:
        dat_file = os.path.join(acvrp_path, f"{inst_name}.dat")
        if not os.path.isfile(dat_file):
            missing.append(dat_file)
            continue
        instances.append(load_instance_from_dat(dat_file, inst_name, bks_value))

    if missing:
        rel = [os.path.relpath(m, base_dir) for m in missing]
        print(f"[loader] skipped {len(rel)} missing instance file(s): "
              f"{', '.join(rel[:3])}{' ...' if len(rel) > 3 else ''}")

    if not instances:
        raise FileNotFoundError(
            f"no benchmark instances found under {os.path.join(base_dir, 'Data')}. "
            "Place CVRPLIB .vrp/.sol files in Data/SCVRP/ (and .dat files in "
            "Data/ACVRP/ for the asymmetric set)."
        )

    print(f"[loader] loaded {len(instances)} instance(s)")
    return instances
