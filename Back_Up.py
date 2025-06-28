"""
GA.py

Comprehensive integrated experimental framework for the Capacitated Vehicle Routing Problem (CVRP)
using a Genetic Algorithm (GA) framework that compares blind‐based (permutation‐based)
and distance‐based (random‐key) crossover operators.

Usage:
  1. Ensure cvrp_loader.py is in the same folder.
  2. Adjust the data paths in cvrp_loader.py as needed.
  3. Run: python GA.py
  4. Results (tables of best, average, standard deviation, average excess, time) will be printed.
"""

import random, time, statistics, numpy as np, matplotlib.pyplot as plt
from scipy import stats
from cvrp_loader import load_all_instances, CVRPInstance

# --------------------------------------------------------------------------------
#                             GA CONFIGURATION
# --------------------------------------------------------------------------------

POP_SIZE = 50
MAX_GENERATIONS = 1000         # For testing; use 5000 for full experiments.
CROSSOVER_RATE = 1.0
MUTATION_RATE = 0.10
NUM_RUNS = 10                 # For testing; use 50 for full experiments.

# --------------------------------------------------------------------------------
#                     SPLIT ALGORITHM FOR COST CALCULATION
# --------------------------------------------------------------------------------
def split_cost(solution, instance: CVRPInstance) -> float:
    """
    Given a giant tour solution (of the form [0, customer1, customer2, ..., customer_n, 0]),
    computes the minimum total cost by optimally partitioning the tour into feasible routes.
    (Each route must satisfy that the sum of customer demands is <= vehicle capacity.)

    We use a dynamic programming approach.
    """
    # Extract the giant tour without the depot markers:
    S = solution[1:-1]  # permutation of customers
    n = len(S)
    F = [float('inf')] * (n + 1)
    F[0] = 0
    # For each starting position i in the permutation, try to extend a route ending at j.
    for i in range(n):
        total_demand = 0
        # cost from depot to first customer in the segment:
        route_cost = instance.distance_matrix[0][S[i]]
        total_demand += instance.demands[S[i]]
        for j in range(i, n):
            if j > i:
                route_cost += instance.distance_matrix[S[j-1]][S[j]]
                total_demand += instance.demands[S[j]]
            if total_demand > instance.vehicle_capacity:
                break
            # Cost for the route covering customers S[i] ... S[j]
            current_route_cost = route_cost + instance.distance_matrix[S[j]][0]
            F[j+1] = min(F[j+1], F[i] + current_route_cost)
    return F[n]

# --------------------------------------------------------------------------------
#                           FITNESS FUNCTION
# --------------------------------------------------------------------------------
def fitness_function(permutation, instance: CVRPInstance) -> float:
    """
    Fitness function using the split algorithm cost.
    (No penalty term is used; the chromosome is assumed to be a giant tour [0, permutation, 0].)
    """
    cost_val = split_cost(permutation, instance)
    return 1.0 / (1.0 + cost_val)

# --------------------------------------------------------------------------------
#                     GIANT TOUR REPRESENTATION FUNCTIONS
# --------------------------------------------------------------------------------
def create_random_permutation(instance: CVRPInstance):
    """
    Returns a giant tour chromosome as [0, (random permutation of customers), 0].
    """
    customers = list(range(1, instance.num_customers + 1))
    random.shuffle(customers)
    return [0] + customers + [0]

# --------------------------------------------------------------------------------
#                   CROSSOVER OPERATORS (Permutation-Based)
# --------------------------------------------------------------------------------
def pmx_crossover(p1, p2):
    """
    Standard PMX for permutation encoding.
    """
    parent1 = p1[1:-1]
    parent2 = p2[1:-1]
    size = len(parent1)
    cp1, cp2 = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[cp1:cp2+1] = parent1[cp1:cp2+1]
    for i in range(cp1, cp2+1):
        gene = parent2[i]
        if gene not in child:
            pos = i
            while True:
                gene_from_parent1 = parent1[pos]
                pos = parent2.index(gene_from_parent1)
                if child[pos] is None:
                    child[pos] = gene
                    break
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    return [0] + child + [0]

def ox_crossover(p1, p2):
    """
    Order Crossover (OX) for permutation encoding.
    """
    parent1 = p1[1:-1]
    parent2 = p2[1:-1]
    size = len(parent1)
    cp1, cp2 = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[cp1:cp2+1] = parent1[cp1:cp2+1]
    fill_positions = list(range(cp2+1, size)) + list(range(0, cp1))
    p2_index = (cp2+1) % size
    for pos in fill_positions:
        while parent2[p2_index] in child:
            p2_index = (p2_index + 1) % size
        child[pos] = parent2[p2_index]
        p2_index = (p2_index + 1) % size
    return [0] + child + [0]

def cx_crossover(p1, p2):
    """
    Cycle Crossover (CX) for permutation encoding.
    """
    parent1 = p1[1:-1]
    parent2 = p2[1:-1]
    size = len(parent1)
    child = [None] * size
    used = [False] * size
    idx = 0
    while False in used:
        if not used[idx]:
            start_val = parent1[idx]
            cur_idx = idx
            while True:
                child[cur_idx] = parent1[cur_idx]
                used[cur_idx] = True
                gene = parent2[cur_idx]
                cur_idx = parent1.index(gene)
                if parent1[cur_idx] == start_val:
                    break
        idx = next((i for i, u in enumerate(used) if not u), idx)
    return [0] + child + [0]

def aex_crossover(p1, p2):
    """
    Alternating Edges Crossover (AEX) for permutation encoding.
    """
    parent1 = p1[1:-1]
    parent2 = p2[1:-1]
    size = len(parent1)
    child = []
    used = set()
    i1, i2 = 0, 0
    use_p1 = True
    while len(child) < size:
        if use_p1:
            while parent1[i1] in used:
                i1 = (i1 + 1) % size
            gene = parent1[i1]
            child.append(gene)
            used.add(gene)
            i1 = (i1 + 1) % size
        else:
            while parent2[i2] in used:
                i2 = (i2 + 1) % size
            gene = parent2[i2]
            child.append(gene)
            used.add(gene)
            i2 = (i2 + 1) % size
        use_p1 = not use_p1
    return [0] + child + [0]

def mutation_permutation(chrom):
    """
    Swap mutation for permutation encoding (excludes depot markers).
    """
    if len(chrom) <= 2:
        return
    idx1, idx2 = random.sample(range(1, len(chrom)-1), 2)
    chrom[idx1], chrom[idx2] = chrom[idx2], chrom[idx1]

# --------------------------------------------------------------------------------
#             CROSSOVER OPERATORS (Random-Key Based)
# --------------------------------------------------------------------------------
def create_random_keys(instance: CVRPInstance):
    """
    Returns a list of random keys (one per customer).
    """
    return [random.random() for _ in range(instance.num_customers)]

def decode_keys_to_permutation(keys, instance: CVRPInstance):
    """
    Decodes a random-key vector into a giant tour.
    Returns [0] + permutation of customers + [0].
    """
    customers = list(range(1, instance.num_customers + 1))
    sorted_customers = sorted(customers, key=lambda c: keys[c-1])
    return [0] + sorted_customers + [0]

def brkga_crossover(p1, p2):
    """
    Biased Random-Key GA crossover (p=0.7 probability from parent1).
    """
    p = 0.7
    child = []
    for i in range(len(p1)):
        child.append(p1[i] if random.random() < p else p2[i])
    return child

def uniform_rk_crossover(p1, p2):
    """
    Uniform crossover for random-key encoding.
    """
    return [p1[i] if random.random() < 0.5 else p2[i] for i in range(len(p1))]

def sorted_rk_greedy_crossover(p1, p2):
    """
    Sorted-based random-key crossover: average sorted keys.
    """
    p1_sorted = sorted(p1)
    p2_sorted = sorted(p2)
    size = len(p1)
    return [0.5 * (p1_sorted[i] + p2_sorted[i]) for i in range(size)]

def two_point_rk_crossover(p1, p2):
    """
    Two-point crossover for random-key encoding.
    """
    size = len(p1)
    cp1, cp2 = sorted(random.sample(range(size), 2))
    child = [None] * size
    for i in range(cp1, cp2+1):
        child[i] = p1[i]
    for i in range(size):
        if child[i] is None:
            child[i] = p2[i]
    return child

def mutation_random_key(chrom):
    """
    Mutation for random-key: small perturbation of a randomly chosen gene.
    """
    idx = random.randint(0, len(chrom)-1)
    delta = (random.random() - 0.5) * 0.2
    chrom[idx] = max(0.0, min(1.0, chrom[idx] + delta))

# --------------------------------------------------------------------------------
#                     LOCAL SEARCH: TWO-OPT IMPROVEMENT
# --------------------------------------------------------------------------------
def compute_route_cost(route, instance):
    """Computes the cost of a given route (list of nodes)."""
    cost = 0.0
    for i in range(len(route)-1):
        cost += instance.distance_matrix[route[i]][route[i+1]]
    return cost

def two_opt_route(route, instance):
    """
    Applies the 2-opt algorithm to a single route.
    The route is assumed to start and end with depot (0).
    Returns the improved route.
    """
    best_route = route[:]
    best_cost = compute_route_cost(best_route, instance)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for k in range(i+1, len(best_route)-1):
                new_route = best_route[:i] + best_route[i:k+1][::-1] + best_route[k+1:]
                new_cost = compute_route_cost(new_route, instance)
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
        # Loop until no improvement.
    return best_route

def split_routes(solution):
    """
    Splits a giant tour solution (with depot markers) into individual routes.
    Example: [0, a, b, 0, c, d, 0] -> [[0, a, b, 0], [0, c, d, 0]]
    """
    routes = []
    current_route = []
    for node in solution:
        if node == 0:
            if current_route:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
            else:
                current_route = [0]
        else:
            current_route.append(node)
    if current_route and current_route[-1] != 0:
        current_route.append(0)
        routes.append(current_route)
    return routes

def two_opt_improvement(solution, instance):
    """
    Applies 2-opt improvement to a full solution by splitting into routes,
    improving each route, and then concatenating back.
    """
    routes = split_routes(solution)
    improved_routes = []
    for route in routes:
        if len(route) > 3:
            improved_route = two_opt_route(route, instance)
        else:
            improved_route = route
        improved_routes.append(improved_route)
    new_solution = improved_routes[0]
    for r in improved_routes[1:]:
        new_solution.extend(r[1:])
    return new_solution

# --------------------------------------------------------------------------------
#                      REPAIR FUNCTION FOR RANDOM-KEY DECODED OFFSPRING
# --------------------------------------------------------------------------------
def repair_random_key_solution(perm, instance: CVRPInstance):
    """
    Checks if the decoded permutation (giant tour) is feasible via the split algorithm.
    If the split cost is infinity (i.e. no feasible partition found), return a new random permutation.
    Otherwise, return the original permutation.
    """
    cost_val = split_cost(perm, instance)
    if cost_val == float('inf'):
        return create_random_permutation(instance)
    return perm

# --------------------------------------------------------------------------------
#                        5. SELECTION & REPLACEMENT
# --------------------------------------------------------------------------------
def roulette_wheel_selection(population, fitnesses):
    """
    Fitness-proportional selection; returns an index.
    """
    total_fit = sum(fitnesses)
    if total_fit == 0:
        return random.randint(0, len(population)-1)
    pick = random.random() * total_fit
    current = 0
    for i, f in enumerate(fitnesses):
        current += f
        if current >= pick:
            return i
    return len(population)-1

def elitist_replacement(offspring, population, old_fitnesses):
    """
    Replaces the worst offspring with the best individual from the previous generation.
    """
    best_idx = max(range(len(population)), key=lambda i: old_fitnesses[i])
    best_sol = population[best_idx]
    best_fit = old_fitnesses[best_idx]
    off_fits = [o["fitness"] for o in offspring]
    worst_idx = min(range(len(offspring)), key=lambda i: off_fits[i])
    offspring[worst_idx] = {"chromosome": best_sol["chromosome"], "fitness": best_fit}
    return offspring

# --------------------------------------------------------------------------------
#             6. GA CORE LOOP (PER-INSTANCE, PER-CROSSOVER)
# --------------------------------------------------------------------------------
def run_ga_for_instance(instance: CVRPInstance, encoding: str, crossover_operator, apply_mutation: bool):
    """
    Runs the GA on a single instance with the specified encoding and crossover operator.
    Returns a dictionary with keys: best_cost, avg_cost, std_dev, avg_exc, time, and convergence_curve.

    For permutation-based encoding, we use the split algorithm cost.
    For random-key encoding, after decoding the keys we repair the offspring if needed.
    """
    population = []
    if encoding == "permutation":
        for _ in range(POP_SIZE):
            chrom = create_random_permutation(instance)
            fit = fitness_function(chrom, instance)
            population.append({"chromosome": chrom, "fitness": fit})
    else:
        for _ in range(POP_SIZE):
            rk = create_random_keys(instance)
            perm = decode_keys_to_permutation(rk, instance)
            # Repair if necessary.
            perm = repair_random_key_solution(perm, instance)
            fit = fitness_function(perm, instance)
            population.append({"chromosome": rk, "fitness": fit})

    convergence_curve = []
    start_time = time.time()
    for gen in range(MAX_GENERATIONS):
        fitnesses = [ind["fitness"] for ind in population]
        # Track best cost using split cost
        costs = []
        for ind in population:
            if encoding == "permutation":
                c_val = split_cost(ind["chromosome"], instance)
            else:
                dec = decode_keys_to_permutation(ind["chromosome"], instance)
                c_val = split_cost(dec, instance)
            costs.append(c_val)
        best_cost = min(costs)
        convergence_curve.append(best_cost)

        new_population = []
        for _ in range(POP_SIZE // 2):
            p1_idx = roulette_wheel_selection(population, fitnesses)
            p2_idx = roulette_wheel_selection(population, fitnesses)
            parent1 = population[p1_idx]["chromosome"]
            parent2 = population[p2_idx]["chromosome"]
            if random.random() < CROSSOVER_RATE:
                child1 = crossover_operator(parent1, parent2)
                child2 = crossover_operator(parent2, parent1)
            else:
                child1 = parent1[:]
                child2 = parent2[:]
            if apply_mutation and random.random() < MUTATION_RATE:
                if encoding == "permutation":
                    mutation_permutation(child1)
                else:
                    mutation_random_key(child1)
            if apply_mutation and random.random() < MUTATION_RATE:
                if encoding == "permutation":
                    mutation_permutation(child2)
                else:
                    mutation_random_key(child2)
            if encoding == "permutation":
                child1 = two_opt_improvement(child1, instance)
                child2 = two_opt_improvement(child2, instance)
                # (For permutation encoding, we assume the giant tour is maintained.)
                f1 = fitness_function(child1, instance)
                f2 = fitness_function(child2, instance)
            else:
                # For random-key: decode, repair if needed, and then compute fitness.
                perm1 = decode_keys_to_permutation(child1, instance)
                perm1 = repair_random_key_solution(perm1, instance)
                perm2 = decode_keys_to_permutation(child2, instance)
                perm2 = repair_random_key_solution(perm2, instance)
                f1 = fitness_function(perm1, instance)
                f2 = fitness_function(perm2, instance)
            new_population.append({"chromosome": child1, "fitness": f1})
            new_population.append({"chromosome": child2, "fitness": f2})
        population = elitist_replacement(new_population, population, fitnesses)

    end_time = time.time()
    total_time = end_time - start_time
    final_costs = []
    for ind in population:
        if encoding == "permutation":
            cost_val = split_cost(ind["chromosome"], instance)
        else:
            dec = decode_keys_to_permutation(ind["chromosome"], instance)
            cost_val = split_cost(dec, instance)
        final_costs.append(cost_val)
    best_sol = min(final_costs)
    avg_sol = statistics.mean(final_costs)
    std_sol = statistics.pstdev(final_costs)
    bks = instance.best_known_solution if instance.best_known_solution else 0.0
    avg_exc = ((avg_sol - bks) / bks) * 100.0 if bks > 0 else 0.0
    return {
        "best_cost": best_sol,
        "avg_cost": avg_sol,
        "std_dev": std_sol,
        "avg_exc": avg_exc,
        "time": total_time,
        "convergence_curve": convergence_curve
    }

# --------------------------------------------------------------------------------
#               7. RUN EXPERIMENTS (ALL INSTANCES, ALL OPERATORS)
# --------------------------------------------------------------------------------
def run_all_experiments():
    """
    Loads all instances and runs GA experiments for each crossover operator,
    with and without mutation, and prints summary tables.
    """
    blind_based_ops = {
        "PMX": pmx_crossover,
        "OX": ox_crossover,
        "CX": cx_crossover,
        "AEX": aex_crossover
    }
    dist_based_ops = {
        "BRKGA": brkga_crossover,
        "RK-U": uniform_rk_crossover,
        "RK-GR": sorted_rk_greedy_crossover,
        "RK-2PX": two_point_rk_crossover
    }
    instances = load_all_instances()
    for inst in instances:
        print(f"\n================= Instance: {inst.name} =================")
        print(f"Dimension (nodes): {inst.num_customers + 1}, VehicleCap={inst.vehicle_capacity}, BKS={inst.best_known_solution}")
        results_no_mut = {}
        results_with_mut = {}
        # Permutation-based operators
        for op_name, op_func in blind_based_ops.items():
            bests, avgs, stds, excs, times = [], [], [], [], []
            for _ in range(NUM_RUNS):
                outcome = run_ga_for_instance(inst, "permutation", op_func, apply_mutation=False)
                bests.append(outcome["best_cost"])
                avgs.append(outcome["avg_cost"])
                stds.append(outcome["std_dev"])
                excs.append(outcome["avg_exc"])
                times.append(outcome["time"])
            results_no_mut[op_name] = {
                "best_sol": min(bests),
                "avg_sol": statistics.mean(avgs),
                "std_sol": statistics.mean(stds),
                "avg_exc": statistics.mean(excs),
                "time": statistics.mean(times)
            }
            bests, avgs, stds, excs, times = [], [], [], [], []
            for _ in range(NUM_RUNS):
                outcome = run_ga_for_instance(inst, "permutation", op_func, apply_mutation=True)
                bests.append(outcome["best_cost"])
                avgs.append(outcome["avg_cost"])
                stds.append(outcome["std_dev"])
                excs.append(outcome["avg_exc"])
                times.append(outcome["time"])
            results_with_mut[op_name] = {
                "best_sol": min(bests),
                "avg_sol": statistics.mean(avgs),
                "std_sol": statistics.mean(stds),
                "avg_exc": statistics.mean(excs),
                "time": statistics.mean(times)
            }
        # Random-key based operators
        for op_name, op_func in dist_based_ops.items():
            bests, avgs, stds, excs, times = [], [], [], [], []
            for _ in range(NUM_RUNS):
                outcome = run_ga_for_instance(inst, "randomkey", op_func, apply_mutation=False)
                bests.append(outcome["best_cost"])
                avgs.append(outcome["avg_cost"])
                stds.append(outcome["std_dev"])
                excs.append(outcome["avg_exc"])
                times.append(outcome["time"])
            results_no_mut[op_name] = {
                "best_sol": min(bests),
                "avg_sol": statistics.mean(avgs),
                "std_sol": statistics.mean(stds),
                "avg_exc": statistics.mean(excs),
                "time": statistics.mean(times)
            }
            bests, avgs, stds, excs, times = [], [], [], [], []
            for _ in range(NUM_RUNS):
                outcome = run_ga_for_instance(inst, "randomkey", op_func, apply_mutation=True)
                bests.append(outcome["best_cost"])
                avgs.append(outcome["avg_cost"])
                stds.append(outcome["std_dev"])
                excs.append(outcome["avg_exc"])
                times.append(outcome["time"])
            results_with_mut[op_name] = {
                "best_sol": min(bests),
                "avg_sol": statistics.mean(avgs),
                "std_sol": statistics.mean(stds),
                "avg_exc": statistics.mean(excs),
                "time": statistics.mean(times)
            }
        # Print results tables
        all_ops = list(blind_based_ops.keys()) + list(dist_based_ops.keys())
        print("\n--- Results (Without Mutation) ---")
        print("Operator | Best Sol |  Avg Sol | AvgExc(%) |   StdDev |  Time(s)")
        print("--------------------------------------------------------------")
        for op_name in all_ops:
            r = results_no_mut[op_name]
            print(f"{op_name:<8} | {r['best_sol']:.2f}   | {r['avg_sol']:.2f}   | {r['avg_exc']:.2f}      | {r['std_sol']:.2f}     | {r['time']:.2f}")
        print("\n--- Results (With Mutation) ---")
        print("Operator | Best Sol |  Avg Sol | AvgExc(%) |   StdDev |  Time(s)")
        print("-------------------------------------------------------------")
        for op_name in all_ops:
            r = results_with_mut[op_name]
            print(f"{op_name:<8} | {r['best_sol']:.2f}   | {r['avg_sol']:.2f}   | {r['avg_exc']:.2f}      | {r['std_sol']:.2f}     | {r['time']:.2f}")

def collect_asymmetric_results(with_mutation=False):
    """
    Runs GA on all asymmetric instances (names starting with 'A') for each operator and
    collects run-by-run average excess percentages.
    Returns a dictionary: results[instance_name][operator] = [avg_exc_run1, avg_exc_run2, ...]
    """
    blind_ops = {"PMX": pmx_crossover, "OX": ox_crossover, "CX": cx_crossover, "AEX": aex_crossover}
    dist_ops = {"BRKGA": brkga_crossover, "RK-U": uniform_rk_crossover, "RK-GR": sorted_rk_greedy_crossover, "RK-2PX": two_point_rk_crossover}
    all_ops = {**blind_ops, **dist_ops}
    results = {}
    instances = load_all_instances()
    for inst in instances:
        if not inst.name.startswith("A"):
            continue
        results[inst.name] = {}
        for op_name, op_func in all_ops.items():
            run_excess = []
            for _ in range(NUM_RUNS):
                outcome = run_ga_for_instance(inst,
                                              "permutation" if op_name in blind_ops else "randomkey",
                                              op_func,
                                              apply_mutation=with_mutation)
                run_excess.append(outcome["avg_exc"])
            results[inst.name][op_name] = run_excess
    return results

def print_t_test_table(results):
    """
    For each asymmetric instance, performs paired t-tests:
      - For blind-based operators: Compare PMX, OX, CX vs. AEX.
      - For distance-based operators: Compare BRKGA, RK-U, RK-2PX vs. RK-GR.
    Then prints a table with the t-statistic values.
    """
    blind_ops = ["PMX", "OX", "CX", "AEX"]
    dist_ops = ["BRKGA", "RK-U", "RK-GR", "RK-2PX"]
    header = ("Instance\t" +
              "t (PMX vs AEX)\t" +
              "t (OX vs AEX)\t" +
              "t (CX vs AEX)\t" +
              "t (BRKGA vs RK-GR)\t" +
              "t (RK-U vs RK-GR)\t" +
              "t (RK-2PX vs RK-GR)")
    print("\nTable 7: t-Test Analysis (Asymmetric Instances, Without Mutation)")
    print(header)
    for inst_name, op_results in results.items():
        if not all(op in op_results for op in blind_ops + dist_ops):
            continue
        try:
            t_pmx_aex = stats.ttest_rel(op_results["PMX"], op_results["AEX"]).statistic
            t_ox_aex = stats.ttest_rel(op_results["OX"], op_results["AEX"]).statistic
            t_cx_aex = stats.ttest_rel(op_results["CX"], op_results["AEX"]).statistic
        except Exception:
            t_pmx_aex, t_ox_aex, t_cx_aex = float('nan'), float('nan'), float('nan')
        try:
            t_brkga_gr = stats.ttest_rel(op_results["BRKGA"], op_results["RK-GR"]).statistic
            t_rku_gr = stats.ttest_rel(op_results["RK-U"], op_results["RK-GR"]).statistic
            t_rk2px_gr = stats.ttest_rel(op_results["RK-2PX"], op_results["RK-GR"]).statistic
        except Exception:
            t_brkga_gr, t_rku_gr, t_rk2px_gr = float('nan'), float('nan'), float('nan')
        print(f"{inst_name}\t{t_pmx_aex:.2f}\t\t{t_ox_aex:.2f}\t\t{t_cx_aex:.2f}\t\t{t_brkga_gr:.2f}\t\t{t_rku_gr:.2f}\t\t{t_rk2px_gr:.2f}")

def plot_bar_chart_avg_excess_fixed(results):
    """
    Plots a bar chart comparing the overall average excess percentage for each operator,
    averaged across all asymmetric instances.
    """
    import matplotlib.pyplot as plt, statistics
    operator_values = {}
    for inst_name, op_results in results.items():
        for op, run_excess_list in op_results.items():
            operator_values.setdefault(op, []).extend(run_excess_list)
    operators = sorted(operator_values.keys())
    avg_excesses = [statistics.mean(operator_values[op]) for op in operators]
    plt.figure(figsize=(10,6))
    plt.bar(operators, avg_excesses, color='skyblue')
    plt.xlabel("Crossover Operator")
    plt.ylabel("Average Excess (%)")
    plt.title("Average Excess (%) Comparison (Asymmetric Instances, No Mutation)")
    plt.tight_layout()
    plt.show()

def plot_eight_ops_convergence(instance_name, with_mutation=False):
    """
    Plots a convergence curve (Excess % vs. Generation) for all 8 crossover operators on the specified instance.
    """
    import matplotlib.pyplot as plt
    all_instances = load_all_instances()
    target_inst = None
    for inst in all_instances:
        if inst.name == instance_name:
            target_inst = inst
            break
    if target_inst is None:
        print(f"Instance '{instance_name}' not found.")
        return
    blind_ops = {"PMX": pmx_crossover, "OX": ox_crossover, "CX": cx_crossover, "AEX": aex_crossover}
    dist_ops = {"BRKGA": brkga_crossover, "RK-U": uniform_rk_crossover, "RK-GR": sorted_rk_greedy_crossover, "RK-2PX": two_point_rk_crossover}
    all_ops = {**blind_ops, **dist_ops}
    plt.figure(figsize=(10,6))
    for op_name, op_func in all_ops.items():
        encoding = "permutation" if op_name in blind_ops else "randomkey"
        outcome = run_ga_for_instance(target_inst, encoding, op_func, apply_mutation=with_mutation)
        bks = target_inst.best_known_solution if target_inst.best_known_solution else 0
        if bks <= 0:
            print(f"No valid BKS for instance '{instance_name}'. Skipping plot.")
            return
        percent_excess_curve = [((c - bks) / bks) * 100 for c in outcome["convergence_curve"]]
        generations = list(range(len(percent_excess_curve)))
        label_text = f"{op_name} ({'Mut' if with_mutation else 'No Mut'})"
        plt.plot(generations, percent_excess_curve, label=label_text)
    plt.xlabel("Generation")
    plt.ylabel("Excess (%)")
    plt.title(f"Convergence of 8 Operators on {instance_name} ({'With' if with_mutation else 'Without'} Mutation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_line_chart_all_ops(results, title="Average Excess (%) by GAs without mutation for asymmetric instances"):
    """
    Plots a multi-line chart: x-axis = instance names, y-axis = average excess, one line per operator.
    """
    import matplotlib.pyplot as plt, statistics
    instance_names = sorted(results.keys())
    all_ops = set()
    for inst in instance_names:
        all_ops.update(results[inst].keys())
    all_ops = sorted(all_ops)
    plt.figure(figsize=(10,6))
    for op in all_ops:
        y_vals = []
        for inst in instance_names:
            run_excess_list = results[inst].get(op, [])
            if run_excess_list:
                avg_val = statistics.mean(run_excess_list)
            else:
                avg_val = None
            y_vals.append(avg_val)
        plt.plot(instance_names, y_vals, marker='o', label=op)
    plt.xlabel("Asymmetric Instances")
    plt.ylabel("Average Excess (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bar_chart_avg_excess(results):
    """
    Plots a bar chart comparing the overall average excess percentage for each operator,
    averaged across all asymmetric instances.
    """
    import matplotlib.pyplot as plt, statistics
    operator_values = {}
    for inst, op_results in results.items():
        for op, values in op_results.items():
            operator_values.setdefault(op, []).extend(values)
    operators = list(operator_values.keys())
    avg_excess = [statistics.mean(operator_values[op]) for op in operators]
    plt.figure(figsize=(10,6))
    plt.bar(operators, avg_excess, color='skyblue')
    plt.xlabel("Crossover Operator")
    plt.ylabel("Average Excess (%)")
    plt.title("Average Excess (%) Comparison Across Asymmetric Instances")
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------
#                              MAIN FUNCTION
# --------------------------------------------------------------------------------
def main():
    run_all_experiments()
    asym_results = collect_asymmetric_results(with_mutation=False)
    print_t_test_table(asym_results)
    plot_bar_chart_avg_excess_fixed(asym_results)
    plot_line_chart_all_ops(asym_results, title="Average Excess (%) by GAs without mutation (all operators)")
    plot_eight_ops_convergence("A071-03f", with_mutation=False)
    plot_eight_ops_convergence("A071-03f", with_mutation=True)
    with_mut_results = collect_asymmetric_results(with_mutation=True)
    plot_line_chart_all_ops(with_mut_results, title="Average Excess (%) by GAs with mutation (all operators)")

def collect_asymmetric_results(with_mutation=False):
    """
    Runs GA on all asymmetric instances (names starting with 'A') for each operator and collects run-by-run
    average excess percentages.
    Returns a dictionary: results[instance_name][operator] = [avg_exc_run1, avg_exc_run2, ...]
    """
    blind_ops = {"PMX": pmx_crossover, "OX": ox_crossover, "CX": cx_crossover, "AEX": aex_crossover}
    dist_ops = {"BRKGA": brkga_crossover, "RK-U": uniform_rk_crossover, "RK-GR": sorted_rk_greedy_crossover, "RK-2PX": two_point_rk_crossover}
    all_ops = {**blind_ops, **dist_ops}
    results = {}
    instances = load_all_instances()
    for inst in instances:
        if not inst.name.startswith("A"):
            continue
        results[inst.name] = {}
        for op_name, op_func in all_ops.items():
            run_excess = []
            for _ in range(NUM_RUNS):
                outcome = run_ga_for_instance(inst,
                                              "permutation" if op_name in blind_ops else "randomkey",
                                              op_func,
                                              apply_mutation=with_mutation)
                run_excess.append(outcome["avg_exc"])
            results[inst.name][op_name] = run_excess
    return results

if __name__ == "__main__":
    main()