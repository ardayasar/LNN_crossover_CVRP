#!/usr/bin/env python3
"""
GA.py – Integrated GA framework with LNN crossover + CSV logging (Phase-1)

• Benchmarks 6 crossover operators (OX, CX, AEX, LNN, BRKGA, RK-2PX)
  under both “with” and “without” mutation regimes.
• Dumps per-run summary CSVs in results/
• Logs per-generation best-cost in results/convergence/… for convergence plots
• Optionally logs LNN training triples if LNN_LOG=1.

Usage:
    LNN_LOG=1 python GA.py        # enable LNN triple logging
    python GA.py                  # just run the GA benchmark
"""
from __future__ import annotations
import os, random, time, statistics, uuid, datetime, pickle, csv
from typing import Callable, Dict, List

import torch
from cvrp_loader import load_all_instances, CVRPInstance
from GA_LNN.lnn.model import LiquidCrossover
from GA_LNN.lnn.encoder import encode_parents
from GA_LNN.lnn.decoder import scores_to_perm
import concurrent.futures
import multiprocessing

# ──────────────────────────────────  CONSTANTS  ─────────────────────────────────
FEATURE_DIM       = 3               # rank1 • rank2 • demand
POP_SIZE          = 25
MAX_GENERATIONS   = 400
CROSSOVER_RATE    = 1.0
MUTATION_RATE     = 0.10
NUM_RUNS          = 15

ENABLE_LNN_LOGGING = os.getenv("LNN_LOG", "0") == "1"
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT       = os.path.dirname(__file__)
RESULTS_DIR        = os.path.join(PROJECT_ROOT, "results")  # update as desired
os.makedirs(RESULTS_DIR, exist_ok=True)

# ────────────────────────────  LNN data-logging buffer  ─────────────────────────
LNN_LOG_DIR = os.path.join(PROJECT_ROOT, "GA_LNN", "data")
os.makedirs(LNN_LOG_DIR, exist_ok=True)
_LOG_BUFFER: List = []
_MAX_CACHE = 5000  # dump every 5k samples


def _flush_lnn_buffer() -> None:
    if not _LOG_BUFFER:
        return
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"lnn_log_{ts}_{uuid.uuid4().hex[:6]}.pkl"
    with open(os.path.join(LNN_LOG_DIR, name), "ab") as fh:
        pickle.dump(_LOG_BUFFER, fh)
    print(f"[LNN-LOG] dumped {len(_LOG_BUFFER)} samples → {name}")
    _LOG_BUFFER.clear()


# ───────────────────────────────  CVRP helpers  ────────────────────────────────
def split_cost(tour: List[int], inst: CVRPInstance) -> float:
    """Compute CVRP split cost of giant tour."""
    S, n = tour[1:-1], len(tour)-2
    F = [float("inf")] * (n+1)
    F[0] = 0.0
    for i in range(n):
        demand, cost = inst.demands[S[i]], inst.distance_matrix[0][S[i]]
        for j in range(i, n):
            if j>i:
                demand += inst.demands[S[j]]
                cost += inst.distance_matrix[S[j-1]][S[j]]
            if demand > inst.vehicle_capacity:
                break
            cur = cost + inst.distance_matrix[S[j]][0]
            F[j+1] = min(F[j+1], F[i] + cur)
    return F[n]

def fitness(tour: List[int], inst: CVRPInstance) -> float:
    return 1.0 / (1.0 + split_cost(tour, inst))


# ───────────────────────────────  Load LNN model  ──────────────────────────────
LNN_MODEL = LiquidCrossover(in_dim=FEATURE_DIM).to(DEVICE)
MODEL_PATH = os.path.join(PROJECT_ROOT, "GA_LNN", "lnn", "lnn_hyx.pt")
try:
    LNN_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    if multiprocessing.current_process().name == "MainProcess":
        print(f"[LNN] loaded weights from {MODEL_PATH}")
except Exception as e:
    if multiprocessing.current_process().name == "MainProcess":
        print(f"[LNN] checkpoint load failed ({e}), using random init")
LNN_MODEL.eval()


# ───────────────────────────  Crossover operators  ───────────────────────────
def pmx_crossover(p1, p2, _):  # PMX
    size = len(p1)-2
    a,b = sorted(random.sample(range(size),2))
    mid = p1[a+1:b+1]
    child = [0] + mid + [None]*(size-len(mid)) + [0]
    fill = [g for g in p2[1:-1] if g not in mid]
    idx=1
    for g in fill:
        while child[idx] is not None:
            idx+=1
        child[idx]=g
    return child

def ox_crossover(p1,p2,_):  # OX
    size = len(p1)-2
    a,b = sorted(random.sample(range(size),2))
    mid = p1[a+1:b+1]
    fill = [g for g in p2[1:-1] if g not in mid]
    return [0] + fill[:a+1] + mid + fill[a+1:] + [0]

def cx_crossover(p1,p2,_):  # CX
    size = len(p1)-2
    child = [None]*size
    visited=set(); cycle=0
    while len(visited)<size:
        start = next(i for i in range(size) if i not in visited)
        i = start
        while i not in visited:
            visited.add(i)
            child[i] = (p1 if cycle%2==0 else p2)[i+1]
            i = p1.index(p2[i+1]) -1
        cycle+=1
    return [0]+child+[0]

def aex_crossover(p1,p2,_):  # AEX
    size = len(p1)-2
    i1,i2,turn = 1,1,True
    child=[0]
    while len(child)<size+1:
        src,idx = (p1,i1) if turn else (p2,i2)
        gene=src[idx]
        if gene not in child: child.append(gene)
        if turn: i1 = (i1%size)+1
        else:   i2 = (i2%size)+1
        turn = not turn
    return child+[0]

def lnn_crossover(p1,p2,inst):  # LNN
    feats = encode_parents(p1[1:-1],p2[1:-1],inst).to(DEVICE)
    with torch.no_grad():
        scores = LNN_MODEL(feats)
    perm = scores_to_perm(scores.cpu())
    return [0] + perm + [0]


# ─────────────────────────  Random-key crossovers  ──────────────────────────
def rk_brkga(u,v,_): return [u[i] if random.random()<0.7 else v[i] for i in range(len(u))]
def rk_uniform(u,v,_): return [u[i] if random.random()<0.5 else v[i] for i in range(len(u))]
def rk_greedy(u,v,_):  return [(a+b)*0.5 for a,b in zip(sorted(u),sorted(v))]
def rk_two_point(u,v,_):
    a,b=sorted(random.sample(range(len(u)),2))
    c=u[:]; c[a:b+1]=v[a:b+1]; return c


# ───────────────────────────  Mutation operators  ───────────────────────────
def mut_perm(chrom):
    i,j = sorted(random.sample(range(1,len(chrom)-1),2))
    chrom[i],chrom[j] = chrom[j],chrom[i]

def mut_rk(vec):
    i=random.randrange(len(vec))
    vec[i]=min(1.0,max(0.0,vec[i]+(random.random()-0.5)*0.2))


# ───────────────────────────  Local search (2-opt)  ──────────────────────────
def compute_route_cost(r,inst):
    return sum(inst.distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1))

def two_opt_route(r,inst):
    best,improved=r[:],True
    while improved:
        improved=False
        for i in range(1,len(best)-2):
            for k in range(i+1,len(best)-1):
                cand = best[:i] + best[i:k+1][::-1] + best[k+1:]
                if compute_route_cost(cand,inst)<compute_route_cost(best,inst):
                    best,improved=cand,True
    return best

def split_routes(tour):
    routes,cur, = [],[]
    for node in tour:
        if node==0:
            if cur: routes.append(cur+[0])
            cur=[0]
        else:
            cur.append(node)
    return routes

def two_opt_improvement(tour,inst):
    parts = [two_opt_route(r,inst) if len(r)>3 else r for r in split_routes(tour)]
    merged = parts[0]
    for r in parts[1:]:
        merged.extend(r[1:])
    return merged


# ─────────────────────────────  GA helper functions  ───────────────────────────
def elitist_replacement(offspring,population,old_fit):
    worst = min(range(len(offspring)), key=lambda i: offspring[i]["fitness"])
    best  = max(range(len(population)), key=lambda i: old_fit[i])
    offspring[worst] = population[best]
    return offspring

def write_metrics_to_csv(metrics,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["run","best_cost","avg_cost","std_dev","avg_exc","time"])
        for i,m in enumerate(metrics,1):
            w.writerow([i,m["best_cost"],m["avg_cost"],m["std_dev"],m["avg_exc"],m["time"]])


# ───────────────────────────────────  GA core  ───────────────────────────────────
def _log_triple(p1,p2,c,inst_name):
    if not ENABLE_LNN_LOGGING: return
    _LOG_BUFFER.append((p1[1:-1],p2[1:-1],c[1:-1],inst_name))
    if len(_LOG_BUFFER)>=_MAX_CACHE:
        _flush_lnn_buffer()


def parallel_run_ga(args):
    inst,op_name,encoding,crossover,apply_mutation,run_id = args
    return run_ga_for_instance(
        inst, op_name, encoding, crossover,
        apply_mutation=apply_mutation, run_id=run_id
    )


def run_ga_for_instance(
    inst: CVRPInstance,
    op_name: str,
    encoding: str,
    crossover: Callable,
    *,
    apply_mutation: bool,
    run_id: int
) -> dict:
    # ---------- init population ----------
    population=[]
    if encoding=="permutation":
        for _ in range(POP_SIZE):
            tour=[0]+random.sample(range(1,inst.num_customers+1),inst.num_customers)+[0]
            population.append({"chromosome":tour,"fitness":fitness(tour,inst)})
    else:
        for _ in range(POP_SIZE):
            rk=[random.random() for _ in range(inst.num_customers)]
            tour=[0]+sorted(range(1,inst.num_customers+1),key=lambda i:rk[i-1])+[0]
            population.append({"chromosome":rk,"fitness":fitness(tour,inst)})

    # ---------- prepare convergence log ----------
    conv_dir = os.path.join(RESULTS_DIR,"convergence",inst.name,
                            op_name, "mut" if apply_mutation else "nomut")
    os.makedirs(conv_dir, exist_ok=True)
    conv_path = os.path.join(conv_dir, f"run_{run_id}_convergence.csv")
    conv_fh = open(conv_path,"w",newline="")
    conv_wr = csv.writer(conv_fh)
    conv_wr.writerow(["generation","best_cost"])

    start_t = time.time()

    # ---------- evolution loop ----------
    for gen in range(1, MAX_GENERATIONS+1):
        fits=[ind["fitness"] for ind in population]
        offspring=[]
        for _ in range(POP_SIZE//2):
            p1=population[random.randrange(len(fits))]["chromosome"]
            p2=population[random.randrange(len(fits))]["chromosome"]
            if random.random()<CROSSOVER_RATE:
                c1=crossover(p1,p2,inst)
                c2=crossover(p2,p1,inst)
            else:
                c1, c2 = p1[:], p2[:]

            if encoding=="permutation":
                _log_triple(p1,p2,c1,inst.name)
                _log_triple(p2,p1,c2,inst.name)

            if apply_mutation and random.random()<MUTATION_RATE:
                (mut_perm if encoding=="permutation" else mut_rk)(c1)
            if apply_mutation and random.random()<MUTATION_RATE:
                (mut_perm if encoding=="permutation" else mut_rk)(c2)

            # local search + fitness
            if encoding=="permutation":
                c1=two_opt_improvement(c1,inst)
                c2=two_opt_improvement(c2,inst)
                f1, f2 = fitness(c1,inst), fitness(c2,inst)
            else:
                t1=[0]+sorted(range(1,inst.num_customers+1),
                              key=lambda i:c1[i-1])+[0]
                t2=[0]+sorted(range(1,inst.num_customers+1),
                              key=lambda i:c2[i-1])+[0]
                f1, f2 = fitness(t1,inst), fitness(t2,inst)

            offspring.extend([{"chromosome":c1,"fitness":f1},
                              {"chromosome":c2,"fitness":f2}])

        population = elitist_replacement(offspring,population,fits)

        # log best cost this generation
        pop_costs=[]
        for ind in population:
            if encoding=="permutation":
                tour = ind["chromosome"]
            else:
                rk = ind["chromosome"]
                tour = [0]+sorted(range(1,inst.num_customers+1),
                                  key=lambda i:rk[i-1])+[0]
            pop_costs.append(split_cost(tour,inst))
        conv_wr.writerow([gen, min(pop_costs)])

    conv_fh.close()
    _flush_lnn_buffer()
    elapsed = time.time()-start_t

    # ---------- final metrics ----------
    final_costs=[]
    for ind in population:
        if encoding=="permutation":
            tour=ind["chromosome"]
        else:
            rk=ind["chromosome"]
            tour=[0]+sorted(range(1,inst.num_customers+1),
                             key=lambda i:rk[i-1])+[0]
        final_costs.append(split_cost(tour,inst))

    best  = min(final_costs)
    avg   = statistics.mean(final_costs)
    std   = statistics.pstdev(final_costs)
    bks   = inst.best_known_solution or 0.0
    avg_exc = ((avg-bks)/bks*100.0) if bks else 0.0

    return {
        "best_cost": best,
        "avg_cost":  avg,
        "std_dev":   std,
        "avg_exc":   avg_exc,
        "time":      elapsed,
    }


# ───────────────────────────  Per-instance driver  ─────────────────────────────
def _print_table(title:str, results:Dict[str,dict]):
    print(title)
    print("Operator | Best | Avg  | Exc%  | StdDev | Time")
    print("------------------------------------------------")
    for op in sorted(results):
        r=results[op]
        print(f"{op:8} | {r['best_cost']:5.1f} | {r['avg_cost']:5.1f} | "
              f"{r['avg_exc']:5.1f} | {r['std_dev']:5.1f} | {r['time']:6.1f}")

def run_instance_all_operators(inst: CVRPInstance):
    print(f"\n=== Instance {inst.name} ({inst.num_customers+1} nodes) ===")
    ALLOWED_OPS = {
        "OX":ox_crossover, "CX":cx_crossover, "AEX":aex_crossover,
        "LNN":lnn_crossover, "BRKGA":rk_brkga, "RK-2PX":rk_two_point
    }
    # WITHOUT MUTATION
    args0=[(inst,op,"permutation" if op in ("OX","CX","AEX","LNN") else "randomkey",
            fn,False,run+1)
           for run in range(NUM_RUNS) for op,fn in ALLOWED_OPS.items()]
    with concurrent.futures.ProcessPoolExecutor() as ex:
        metrics0=list(ex.map(parallel_run_ga,args0))
    res0={op:{"best_cost":min(m["best_cost"] for m in metrics0 if m["operator"]==op),
              "avg_cost":statistics.mean(m["avg_cost"] for m in metrics0 if m["operator"]==op),
              "std_dev":statistics.mean(m["std_dev"] for m in metrics0 if m["operator"]==op),
              "avg_exc":statistics.mean(m["avg_exc"] for m in metrics0 if m["operator"]==op),
              "time":statistics.mean(m["time"] for m in metrics0 if m["operator"]==op)}
          for op in ALLOWED_OPS}
    write_metrics_to_csv(metrics0, os.path.join(RESULTS_DIR,f"{inst.name}_summary_nomut.csv"))
    _print_table("--- Without Mutation ---",res0)

    # WITH MUTATION
    args1=[(inst,op,"permutation" if op in ("OX","CX","AEX","LNN") else "randomkey",
            fn,True,run+1)
           for run in range(NUM_RUNS) for op,fn in ALLOWED_OPS.items()]
    with concurrent.futures.ProcessPoolExecutor() as ex:
        metrics1=list(ex.map(parallel_run_ga,args1))
    res1={op:{"best_cost":min(m["best_cost"] for m in metrics1 if m["operator"]==op),
              "avg_cost":statistics.mean(m["avg_cost"] for m in metrics1 if m["operator"]==op),
              "std_dev":statistics.mean(m["std_dev"] for m in metrics1 if m["operator"]==op),
              "avg_exc":statistics.mean(m["avg_exc"] for m in metrics1 if m["operator"]==op),
              "time":statistics.mean(m["time"] for m in metrics1 if m["operator"]==op)}
          for op in ALLOWED_OPS}
    write_metrics_to_csv(metrics1, os.path.join(RESULTS_DIR,f"{inst.name}_summary_mut.csv"))
    _print_table("--- With Mutation ---",res1)


if __name__=="__main__":
    multiprocessing.set_start_method("spawn", force=True)
    TARGET={"E-n22-k4"}
    for inst in load_all_instances():
        if inst.name in TARGET:
            run_instance_all_operators(inst)
        else:
            print(f"Skipping {inst.name}")
    print(f"\nDone. Outputs in: {RESULTS_DIR}")