#!/usr/bin/env python3
"""
smoke_test.py - fast end-to-end sanity check for LNN_crossover_CVRP.

Runs in seconds, not the ~10 minutes a real `python GA.py` takes. Checks the
things that have actually broken: the environment, instance loading, the
distance/Split evaluator against published reference solutions, the LNN
forward pass, every crossover operator, and a miniature GA run.

Usage:
    python smoke_test.py            # all checks
    python smoke_test.py -v         # show per-instance detail
Exit code 0 if everything passed, 1 otherwise.
"""
from __future__ import annotations
import os, sys, time, traceback

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv

# Project root = wherever cvrp_loader.py lives: the cwd, or this script's dir.
ROOT = next((d for d in (os.getcwd(), os.path.dirname(os.path.abspath(__file__)))
             if os.path.isfile(os.path.join(d, "cvrp_loader.py"))), None)
if ROOT is None:
    sys.exit("error: cannot find cvrp_loader.py - run this from the project root")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str]] = []
_t0 = time.time()

if sys.stdout.isatty():
    C = {PASS: "\033[32m", FAIL: "\033[31m", SKIP: "\033[33m", "R": "\033[0m", "B": "\033[1m"}
else:
    C = {PASS: "", FAIL: "", SKIP: "", "R": "", "B": ""}


def record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    print(f"  {C[status]}{status}{C['R']}  {name}" + (f" - {detail}" if detail else ""))


def check(name):
    """Decorator: run a check, turn a return value into PASS/detail, catch errors."""
    def deco(fn):
        def run(*a, **kw):
            try:
                detail = fn(*a, **kw)
                record(name, PASS, detail or "")
                return True
            except _Skip as e:
                record(name, SKIP, str(e))
                return None
            except AssertionError as e:
                record(name, FAIL, str(e) or "assertion failed")
                return False
            except Exception as e:
                record(name, FAIL, f"{type(e).__name__}: {e}")
                if VERBOSE:
                    traceback.print_exc()
                return False
        return run
    return deco


class _Skip(Exception):
    pass


# ---------------------------------------------------------------- 1. environment
print(f"\n{C['B']}1. Environment{C['R']}")

HAVE_TORCH = False


@check("python >= 3.10")
def c_python():
    assert sys.version_info[:2] >= (3, 10), f"found {sys.version.split()[0]}"
    return sys.version.split()[0]


@check("numpy importable")
def c_numpy():
    import numpy
    return numpy.__version__


@check("torch importable")
def c_torch():
    global HAVE_TORCH
    import torch
    HAVE_TORCH = True
    dev = "cpu"
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        dev = f"cuda {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    return f"{torch.__version__} -> {dev}"


c_python(); c_numpy(); c_torch()

# ---------------------------------------------------------------- 2. loader
print(f"\n{C['B']}2. Instance loading{C['R']}")
INSTANCES = []


@check("load_all_instances()")
def c_load():
    import cvrp_loader
    global INSTANCES
    INSTANCES = cvrp_loader.load_all_instances()
    assert INSTANCES, "no instances returned"
    return f"{len(INSTANCES)} instances"


@check("instance fields are well formed")
def c_fields():
    import numpy as np
    if not INSTANCES:
        raise _Skip("loader failed")
    for inst in INSTANCES:
        d = np.array(inst.distance_matrix)
        n = inst.num_customers
        assert d.shape == (n + 1, n + 1), f"{inst.name}: matrix {d.shape} vs customers {n}"
        assert len(inst.demands) == n + 1, f"{inst.name}: {len(inst.demands)} demands vs {n + 1} nodes"
        assert inst.vehicle_capacity > 0, f"{inst.name}: capacity {inst.vehicle_capacity}"
        assert inst.demands[0] == 0, f"{inst.name}: depot demand {inst.demands[0]} != 0"
        assert max(inst.demands) <= inst.vehicle_capacity, f"{inst.name}: a demand exceeds capacity"
        if VERBOSE:
            print(f"        {inst.name:12} n={n:4} cap={inst.vehicle_capacity:6} "
                  f"demand={sum(inst.demands):7} bks={inst.best_known_solution}")
    return f"{len(INSTANCES)} instances checked"


c_load(); c_fields()

# ------------------------------------------------- 3. independent solution check
print(f"\n{C['B']}3. Reference solutions (independent route checker){C['R']}")


def _parse_sol(path):
    routes = []
    declared = None
    with open(path) as fh:
        for line in fh:
            low = line.strip().lower()
            if low.startswith("route"):
                routes.append([int(x) for x in line.split(":")[1].split()])
            elif low.startswith("cost"):
                declared = float(line.split()[-1])
    return routes, declared


@check("reference routes reproduce the declared cost")
def c_refcost():
    import numpy as np
    if not INSTANCES:
        raise _Skip("loader failed")
    checked = 0
    for inst in INSTANCES:
        sol = os.path.join(ROOT, "Data", "SCVRP", f"{inst.name}.sol")
        if not os.path.isfile(sol):
            continue
        routes, declared = _parse_sol(sol)
        if not routes or declared is None:
            continue
        d = np.array(inst.distance_matrix)

        served = sorted(c for r in routes for c in r)
        assert served == list(range(1, inst.num_customers + 1)), \
            f"{inst.name}: routes do not serve each customer exactly once"

        total = 0.0
        for r in routes:
            load = sum(inst.demands[c] for c in r)
            assert load <= inst.vehicle_capacity, \
                f"{inst.name}: route load {load} > capacity {inst.vehicle_capacity}"
            full = [0] + r + [0]
            total += sum(d[full[k]][full[k + 1]] for k in range(len(full) - 1))

        assert abs(total - declared) < 1e-6, \
            f"{inst.name}: recomputed {total} but .sol declares {declared}"
        if VERBOSE:
            print(f"        {inst.name:12} {len(routes)} routes, cost {total:.0f} == declared")
        checked += 1
    assert checked, "no .sol files found to check"
    return f"{checked} reference solutions verified"


@check("split_cost() reproduces the reference optimum")
def c_split():
    if not INSTANCES:
        raise _Skip("loader failed")
    if not HAVE_TORCH:
        raise _Skip("needs torch (GA.py imports it)")
    import GA
    checked = worse = 0
    for inst in INSTANCES:
        sol = os.path.join(ROOT, "Data", "SCVRP", f"{inst.name}.sol")
        if not os.path.isfile(sol):
            continue
        routes, declared = _parse_sol(sol)
        if not routes or declared is None:
            continue
        giant = [0] + [c for r in routes for c in r] + [0]
        got = GA.split_cost(giant, inst)
        assert got <= declared + 1e-6, \
            f"{inst.name}: split_cost {got} exceeds reference {declared}"
        if got < declared - 1e-6:
            worse += 1
        checked += 1
    note = f"{checked} checked"
    if worse:
        note += f"; {worse} below the reference value - investigate"
    return note


c_refcost(); c_split()

# ---------------------------------------------------------------- 4. LNN model
print(f"\n{C['B']}4. LNN model{C['R']}")


@check("weights load and forward pass yields a valid permutation")
def c_model():
    if not HAVE_TORCH:
        raise _Skip("needs torch")
    if not INSTANCES:
        raise _Skip("loader failed")
    import torch, random
    from GA_LNN.lnn.model import LiquidCrossover
    from GA_LNN.lnn.encoder import encode_parents
    from GA_LNN.lnn.decoder import scores_to_perm

    inst = INSTANCES[0]
    n = inst.num_customers
    model = LiquidCrossover(in_dim=3)
    ckpt = os.path.join(ROOT, "GA_LNN", "lnn", "lnn_hyx.pt")
    loaded = "random init"
    if os.path.isfile(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        loaded = "checkpoint"
    model.eval()

    p1 = random.sample(range(1, n + 1), n)
    p2 = random.sample(range(1, n + 1), n)
    feats = encode_parents(p1, p2, inst)
    assert feats.shape == (n, 3), f"encoder gave {tuple(feats.shape)}, expected ({n}, 3)"
    assert torch.isfinite(feats).all(), "encoder produced non-finite features"

    with torch.no_grad():
        scores = model(feats)
    assert scores.shape == (n,), f"model gave {tuple(scores.shape)}, expected ({n},)"
    assert torch.isfinite(scores).all(), "model produced non-finite scores"

    perm = scores_to_perm(scores)
    assert sorted(perm) == list(range(1, n + 1)), "decoder did not return a valid permutation"

    params = sum(p.numel() for p in model.parameters())
    return f"{params:,} params, {loaded}, n={n}"


c_model()

# ------------------------------------------------------------ 5. operators
print(f"\n{C['B']}5. Crossover operators{C['R']}")


@check("every operator returns a valid chromosome")
def c_ops():
    if not HAVE_TORCH:
        raise _Skip("needs torch (GA.py imports it)")
    if not INSTANCES:
        raise _Skip("loader failed")
    import random, GA
    inst = INSTANCES[0]
    n = inst.num_customers

    perm_ops = {"PMX": GA.pmx_crossover, "OX": GA.ox_crossover, "CX": GA.cx_crossover,
                "AEX": GA.aex_crossover, "LNN": GA.lnn_crossover}
    rk_ops = {"BRKGA": GA.rk_brkga, "RK-U": GA.rk_uniform,
              "RK-GR": GA.rk_greedy, "RK-2PX": GA.rk_two_point}

    for name, fn in perm_ops.items():
        p1 = [0] + random.sample(range(1, n + 1), n) + [0]
        p2 = [0] + random.sample(range(1, n + 1), n) + [0]
        c = fn(p1, p2, inst)
        assert c[0] == 0 and c[-1] == 0, f"{name}: chromosome not depot-delimited"
        body = c[1:-1]
        assert sorted(body) == list(range(1, n + 1)), \
            f"{name}: body is not a permutation of 1..{n} (len {len(body)}, dups {len(body)-len(set(body))})"
        assert GA.split_cost(c, inst) > 0, f"{name}: non-positive split cost"

    for name, fn in rk_ops.items():
        u = [random.random() for _ in range(n)]
        v = [random.random() for _ in range(n)]
        c = fn(u, v, inst)
        assert len(c) == n, f"{name}: length {len(c)} != {n}"
        assert all(0.0 <= x <= 1.0 for x in c), f"{name}: keys outside [0,1]"

    return f"{len(perm_ops)} permutation + {len(rk_ops)} random-key operators"


c_ops()

# ------------------------------------------------------------ 6. miniature GA
print(f"\n{C['B']}6. Miniature GA run{C['R']}")


@check("run_ga_for_instance() completes for all wired operators")
def c_ga():
    if not HAVE_TORCH:
        raise _Skip("needs torch")
    if not INSTANCES:
        raise _Skip("loader failed")
    import GA, tempfile
    inst = min(INSTANCES, key=lambda i: i.num_customers)

    orig = (GA.POP_SIZE, GA.MAX_GENERATIONS, GA.RESULTS_DIR)
    GA.POP_SIZE, GA.MAX_GENERATIONS = 6, 3
    tmp = tempfile.mkdtemp(prefix="smoke_")
    GA.RESULTS_DIR = tmp
    try:
        ops = {"OX": ("permutation", GA.ox_crossover), "CX": ("permutation", GA.cx_crossover),
               "AEX": ("permutation", GA.aex_crossover), "LNN": ("permutation", GA.lnn_crossover),
               "BRKGA": ("randomkey", GA.rk_brkga), "RK-2PX": ("randomkey", GA.rk_two_point)}
        for name, (enc, fn) in ops.items():
            for mut in (False, True):
                m = GA.run_ga_for_instance(inst, name, enc, fn,
                                           apply_mutation=mut, run_id=1)
                # the aggregation step in run_instance_all_operators needs this key
                assert "operator" in m, f"{name}: metrics dict is missing 'operator'"
                assert m["operator"] == name, f"{name}: operator key is {m['operator']!r}"
                for k in ("best_cost", "avg_cost", "std_dev", "time"):
                    assert k in m, f"{name}: metrics missing {k!r}"
                    assert m[k] == m[k], f"{name}: {k} is NaN"
                assert m["best_cost"] > 0, f"{name}: non-positive best cost"
                assert m["best_cost"] <= m["avg_cost"] + 1e-9, \
                    f"{name}: best {m['best_cost']} > avg {m['avg_cost']}"
        return f"{inst.name}, {len(ops)} operators x 2 regimes, pop=6 gens=3"
    finally:
        GA.POP_SIZE, GA.MAX_GENERATIONS, GA.RESULTS_DIR = orig


c_ga()

# ---------------------------------------------------------------- summary
elapsed = time.time() - _t0
n_pass = sum(1 for _, s, _ in _results if s == PASS)
n_fail = sum(1 for _, s, _ in _results if s == FAIL)
n_skip = sum(1 for _, s, _ in _results if s == SKIP)

print(f"\n{C['B']}Summary{C['R']}")
print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped   ({elapsed:.1f}s)")
if n_fail:
    print(f"\n  {C[FAIL]}failed checks:{C['R']}")
    for name, s, detail in _results:
        if s == FAIL:
            print(f"    - {name}: {detail}")
if n_skip:
    print(f"\n  {C[SKIP]}skipped:{C['R']}")
    for name, s, detail in _results:
        if s == SKIP:
            print(f"    - {name}: {detail}")
sys.exit(1 if n_fail else 0)
