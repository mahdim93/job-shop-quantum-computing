import numpy as np
import dimod
from collections import defaultdict
from itertools import product
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.system import LeapHybridCQMSampler
import dwave.inspector
from scipy import stats
import random
import time
import pickle
import pandas as pd
import time
import json
import ast
import math
import copy
import itertools
import ctypes
import sys


from c_modules.c_utilities import *
# Load the shared library
if sys.platform.startswith("win"):
    c_library = ctypes.CDLL(r'./c_modules/my_program.dll')
elif sys.platform.startswith("linux"):
    c_library = ctypes.CDLL(r'./c_modules/my_program.so')  # Use '.so' on Linux
else:
    raise OSError("Unsupported operating system")
# Define the argument type for the C function
c_library.my_main.argtypes = [ctypes.c_char_p, ctypes.POINTER(Results), ctypes.c_float, ctypes.c_float]
c_library.my_main.restype = Results


def generate_qubo_job_path_optimization(n_jobs, paths, valid_indices, w, lb, ub, h, alpha = 0.1, numer_of_shots = 1000, howtosolve='quantum_hybrid'):
    Q = {}
    print(howtosolve)
    variables = [(paths[item]['job'], paths[item]['path']) for j in range(1, n_jobs + 1) for item in valid_indices]
    # Initialize QUBO with job-path binary variables
    for (j1, p1), (j2, p2) in product(variables, repeat=2):
        # Initialize QUBO entry if not yet defined
        if (p1, p2) not in Q:
            Q[(p1, p2)] = 0
            if p1 == p2:
                Q[(p1, p1)] = -h[1]  # Constant term ensuring one path is selected
            Q[(p1, p2)] += w.get((p1, p2), 0) * h[0]  # scaled by h[0]

        # Penalize if the same job is assigned to multiple paths
        if j1 == j2 and p1 != p2:
            Q[(p1, p2)] += h[1]  # penalty for assigning multiple paths to the same job

    if howtosolve == 'quantum_SA':
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        # print(Q)
        sampler = dimod.SimulatedAnnealingSampler()
        beta_range = [0.1, 10.0]
        response = sampler.sample(bqm, num_reads=numer_of_shots, beta_range=beta_range)
        qpu_access_time = 0
        run_time = 0

    else:
        start_time = time.time()
        sampler = LeapHybridSampler(solver={'category': 'hybrid'})
        response = sampler.sample_qubo(Q)
        # print(response.info)
        qpu_access_time = response.info['qpu_access_time'] / 1e6  # Convert to seconds
        run_time = response.info['run_time'] / 1e6  # Convert to seconds
        
        print(f"QPU Access Time: {qpu_access_time:.6f} seconds")
        print(f"Total Run Time: {run_time:.6f} seconds")

        # time_to_solve = response.info['timing']['qpu_access_time']  # Access time in microseconds
        # print(f"Time to solve: {time_to_solve / 1e6:.6f} seconds")

    solution_counts = defaultdict(list)
    for sample, energy in zip(response.record.sample, response.record.energy):
        solution = [var[1] for var, value in zip(Q.keys(), sample) if value > 0]
        # solutions.append((solution, np.round(energy + n_jobs * h[1],2)))
        solution_counts[np.round(energy + n_jobs * h[1],2)].append(solution)

    return np.round(response.first.energy + n_jobs*h[1],2), [], qpu_access_time, run_time


def generate_all_paths_for_jobs_v4(instance, ub, lb, lamda, epsilon, v_penalty, time_limit, numer_of_shots=1000, howtosolve='digital'):
    
    loginfo = {}
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    tau = instance['num_time_slots']  # available time priods
    old_lb = 0
    for iter in range(1):
        
        ctype_instance = convertInstanceToCtype(instance)
        if iter > 1:
            ctype_results = c_library.my_main(ctype_instance, ctypes.byref(ctype_results), lb, ub)
        else:
            ctype_results = c_library.my_main(ctype_instance, None, lb, ub)
        
        results = ResultsToDict(ctype_results, num_jobs, num_machines)
    
        valid_indices = set(results["paths"].keys())
        paths = results["paths"]
        n_overlaps_after = 0;
        for p in valid_indices:
            n_overlaps_after += len({i for i in paths[p]["overlaps"] if i in valid_indices})
            paths[p]["total_costs"] = sum(paths[p]["costs"])
        n_overlaps_after = int(n_overlaps_after/2)
        print("n_overlaps_after", n_overlaps_after)
        
        if howtosolve == 'digital':
            objective_value, variables, total_solving_time, problem_generation_time = \
                solve_binary_optimization_v2(num_jobs, paths, results["num_paths"], old_lb, ub, time_limit, valid_indices)
        else:
            weights = {}
            for i in valid_indices:
                weights[(i, i)] = paths[i]["total_costs"]
            for p in valid_indices:
                filtered = {i for i in paths[p]["overlaps"] if i in valid_indices}
                for j in filtered:
                    weights[(p, j)] = v_penalty
            objective_value, variables, total_solving_time, problem_generation_time = \
                generate_qubo_job_path_optimization(num_jobs, paths, valid_indices, weights, old_lb, ub, [1, v_penalty], numer_of_shots=numer_of_shots, howtosolve=howtosolve)
        print(variables)
        print("Iter: ", iter, " Objective value is:", objective_value, "lower bound is:", lb, 'upper bound is', ub)
        loginfo[iter] = [iter, results['num_paths_before'], results['num_overlaps_before'], len(paths), n_overlaps_after, results['time_to_find_paths'], results['time_to_find_overlaps'], 
                         results['time_to_remove_paths_tom'], results['time_to_remove_paths_mahdi'], 
                         problem_generation_time, total_solving_time, objective_value]
    
        if objective_value is not None:
            ub = min(ub, objective_value)
    
        if ub - lb <= 0:
            break
        elif objective_value <= ub:
            old_lb = lb
            lb = objective_value
        else:
            old_lb = lb
            lb = lb*lamda + (1 - lamda)*ub
            if abs(lb - ub) <= epsilon:
                lb = ub
    return paths, [], objective_value, variables, loginfo, total_solving_time

    