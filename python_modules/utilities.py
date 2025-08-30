import numpy as np
import dimod
from collections import defaultdict
from itertools import product
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.system import LeapHybridCQMSampler
import dwave.inspector
from pyomo.environ import *
from scipy import stats
from docplex.cp.model import CpoModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import time
import pickle
import pandas as pd
import time
import json
import ast
import math
import copy
import pandas as pd
from joblib import Parallel, delayed
import itertools
import ctypes
import sys

# Carugno et al. (2022) packages to generate random Jobshop instances
from python_modules.job_shop_experiment import JobShopExperiment

from c_modules.c_utilities import *
# Load the shared library
if sys.platform.startswith("win"):
    c_library = ctypes.CDLL(r'./c_modules/my_program.dll')
elif sys.platform.startswith("linux"):
    c_library = ctypes.CDLL(r'./c_modules/my_program.so')  # Use '.so' on Linux
else:
    raise OSError("Unsupported operating system")
# Define the argument type for the C function
c_library.my_main.argtypes = [ctypes.c_char_p, ctypes.POINTER(Results), ctypes.c_float, ctypes.c_float, ctypes.c_float]
c_library.my_main.restype = Results


def plot_gantt_chart(num_jobs, sequences, start_times, end_times):
    """
    Plots a Gantt chart for job scheduling.
    
    Parameters:
    - num_jobs: Total number of jobs.
    - sequences: Dictionary with job sequences, where each value is a list of machines for operations.
    - start_times: Dictionary with job start times, where each value is a dict of operation index to start time.
    - end_times: Dictionary with job end times, where each value is a dict of operation index to end time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define unique colors for each job
    colors = plt.cm.tab10.colors
    
    # Iterate through each job
    for job in range(1, num_jobs + 1):
        for op_index, start_time in start_times[job].items():
            end_time = end_times[job][op_index]
            operation = sequences[job][op_index - 1]  # Machine for the operation
            
            # Add a rectangle to represent the operation
            ax.barh(operation, end_time - start_time, left=start_time, color=colors[job % len(colors)], edgecolor='black', height=0.8, label=f"Job {job}" if op_index == 1 else "")
            
            # Add text for the operation
            ax.text((start_time + end_time) / 2, operation, f"J{job}-O{op_index}", ha='center', va='center', color='white', fontsize=8)
    
    # Add labels and grid
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Gantt Chart')
    ax.set_yticks(range(1, max(max(sequences.values(), key=max)) + 1))  # Ensure all machines are shown
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', title="Jobs")
    
    plt.tight_layout()
    plt.show()

def generate_random_problem(number_of_jobs = 2, number_of_machine = 2, range_of_processing_times = [1, 2], order_of_jobs = 'cycle'):    

    earlycost_lowerbound = 0
    earlycost_upperbound = 0.5
    tardyduedate_factor = 1.5
    
    # Generate a basic JJS instance
    JJS = JobShopExperiment ('data')
    instance = JJS.get_problem(machines=number_of_machine,
                    jobs=number_of_jobs, 
                    ops=number_of_machine, 
                    time=range_of_processing_times, 
                    ordering=order_of_jobs)

    # Convert Carugno basic instance to be ready for our coding
    num_jobs = number_of_jobs
    num_machines = number_of_machine
    
    J = range(1, num_jobs+1)  # jobs
    M = range(1, num_machines + 1)   # machines
    
    p = {} # (j, m): processing times
    d = {} # j: due dates
    w = {}  # j: tardiness weights
    E = {}  # (j, m): earliness weights
    aggE = {} # {(j, m)} aggregated earliness weights for each machine
    sequence = {} # {job: []} sequence of machines for each job
    f = {} # j: last machine
    mp = {} # (j, m): previous machine


    for i, tasks in instance.items():
        tmpsum = 0
        job = int("".join([char for char in i if char.isnumeric()])) + 1
        tmp2 = 0
        processTime = 0
        perv = 0
        machine = 0
        # Initialize an empty list for each job in the sequence
        if job not in sequence:
            sequence[job] = []
            
        for j in tasks:
            tmp2 = j[0]
            perv = machine
            machine = int("".join([char for char in tmp2 if char.isnumeric()])) + 1
            processTime = j[1]
            p[(job, machine)] = processTime
            mp[(job, machine)] = perv
            sequence[job].append(machine)
            tmpsum = tmpsum + processTime
            randn = random.uniform(earlycost_lowerbound, earlycost_upperbound)
            E[(job, machine)] = round(randn, 3)
            if perv > 0:
                aggE[(job, machine)] = aggE[(job, perv)] + E[(job, machine)] 
            else:
                aggE[(job, machine)] = E[(job, machine)] 
        f[job] = machine
        # p[(job, num_machines)] = 0
        # E[(job, num_machines)] = 0
        mp[(job, num_machines)] = machine
        d[job] = round(tmpsum*tardyduedate_factor)
        if job/num_jobs <= 0.20:
            w[job] = 4
        elif job/num_jobs <= 0.80:
            w[job] = 2
        else:
            w[job] = 1
    
    maximum_makespan = sum(p.values())
    num_time_slots = maximum_makespan
    T = range(1, num_time_slots+1)  # time slots


    # Variables to save
    variables_to_save = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "num_time_slots": num_time_slots,
        "p": p,
        "d": d,
        "w": w,
        "E": E,
        "aggE": aggE,
        "sequence": sequence,
        "mp": mp,
        "f": f
    }
    return variables_to_save

def load_a_problem(file_path):
    import pandas as pd

    # Read the Excel file
    df = pd.read_excel(file_path = "my_instance.xlsx")
    
    # Initialize the dictionary to store the processed data
    variables = {
        "num_jobs": 0,
        "num_machines": 0,
        "p": {},
        "d": {},
        "w": {},
        "E": {},
        "sequence": {}
    }
    
    # Define the number of jobs and machines based on the data
    variables["num_jobs"] = df['Job j'].nunique()
    variables["num_machines"] = df['Machine'].nunique()
    
    # Process the "Proc Time" (processing times)
    for index, row in df.iterrows():
        variables["p"][(int(row['Job j']), int(row['Machine']))] = row['Proc Time']
    
    # Process the "d" (due dates) and "w" (weights)
    for index, row in df.iterrows():
        if row['Job j'] not in variables["d"]:
            variables["d"][int(row['Job j'])] = row['dj']
        if row['Job j'] not in variables["w"]:
            variables["w"][int(row['Job j'])] = row['Wj']
    
    # Process the "E" (early times) values
    for index, row in df.iterrows():
        variables["E"][(int(row['Job j']), int(row['Machine']))] = row['Eij']
    
    # Process the "Machine" (machine assignment per job)
    for index, row in df.iterrows():
        if row['Job j'] not in variables["sequence"]:
            variables["sequence"][int(row['Job j'])] = []
        variables["sequence"][int(row['Job j'])].append(int(row['Machine']))
    
    # Print the dictionary to verify the result
    variables['num_time_slots'] = int(sum(variables['p'].values()))
    variables["aggE"] = []
    return variables

def load_a_problem_2(file_path = "my_instance.xlsx"):
    variables = {
        "num_jobs": 2,
        "num_machines": 3,
        "num_time_slots": 18,
        "p": {(1,1):4, (1,2):4, (1,3):4, (2,3):2, (2,1):3, (2,2):5},
        "d": {1:12, 2:14},
        "w": {1:5, 2:2},
        "E": {(1,1):1, (1,2):0.5, (1,3):1.5, (2,3):0.6, (2,1):1, (2,2):0.9},
        "aggE": {(1,1):1, (1,2):1.5, (1,3):3, (2,3):0.6, (2,1):1.6, (2,2):2.5},
        "sequence": {1:[1, 2, 3], 2:[3, 1, 2]},
          "mp": {
            (1, 1): 0,
            (1, 2): 1,
            (1, 3): 2,
            (2, 1): 3,
            (2, 2): 1,
            (2, 3): 0
          },
          "f": {
            1: 3,
            2: 2
          }
    }
    return variables

def ATC_rule_for_ub(variables_):
    variables = copy.deepcopy(variables_)
    num_jobs = variables["num_jobs"]
    num_machines = variables["num_machines"]
    processing_times = variables["p"]
    due_dates = variables["d"]
    weights = variables["w"]
    sequences = variables["sequence"]
    es = variables["E"]
    
    # Initialize start and end times for operations
    start_times = defaultdict(dict)
    end_times = defaultdict(dict)
    machine_available_time = {m: 0 for m in range(1, num_machines + 1)}
    job_next_op = {j: 1 for j in range(1, num_jobs + 1)}  # Next operation index for each job
    job_available_time = {j: 0 for j in range(1, num_jobs + 1)}
    
    # Helper to calculate expected waiting time for a job
    def calculate_expected_waiting_time(machine, job):
        return max(machine_available_time[machine] - job_available_time[job], 0)
    
    # ATC priority index calculation function
    def calculate_priority_index_ATC(Tij, Di, mi, remaining_workload, expected_waiting_time, tau, T_bar, localduedate, tardiness, k):
        slack_term = Di - expected_waiting_time - remaining_workload - tau - Tij
        due_proc = localduedate - Tij - tau
        # exponent = -max(slack_term / (k * T_bar), 0)
        exponent = -max(due_proc / (k * T_bar), 0)
        Z_i_prime = (tardiness / Tij) * np.exp(exponent)
        return Z_i_prime
    
    # Initialize job completion status
    jobs_remaining = set(range(1, num_jobs + 1))
    
    # Scheduling loop until all jobs are scheduled
    current_time = 0
    while jobs_remaining:
        available_operations = []
    
        # Collect all available operations
        for job in jobs_remaining:
            op_index = job_next_op[job]
            if op_index > len(sequences[job]):
                continue
            
            machine = sequences[job][op_index - 1]  # Get the actual operation number
            processing_time = processing_times[(job, machine)]
            due_date = due_dates[job]
            expected_waiting_time = calculate_expected_waiting_time(machine, job)
            
            # Calculate remaining workload (sum of processing times for the remaining operations)
            remaining_workload = sum(processing_times[(job, seq)] for seq in sequences[job][op_index-1:])
            T_bar = 0
            for j in range(1, num_jobs+1):
                T_bar = processing_times[(j, machine)]
            # T_bar = remaining_workload/len(sequences[job][op_index-1:])
            localduedate = due_date - remaining_workload + processing_time
            tardiness = weights[job]
            # Calculate ATC priority index
            priority_index = calculate_priority_index_ATC(processing_time, due_date, machine, remaining_workload, expected_waiting_time, current_time, T_bar, localduedate, tardiness, 2)
    
            available_operations.append((job, machine, priority_index))
    
        # Select the operation with the highest priority for the current time
        while True:
            # Filter available operations based on the current time
            filtered_operations = [
                (job, machine, priority_index)
                for job, machine, priority_index in available_operations
                if job_available_time[job] <= current_time and machine_available_time[machine] <= current_time
            ]
            
            # If no operations are available, increment current_time
            # print(current_time, job_available_time, machine_available_time)
            if not filtered_operations:
                next_job_time = min(job_available_time[job] for job in range(1, num_jobs + 1) if job_available_time[job] > current_time)
                next_machine_time = min(machine_available_time[machine] for machine in range(1, num_machines + 1) if machine_available_time[machine] > current_time)
                current_time = min(next_job_time, next_machine_time)
            else:
                break  # Exit the loop once filtered_operations has at least one element

        selected_job, selected_machine, min_priority = max(filtered_operations, key=lambda x: x[2], default=(None, None, -math.inf))
        # Schedule the selected operation
        if selected_job is not None:
            op_index = job_next_op[selected_job]
            processing_time = processing_times[(selected_job, selected_machine)]
    
            # Calculate start and end times for this operation
            start_time = max(machine_available_time[selected_machine], job_available_time[selected_job])
            end_time = start_time + processing_time
    
            # Update records
            start_times[selected_job][op_index] = start_time
            end_times[selected_job][op_index] = end_time
            machine_available_time[selected_machine] = end_time
            job_available_time[selected_job] = end_time
    
            # Move to the next operation for the job
            job_next_op[selected_job] += 1
            if job_next_op[selected_job] > len(sequences[selected_job]):
                jobs_remaining.remove(selected_job)
    
    # Calculate tardiness and weighted tardiness
    total_weighted_tardiness = 0
    delivery_times = {}
    for job in range(1, num_jobs + 1):
        job_completion_time = max(end_times[job].values())
        delivery_times[job] = job_completion_time
        tardiness = max(0, job_completion_time - due_dates[job])
        weighted_tardiness = tardiness * weights[job]
        total_weighted_tardiness += weighted_tardiness
    
    # Print start and end times for each operation
    # print("\nSchedule:")
    # for job in range(1, num_jobs + 1):
    #     for op_index, start_time in start_times[job].items():
    #         end_time = end_times[job][op_index]
    #         operation = sequences[job][op_index - 1]
    #         print(f"Job {job}, Operation {op_index} on Machine {operation} - Start: {start_time}, End: {end_time}")
    
    # print(f"\nTotal Weighted Tardiness: {total_weighted_tardiness}")
    # plot_gantt_chart(num_jobs, sequences, start_times, end_times)

    # Calculate total sums for each job
    job_sums = {j: 0 for j in range(1, num_jobs + 1)}  # Initialize total sums for each job
    seq = []
    for job in range(1, num_jobs + 1):
        for op_index in start_times[job]:
            machine = sequences[job][op_index - 1]
            # Calculate total processing time from current operation to last operation
            total_processing_time = sum(processing_times[(job, idx)] for idx in sequences[job][op_index - 1:])
            
            job_sums[job] += es[(job, machine)] * (
                (max(delivery_times[job], due_dates[job]) - start_times[job][op_index]) - total_processing_time
            )
            for j2 in range(job+1, num_jobs + 1):
                for op_index2 in start_times[j2]:
                    m2 = sequences[j2][op_index2 - 1]
                    if machine == m2:
                        if start_times[job][op_index] < start_times[j2][op_index2]:
                            seq.append(((job, op_index),(j2, op_index2)))
                        else:
                            seq.append(((j2, op_index2), (job, op_index)))
    # print(f"\nTotal Weighted Tardiness: {total_weighted_tardiness}")
    # for job in range(1, num_jobs + 1):
        # print(f"Total Sum for Job {job}: {job_sums[job]}")
    # Overall total_sum
    total_sum = sum(job_sums.values())
    # print(f"\nUpper bound is: {total_sum + total_weighted_tardiness}")
    objValue, solving_time, max_time= solvingJJSbyCP(variables, seq, time_limit = 300)
    
    # print(total_sum + total_weighted_tardiness)
    return objValue, max_time

def solvingJJSbyCP(variables_, seq = 0, whichobj='wet', time_limit = 300, solving_time2 = 0):
    variables = copy.deepcopy(variables_)
    mdl = CpoModel()
    # Parameters
    nj = variables["num_jobs"]  # Number of jobs
    nm = variables["num_machines"]  # Number of machines
    tau = variables['num_time_slots']
    I = range(1, nj*nm+1)
    M = range(1, nm+1)
    
    # Initialize variables
    e = variables["E"]
    es = variables["aggE"]
    w = variables["w"]
    d = variables["d"]
    mi = variables["sequence"]
    t = variables["p"]
    t2 = t
    ei = list(e.values())
    t = list(t.values())
    mim = list(mi.values())
    mim = [item for row in mim for item in row]
    # Decision variables
    tt = {i: mdl.integer_var(0, tau*3, name=f"tt_{i}") for i in I}  # start times of tasks
    
    # Expressions
    duedate = {j: mdl.max(0, (tt[(j-1)*nm+nm] + t[(j-1)*nm+nm-1]) - d[j]) for j in range(1, nj + 1)}
    duedatepen = mdl.sum(w[j] * duedate[j] for j in range(1, nj + 1))
    
    earlypen = mdl.sum(ei[(j-1)*nm+m-1] * (mdl.max((tt[(j-1)*nm+nm] + t[(j-1)*nm+nm-1]) - tt[(j-1)*nm+m], d[j] - tt[(j-1)*nm+m])
                       - sum(value for (x, y), value in t2.items() if x == j and y in mi[j][m-1:])
                                          )
                       for j in range(1, nj + 1) for m in range(1, nm + 1))
    
        
    # Constraints
    for i in range(1, nj+1):
        for m in range(1, nm):
            mdl.add(tt[(i-1)*nm + m] + t[(i-1)*nm + m-1] <= tt[(i-1)*nm + m+1])
    
    for i in I:
        for j in I:
            if mim[i-1] == mim[j-1] and i < j:
                mdl.add((tt[i] + t[i-1] <= tt[j]) | (tt[j] + t[j-1] <= tt[i]))

    if seq:
        mdl.add(mdl.minimize(duedatepen + earlypen))
        for item in seq:
            j1 = item[0][0]
            m1 = item[0][1]
            j2 = item[1][0]
            m2 = item[1][1]
            mdl.add(tt[(j1-1)*nm + m1] + t[(j1-1)*nm + m1-1] <= tt[(j2-1)*nm + m2])
    else:
        if whichobj=='wet':
            mdl.add(mdl.minimize(duedatepen + earlypen))
        else:
            mdl.add(mdl.minimize(duedatepen))
    print(whichobj)
    # Solve the model
    start_time = time.time()
    solution = mdl.solve(agent='local', execfile='/opt/ibm/ILOG/CPLEX_Studio221/cpoptimizer/bin/x86-64_linux/cpoptimizer', 
                               log_output=None, TimeLimit=time_limit)
    solving_time = time.time() - start_time
    
    # Display results
    if solution:
        # print("Objective Value:", np.round(solution.get_objective_value(),2))
        # for i in I:
        #     print(f"Task {i}: Start = {solution.get_var_solution(tt[i])}")
        start_times = defaultdict(dict)
        end_times = defaultdict(dict)
        for j in range(1, nj + 1):
            for m in M:
                start_times[j][m] = solution.get_var_solution(tt[(j-1)*nm + m]).get_value()
                end_times[j][m] = solution.get_var_solution(tt[(j-1)*nm + m]).get_value() + t2[(j,mi[j][m-1])]
        # plot_gantt_chart(nj, mi, start_times, end_times)
    else:
        print("No solution found within the time limit.")

    if solution:        
        # Collect the start times for each task
        start_times = {i: solution.get_var_solution(tt[i]).get_value() for i in I}  # Convert to int or float
        
        # Map tasks to machines and sort by start time
        machine_tasks = {m: [] for m in M}  # Initialize empty list for each machine
        for i in I:
            job = (i - 1) // nm + 1  # Job number
            machine = mim[i - 1]    # Machine number
            machine_tasks[machine].append((job, i, start_times[i]))  # Store (job, op_index, start_time)
        
        # Create the sequence list
        seq_list = []
        for job in range(1, nj + 1):  # For each job
            for op_index in range(1, nm + 1):  # For each operation in the job
                machine = mi[job][op_index - 1]
                # Compare the current task with all other tasks that are on the same machine
                for job2 in range(job + 1, nj + 1):  # Compare with subsequent jobs
                    for op_index2 in range(1, nm + 1):  # Compare with subsequent operations
                        m2 = mi[job2][op_index2 - 1]
                        if machine == m2:
                            if start_times[(job - 1) * nm + op_index] < start_times[(job2 - 1) * nm + op_index2]:
                                seq_list.append(((job, op_index), (job2, op_index2)))
                            else:
                                seq_list.append(((job2, op_index2), (job, op_index)))

    # Print duedate values and duedatepen
    computed_duedate = {}
    duedatepen_value = 0
    
    for j in range(1, nj + 1):
        # index of last operation for job j
        idx = (j - 1) * nm + nm
        start_time = solution.get_var_solution(tt[idx]).get_value()
        processing_time = t[idx - 1]  # since t is 0-indexed
        completion_time = start_time + processing_time
        lateness = max(0, completion_time - d[j])
        computed_duedate[j] = lateness
        duedatepen_value += w[j] * lateness
    
    print(f"Total Duedate Penalty: {duedatepen_value:.1f}")


    if seq:
        return np.round(solution.get_objective_value(),2), solving_time+solving_time2, max(end_times[j][m] for j in end_times for m in end_times[j])
    elif whichobj != 'wet':
        return solvingJJSbyCP(variables, seq_list, time_limit = time_limit, solving_time2 = solving_time), solving_time, max(end_times[j][m] for j in end_times for m in end_times[j])
    else:
        return np.round(solution.get_objective_value(),2), solving_time+solving_time2, max(end_times[j][m] for j in end_times for m in end_times[j])


def solvingJJSbyCPLEX(variables_, seq = 0, whichobj='wet', time_limit = 300):
    from docplex.mp.model import Model
    variables = copy.deepcopy(variables_)
    mdl = Model(name="Job Scheduling")
    # Parameters
    nj = variables["num_jobs"]  # Number of jobs
    nm = variables["num_machines"]  # Number of machines
    tau = variables['num_time_slots']
    I = range(1, nj*nm+1)
    M = range(1, nm+1)
    J = range(1, nj+1)
    bigM = nj*nm*tau
    # Initialize variables
    e = variables["E"]
    es = variables["aggE"]
    w = variables["w"]
    d = variables["d"]
    mi = variables["sequence"]
    t = variables["p"]
    t2 = t
    ei = list(e.values())
    t = list(t.values())
    mim = list(mi.values())
    mim = [item for row in mim for item in row]

    mdl = Model(name="Job Scheduling")

    # Decision variables
    tt = mdl.continuous_var_matrix(J, M, lb=0, name="tt")  # Start times
    ff = mdl.continuous_var_dict(J, lb=0, name="ff")  # Exceeded times
    ss = mdl.continuous_var_matrix(J, M, lb=0, name="ss")  # Dual variables
    x = mdl.binary_var_dict(((j1, m1, j2, m2) for j1 in J for m1 in M for j2 in J for m2 in M), name="x")
    # Objective function
    mdl.minimize(
        mdl.sum(w[j] * ff[j] for j in J) +
        mdl.sum(es[(j,mi[j][m-1])] * ss[j, m] for m in M for j in J)
    )

    # Precedence constraints
    for m in M:
        for i in range(1, nj + 1):
            for m2 in M:
                for j in range(1, nj + 1):
                    if mi[i][m-1] == mi[j][m2-1] and i!=j:
                        mdl.add_constraint(x[(i, m, j, m2)] + x[(j, m2, i, m)] == 1)

    # Tardiness constraints
    for j in J:
        mdl.add_constraint(
            ff[j] - ss[j, nm] == tt[j, nm] + t2[(j,mi[j][-1])] - d[j],
            f"tardiness_{j}")

    # Sequencing constraints
    for m in M:
        for i in range(1, nj + 1):
            for m2 in M:
                for j in range(1, nj + 1):
                    if mi[i][m-1] == mi[j][m2-1] and i !=j:
                        mdl.add_constraint(
                            tt[j, m2] - tt[i, m] >= t2[(i,mi[i][m-1])] - bigM * (1 - x[(i, m, j, m2)]))

    # Dual variable constraints
    for j in J:
        for m in range(1, nm):
            mdl.add_constraint(
                tt[j, m+1] - tt[j, m] == t2[(j,mi[j][m-1])] + ss[j, m])
    
    # Solve the model
    mdl.parameters.timelimit = 300  # Set time limit (in seconds)
    mdl.parameters.mip.tolerances.mipgap = 0  # Set optimality gap to 0%
    start_time = time.time()
    solution = mdl.solve(log_output=False)
    solving_time = time.time() - start_time
    # if solution:
    #     print("Solution found:")
    #     print("Objective function value:", solution.get_objective_value())
    #     print("Start times:", {(i, m): tt[i, m].solution_value for m in M for i in J})
    #     print("Exceeded times:", {j: ff[j].solution_value for j in J})
    #     print("Dual variables:", {(i,m): ss[i, m].solution_value for m in M for i in J})
    #     print("X variables:", {(i, m, j, m2): x[(i, m, j, m2)].solution_value for m2 in M for j in J for m in M for i in J if x[(i, m, j, m2)].solution_value > 0.1})
    #     start_times = defaultdict(dict)
    #     end_times = defaultdict(dict)
    #     for j in range(1, nj + 1):
    #         for m in M:
    #             start_times[j][m] = (tt[j, m].solution_value)
    #             end_times[j][m] = (tt[j, m].solution_value + t2[(j,mi[j][m-1])])
    #     plot_gantt_chart(3, mi, start_times, end_times)
    # else:
    #     print("No solution found.")
    
    return solution.get_objective_value(), solving_time
    # if solution:
    #     weighted_tardiness = sum(w[j] * ff[j].solution_value for j in J)
    #     return weighted_tardiness, solving_time
    # else:
    #     return None, solving_time


def solvingJJSbyPyomo_Gurobi(variables_, seq=None, whichobj='wet', time_limit=300):
    variables = copy.deepcopy(variables_)

    # Parameters
    nj = variables["num_jobs"]
    nm = variables["num_machines"]
    tau = variables['num_time_slots']
    bigM = nj * nm * tau
    I = range(1, nj * nm + 1)
    M = range(1, nm + 1)
    J = range(1, nj + 1)
    e = variables["E"]
    es = variables["aggE"]
    w = variables["w"]
    d = variables["d"]
    mi = variables["sequence"]
    t2 = variables["p"]  # processing times
    mim = [item for sublist in mi.values() for item in sublist]

    model = ConcreteModel()

    # Sets
    model.J = Set(initialize=J)
    model.M = Set(initialize=M)
    JMJ2 = [(j1, m1, j2, m2) for j1 in J for m1 in M for j2 in J for m2 in M]
    model.JMJ2 = Set(initialize=JMJ2)
    model.JM = model.J * model.M
    # model.JMJ2 = model.J * model.M * model.J * model.M

    # Variables
    model.tt = Var(model.JM, domain=NonNegativeReals)  # Start times
    model.ff = Var(model.J, domain=NonNegativeReals)   # Exceeded times
    model.ss = Var(model.JM, domain=NonNegativeReals)  # Dual variables

    
    if seq:
        model.x = Param(model.JMJ2, initialize=seq, default=0, within=Binary)
        # Objective function
        def obj_expression(m):
            return sum(w[j] * m.ff[j] for j in m.J) + \
                   sum(es[(j, mi[j][mm - 1])] * m.ss[j, mm] for (j, mm) in m.JM)
        model.obj = Objective(rule=obj_expression, sense=minimize)
    else:
        if whichobj=='wet':
            model.x = Var(model.JMJ2, domain=Binary)
            # Objective function
            def obj_expression(m):
                return sum(w[j] * m.ff[j] for j in m.J) + \
                       sum(es[(j, mi[j][mm - 1])] * m.ss[j, mm] for (j, mm) in m.JM)
            model.obj = Objective(rule=obj_expression, sense=minimize)
        else:
            model.x = Var(model.JMJ2, domain=Binary)
            # Objective function
            def obj_expression(m):
                return sum(w[j] * m.ff[j] for j in m.J)

            model.obj = Objective(rule=obj_expression, sense=minimize)

    # Precedence constraints
    def precedence_rule(m, i, m1, j, m2):
        if i != j and mi[i][m1 - 1] == mi[j][m2 - 1]:
            return m.x[i, m1, j, m2] + m.x[j, m2, i, m1] == 1
        else:
            return Constraint.Skip
    if seq:
        pass
    else:
        model.precedence = Constraint(model.JMJ2, rule=precedence_rule)

    # Tardiness constraints
    def tardiness_rule(m, j):
        return m.ff[j] - m.ss[j, nm] == m.tt[j, nm] + t2[(j, mi[j][-1])] - d[j]
    model.tardiness = Constraint(model.J, rule=tardiness_rule)

    # Sequencing constraints
    def sequencing_rule(m, i, m1, j, m2):
        if i != j and mi[i][m1 - 1] == mi[j][m2 - 1]:
            return m.tt[j, m2] - m.tt[i, m1] >= t2[(i, mi[i][m1 - 1])] - bigM * (1 - m.x[i, m1, j, m2])
        else:
            return Constraint.Skip
    model.sequencing = Constraint(model.JMJ2, rule=sequencing_rule)

    # Dual variable constraints
    def dual_var_rule(m, j, m_):
        if m_ < nm:
            return m.tt[j, m_ + 1] - m.tt[j, m_] == t2[(j, mi[j][m_ - 1])] + m.ss[j, m_]
        else:
            return Constraint.Skip
    model.dual = Constraint(model.J, model.M, rule=dual_var_rule)

    # Solve
    solver = SolverFactory('cplex', solver_io='python')
    # solver.options['TimeLimit'] = time_limit
    solver.options['mipgap'] = 0

    start_time = time.time()
    result = solver.solve(model, tee=False)
    solving_time = time.time() - start_time
    print("solveddddddddddddddddddddddd,")
    if not seq:
        # Extract x as seq
        seq_result = {
            (i, m1, j, m2): int(round(value(model.x[i, m1, j, m2])))
            for (i, m1, j, m2) in JMJ2 if (i != j and mi[i][m1 - 1] == mi[j][m2 - 1])
        }
        obj_val, solving_time = solvingJJSbyPyomo_Gurobi(variables_, seq=seq_result, whichobj='wet', time_limit=300)
    else:
        obj_val = value(model.obj) if result.solver.termination_condition == TerminationCondition.optimal else None
    return obj_val, solving_time


def solvingJJSbyQuantumComputing(variables_, penalty = 1000, numer_of_shots = 1000, howtosolve = 'hybrid'):
    
    # Assign variables from the loaded dictionary
    loaded_variables = copy.deepcopy(variables_)
    num_jobs = loaded_variables["num_jobs"]
    num_machines = loaded_variables["num_machines"]+1
    num_time_slots = int(loaded_variables["num_time_slots"])
    
    J = range(1, num_jobs+1)
    M = range(1, num_machines+1)
    T = range(1, num_time_slots+1)
    
    p = loaded_variables["p"]
    # mp = loaded_variables["mp"]
    mp = {}
    d = loaded_variables["d"]
    w = loaded_variables["w"]
    E = loaded_variables["E"]
    f = loaded_variables["f"]

    mi = loaded_variables["sequence"]
    
    for j in J:
        p[(j,num_machines)] = 0
        E[(j,num_machines)] = 0
        mp[(j,num_machines)] = mi[j][-1]
        mp[(j,mi[j][0])] = 0
        for m in range(1, num_machines-1):
            mp[(j,mi[j][m])] = mi[j][m-1]

    # loaded_variables["num_machines"] = num_machines
    # loaded_variables["p"] = p
    # loaded_variables["mp"] = mp
    # loaded_variables["E"] = E

    shots = numer_of_shots
    h = [1, penalty, penalty, penalty, penalty, penalty]

    start_time = time.time()
    qubo = generate_qubo_with_completion_time_variable(num_jobs, num_machines, num_time_slots, J, M, T, w, E, p, d, mp, h)
    # print(qubo)
    if howtosolve == 'hybrid':
        response = solve_qubo_by_DWave_software_LeapHybrid(qubo)
    else:
        response = solve_qubo_by_DWave_software_QA(qubo, numer_of_shots)
    Qexetime = time.time() - start_time
    finish_times = {}
    penalty_ = 0
    for variable, value in response.first.sample.items():
        if value == 1 and variable == ():
            penalty_ = - qubo[(), ()]
        elif value == 1 and variable != ():
            finish_times[(variable[0], variable[1])] = variable[2]

    print(response.first.energy, qubo[(), ()], penalty_)
    print(finish_times)
    model, results = solve_LP_after_QA(finish_times, num_jobs, num_machines, num_time_slots, J, M, T, w, E, p, d, mp, f)
    if (results.solver.termination_condition == 'optimal') or (results.solver.termination_condition == 'feasible'):
        obj__ = model.obj()
    else:
        obj__ = 'inf'
    return response.first.energy + qubo[(), ()], obj__


    # print("Start times:")
    # for j in model.J:
    #     for m in model.M:
    #         print(f"Job {j}, Machine {m}: {model.start_time[j, m].value}")
    
    # print("\nDelays:")
    # for j in model.J:
    #     print(f"Job {j}: {model.delays[j].value}")
    
    # print("\nDelivery times:")
    # for j in model.J:
    #     print(f"Job {j}: {model.delivery_time[j].value}")
    
    # Print the objective function value
    # print("\nObjective function value:", model.obj.expr())
    
    # return response.first.energy + qubo[(), ()] + penalty_
    # return model.obj()
    # return loaded_variables

def generate_qubo_with_completion_time_variable(num_jobs, num_machines, num_time_slots, J, M, T, w, E, p, d, mp, h):
    P = {}
    Pp = {}
    for j in J:
        P[(j, num_machines)] = 0
        Pp[(j, num_machines)] = 0
        m = num_machines
        while (j, m) in mp:
                mpp = mp[(j, m)]
                if mpp != 0:
                    P[(j, mpp)] = P[(j, m)] + p[(j,mpp)]
                    Pp[(j, mpp)] = Pp[(j, m)] + p[(j,m)]
                m = mpp
    
    sumE = {i: sum(value for key, value in E.items() if key[0] == i) for i in set(key[0] for key in E)}
    
    sum_E_per_j = {j: sum(E[j, m] for m in range(1, num_machines + 1)) for j in range(1, num_jobs + 1)}


    ## Create Variables for the problem
    # Generate all combinations of variables
    variables = list(product(range(1, num_jobs + 1), range(1, num_machines + 1), range(1, num_time_slots + 1)))
    
    # Initialize an empty QUBO dictionary
    Q = {}
    
    # Populate QUBO dictionary with random coefficients
    for (j1, m1, t1) in variables:
        for (j2, m2, t2) in variables:
            Q[((j1,m1,t1), (j2,m2,t2))] = 0
    Q[(), ()] = 0

    ## Objective function: W E/T Cost
    # Define the objective function
    constant_value = 0
    for j in J:
        constant_value = - d[j]*w[j]*h[0]
        Q[(), ()] += constant_value
        for t in range(d[j], num_time_slots + 1):
            var_name = (j, num_machines, t)
            # Linear term (w[j] + sumE[j]) * t * x_jmt
            Q[var_name, var_name] += ((w[j] + sumE[j]) * t)*h[0]
            
        for tp in range(1, num_time_slots + 1):
            for m in range(1, num_machines):
                var_name = (j, m, tp)
                # Quadratic term -E[j,m] * (tp + Pp[j,m]) * x_jmt
                Q[var_name, var_name] += (-E[j, m] * (tp + Pp[j, m]))*h[0]


    ## Constraint 1: No − overlap
    for m in range(1, num_machines):
        for j in J:
            for t in T:
                for jp in J:
                    if j != jp:
                        for tp in range(t, t + p[jp, m] - 1 + 1):
                            if tp <= num_time_slots:
                                var1 = (j, m, t)
                                var2 = (jp, m, tp)
                                Q[(var1, var2)] += 1*h[1]


    ## Constraint 2: Precedence
    for j in J:
        for m in range(1, num_machines):
            if (j, m) in mp:
                for t in T:
                    for tp in range(1, t + p[j, m] - 1 + 1):
                        if tp <= num_time_slots:
                            if mp[j, m] != 0:
                                var1 = (j, mp[j, m], t)
                                var2 = (j, m, tp)
                                Q[(var1, var2)] += 1*h[2]
    
    for j in J:
        m = num_machines
        for t in range(d[j], num_time_slots+1):
            for tp in range(1, t- 1 + 1):
                if mp[j, m] != 0:
                    var1 = (j, mp[j, m], t)
                    var2 = (j, m, tp)
                    Q[(var1, var2)] += 1*h[2]

    
    ## Constraint 3: Operation once
    for j in J:
        for m in range(1, num_machines):
            if p[j, m] > 0:
                constant_value = + h[3]
                Q[(), ()] += constant_value
                # Add quadratic terms
                for t in range(p[j, m], num_time_slots+1):
                    var1 = (j, m, t)
                    Q[(var1, var1)] -= 1*h[3]  # y_{j,m,t}^2
                    for tp in range(p[j, m], num_time_slots+1):
                        if t != tp:
                            var2 = (j, m, tp)
                            Q[(var1, var2)] += 2*h[3]  # y_{j,m,t} y_{j,m,tp}
                for t in range(1, p[j, m]):
                    for tp in range(1, p[j, m]):
                        var1 = (j, m, t)
                        var2 = (j, m, tp)
                        Q[(var1, var2)] += 1*h[3]  # y_{j,m,t} y_{j,m,tp}
    

    ## Constraint 4: Tardy − once
    for j in J:
        constant_value = + h[4]
        Q[(), ()] += constant_value
        tmp = 0
        for t in range(d[j], num_time_slots + 1):
            var = (j, num_machines, t)
            tmp += 1
            Q[(var, var)] += 1*h[4]
            for tp in range(d[j], num_time_slots + 1):
                if tp != t:
                    Q[(var, (j, num_machines, tp))] += 2*h[4]
            Q[(var, var)] -= 2*h[4]

    
    qubo = dict(Q)
    return qubo


def generate_qubo_with_start_time_variable(num_jobs, num_machines, num_time_slots, J, M, T, w, E, p, d, mp, h):
    P = {}
    Pp = {}
    for j in J:
        P[(j, num_machines)] = 0
        Pp[(j, num_machines)] = 0
        m = num_machines
        while (j, m) in mp:
                mpp = mp[(j, m)]
                if mpp != 0:
                    P[(j, mpp)] = P[(j, m)] + p[(j,mpp)]
                    Pp[(j, mpp)] = Pp[(j, m)] + p[(j,m)]
                m = mpp
    sumE = {i: sum(value for key, value in E.items() if key[0] == i) for i in set(key[0] for key in E)}
    sum_E_per_j = {j: sum(E[j, m] for m in range(1, num_machines + 1)) for j in range(1, num_jobs + 1)}
    
    variables = list(product(J, M, T))
    Q = {}
    
    for (j1, m1, t1) in variables:
        for (j2, m2, t2) in variables:
            Q[((j1,m1,t1), (j2,m2,t2))] = 0
    Q[(), ()] = 0

    ## Objective function: W E/T Cost
    # Define the objective function
    constant_value = 0
    for j in J:
        constant_value = - d[j]*w[j]*h[0]
        Q[(), ()] += constant_value
        for t in range(d[j], num_time_slots+1):
            var_name = (j, num_machines, t)
            Q[var_name, var_name] += ((w[j] + sumE[j]) * t)*h[0]
    
        for m in range(1, num_machines):
            for tp in T:
                var_name = (j, m, tp)
                Q[var_name, var_name] += (-E[j, m] * (tp + P[j, m]))*h[0]

    ## Constraint 1: No − overlap
    for m in range(1, num_machines):
        for j in J:
            for t in T:
                for jp in J:
                    if j != jp:
                        for tp in range(t, min(t + p[j, m], num_time_slots+1)):
                            var1 = (j, m, t)
                            var2 = (jp, m, tp)
                            Q[(var1, var2)] += 1*h[1]


    ## Constraint 2: Precedence
    for j in J:
        # for m in range(1, num_machines):
        for m in M:
            if (j, m) in mp:
                if mp[j, m] != 0:
                    for t in T:
                        for tp in range(0, min(t + p[j, mp[j, m]], num_time_slots+1)):
                            var1 = (j, mp[j, m], t)
                            var2 = (j, m, tp)
                            Q[(var1, var2)] += 1*h[2]


    ## Constraint 3: Operation once
    for j in J:
        for m in range(1, num_machines):
            if p[j, m] > 0:
                constant_value = + h[3]
                Q[(), ()] += constant_value
                # Add quadratic terms
                for t in T:
                    var1 = (j, m, t)
                    Q[(var1, var1)] -= 1*h[3]  # y_{j,m,t}^2
                    for tp in T:
                        if t!= tp:
                            var2 = (j, m, tp)
                            Q[(var1, var2)] += 2*h[3]  # y_{j,m,t} y_{j,m,tp}
    

    ## Constraint 4: Tardy − once
    for j in J:
        constant_value = + h[4]
        Q[(), ()] += constant_value
        tmp = 0
        for t in range(d[j], num_time_slots+1):
            var = (j, num_machines, t)
            tmp += 1
            Q[(var, var)] += 1*h[4]
            for tp in range(d[j], num_time_slots+1):
                if tp != t:
                    Q[(var, (j, num_machines, tp))] += 2*h[4]
            Q[(var, var)] -= 2*h[4]


    qubo = dict(Q)
    return qubo


def solve_qubo_by_DWave_software(Q, solver, num_reads):
    if solver == 'SA':
        status = solve_qubo_by_DWave_software_SA(Q, num_reads)
    elif solver == 'QA':
        status = solve_qubo_by_DWave_software_QA(Q, num_reads)
    else:
        status = solve_qubo_by_DWave_software_LeapHybrid(Q)
    return status


def solve_qubo_by_DWave_software_SA(Q, num_reads):
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = dimod.SimulatedAnnealingSampler()
    # Create a linear beta schedule with the desired number of sweeps
    beta_range = [0.1, 10.0]

    response = sampler.sample(bqm, num_reads=num_reads, beta_range=beta_range)
    return response

def solve_qubo_by_DWave_software_QA(qubo, num_reads):
    
    sampler = EmbeddingComposite(DWaveSampler(solver={'name': 'Advantage_system6.4'}))
    response = sampler.sample_qubo(qubo, num_reads=num_reads)
    return response


def solve_qubo_by_DWave_software_LeapHybrid(qubo):

    sampler = LeapHybridSampler(solver={'category': 'hybrid'})
    response = sampler.sample_qubo(qubo)
    return response


def objective_function(model):
    return sum(model.w[j] * model.delays[j] for j in model.J) + sum(model.es[j,m] * (model.delivery_time[j] 
                                                             - sum(model.t[j,mp] for mp in model.M if model.finish_times[(j,mp)] >= model.finish_times[(j,m)])
                                                             - model.start_time[j,m]) for m in model.M for j in model.J)

def eq2_rule(model, j):
    return model.delays[j] >= model.start_time[j,model.f[j]] + model.t[j,model.f[j]] - model.d[j]

def eq3_rule(model, i, j, m, mp):
    if i == j and m !=mp:
        if model.finish_times[(i,m)] <= model.finish_times[(j,mp)]:
            return model.start_time[i,m] + model.t[i,m] <= model.start_time[j,mp]
    if m == mp and i!=j:
        if model.finish_times[(i,m)] <= model.finish_times[(j,mp)]:
            return model.start_time[i,m] + model.t[i,m] <= model.start_time[j,mp]
    return Constraint.Skip

def eq4_rule(model, j):
    return model.delivery_time[j] >= model.d[j]

def eq5_rule(model, j):
    return model.delivery_time[j] >= model.start_time[j,model.f[j]] + model.t[j,model.f[j]]


def solve_LP_after_QA(finish_times, num_jobs, num_machines, num_time_slots, J, M, T, w, E, p, d, mp, f):
    finish_times = {k: v for k, v in finish_times.items() if k[1] != num_machines}
    p = {k: v for k, v in p.items() if k[1] != num_machines}
    E = {k: v for k, v in E.items() if k[1] != num_machines} 
    
    model = ConcreteModel()
    model.J = Set(initialize=range(1, num_jobs+1))  # Set of jobs
    model.M = Set(initialize=range(1, num_machines))  # Set of machines
    
    model.w = Param(model.J, initialize=w)
    model.t = Param(model.J, model.M, initialize=p)
    model.d = Param(model.J, initialize=d)
    model.es = Param(model.J, model.M, initialize=E)
    model.f = Param(model.J, initialize=f)
    model.finish_times = Param(model.J, model.M, initialize=finish_times, default=0)
    
    model.start_time = Var(model.J, model.M, within=NonNegativeReals)
    model.delays = Var(model.J, within=NonNegativeReals)
    model.delivery_time = Var(model.J, within=NonNegativeReals)
    
    model.obj = Objective(rule=objective_function, sense=minimize)
    model.eq2 = Constraint(model.J, rule=eq2_rule)
    model.eq3 = Constraint(model.J, model.J, model.M, model.M, rule=eq3_rule)
    model.eq4 = Constraint(model.J, rule=eq4_rule)
    model.eq5 = Constraint(model.J, rule=eq5_rule)
    
    # model.pprint()
    
    solver = SolverFactory('cplex', solver_io='python')
    # solver = SolverFactory('glpk')
    results = solver.solve(model, tee=False)

    return model, results


def calculate_hypothesis_tests(data_for_hypo, alpha = 0.05):
    # Step 2: Identify the first two minimum values of the makespan before LP colum
    df_sorted = data_for_hypo.sort_values(by=['makespan before LP', 'WET cost after LP'])
    # print(df_sorted)
    df = df_sorted # ['makespan before LP'].nsmallest(2)
    # print(df)
    grouped = df.groupby('makespan before LP')
    is_not_done = True
    i = [0]
    ii = 1
    while is_not_done:
        print("hereeee")
        group_names = list(grouped.groups.keys())
        group_lowest = []
        for index in i:
            group = grouped.get_group(group_names[index])['WET cost after LP'].tolist()
            group_lowest.extend(group)
        if ii < len(group_names):
            group_next_to = grouped.get_group(group_names[ii])['WET cost after LP'].tolist()
        else:
            is_not_done = False
            return False, group_names[ii]
    
        # Calculate for group_next_to
        n_next_to = len(group_next_to)
        mean_next_to = np.mean(group_next_to)
        std_dev_next_to = np.std(group_next_to, ddof=1)  # Sample standard deviation
        
        # Calculate for group_lowest
        n_lowest = len(group_lowest)
        mean_lowest = np.mean(group_lowest)
        std_dev_lowest = np.std(group_lowest, ddof=1)  # Sample standard deviation
        
        s_p = np.sqrt(((n_next_to - 1) * std_dev_next_to**2 + (n_lowest - 1) * std_dev_lowest**2) / (n_next_to + n_lowest - 2))
        
        t_statistic = (mean_next_to - mean_lowest) / (s_p * np.sqrt(1 / n_next_to + 1 / n_lowest))
        
        
        df = n_next_to + n_lowest - 2
        # One-sided critical t-value
        t_critical_one_sided = stats.t.ppf(1 - alpha, df)
    
        if t_statistic >= t_critical_one_sided:
            # print("Reject the null hypothesis (H_0): Mean_next_to > Mean_lowest.")
            is_not_done = False
            print("okay")
            return True, group_names[i[-1]]
        else:
            i.append(ii)
            ii = ii + 1


def check_overlaps(paths, new_path, machine_sequences, durations, num_jobs, ub, v_penalty, dictionary_overlaps, dictionary_costs, machine_index):    
    #weights = {(path1, path2): weight}
    weights = {}
    weights[(new_path[1], new_path[1])] = new_path[3][-1]
    dictionary_costs[new_path[1]] = {}
    dictionary_costs[new_path[1]][new_path[0]] = new_path[3][-1]
    
    path2 = new_path[2]
    job2 = new_path[0]
    for path01 in [p for p in paths if p[0] != new_path[0]]:
        
        path1 = path01[2]
        job1 = path01[0]
        # Check overlaps across all machines
        is_overlapping = False
        for machine_index1 in range(len(path1)):
            
            machinee = machine_sequences[job1][machine_index1]
            start_time1 = path1[machine_index1]
            end_time1 = start_time1 + durations[(job1, machinee)]
            
            start_time2 = path2[machine_index[(job2, machinee)]]
            end_time2 = start_time2 + durations[(job2, machinee)]
            # Check if the time intervals overlap
            if (start_time1 < end_time2) and (start_time2 < end_time1):
                is_overlapping = True
                break  # No need to check further if there's an overlap
            
        if is_overlapping:
            weights[(path01[1],new_path[1])] = v_penalty
            dictionary_overlaps[path01[1]].add(new_path[1])
            dictionary_overlaps[new_path[1]].add(path01[1])
        else:
            # Check if path01[1] exists in dictionary_costs and update its nested dictionary
            dictionary_costs[path01[1]].setdefault(new_path[0], new_path[3][-1])
            dictionary_costs[new_path[1]].setdefault(path01[0], path01[3][-1])
            dictionary_costs[path01[1]][new_path[0]] = min(
                dictionary_costs[path01[1]][new_path[0]], new_path[3][-1]
            )
            dictionary_costs[new_path[1]][path01[0]] = min(
                dictionary_costs[new_path[1]][path01[0]], path01[3][-1]
            )
    return weights, dictionary_overlaps, dictionary_costs


def check_feasibility(solutions, w):
    solstatus = np.ones(len(solutions))
    for i in range(len(solutions)):
        solution = solutions[i]
        for i1 in range(len(solution)-1):
            for i2 in range(i1+1, len(solution)):
                if (solution[i1], solution[i2]) in w or (solution[i2], solution[i1]) in w:
                    solstatus[i] = 0
                    break
            if solstatus[i] == 0:
                break
        if solstatus[i] == 1:
            return i
    return 0

def outliers_solutions(data, alpha):

    # Calculate the z-score threshold for the right tail
    z_threshold = stats.norm.ppf(1 - alpha)  
    
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Identify candidate solutions
    candidate_indices = [i for i, x in enumerate(data) if (x - mean) / std_dev >= z_threshold]
    return candidate_indices

def solve_binary_optimization_v2(n_jobs, paths, n_paths, lb, ub, time_limit, valid_indices):
    my_start_time = time.time()
    try:
        start_time = time.time()
        # Create a model
        model = ConcreteModel()
        # Define binary variables
        model.x = Var(valid_indices, within=Binary)
        expression_obj = sum(paths[i]["total_costs"] * model.x[i] for i in valid_indices)
        model.obj = Objective(expr= expression_obj, sense=minimize)
        # Define constraints
        model.constraints = ConstraintList()  # Create a ConstraintList to store multiple constraints
        for j in range(1, n_jobs+1):
            # Add constraint that at least one path must be selected for each job
            model.constraints.add(sum(model.x[path_key] for path_key, value in paths.items() if value['job'] == j) == 1)
        # for (i, j) in w:
        #     if i!=j:
        #         model.constraints.add(model.x[i] + model.x[j] <= 1)
        for p in valid_indices:
            filtered = {i for i in paths[p]["overlaps"] if i in valid_indices}
            model.constraints.add((model.x[p]-1)*len(filtered) 
                          + sum(model.x[j] for j in filtered) <= 0)
    
        model.constraints.add(expression_obj <= ub+0.1)
        problem_generation_time = time.time() - start_time
        
        # Solve the model
        solver = SolverFactory('cplex', solver_io='python')  # You can use other solvers like 'cbc', 'ipopt', etc.
        solver.options['timelimit'] = time_limit  # Set time limit (in seconds)
        solver.options['mipgap'] = 0               # Set MIP gap to zero
        start_time = time.time()
        result = solver.solve(model)
        solving_time = time.time() - start_time
    
        # Check if the solver found a solution
        if result.solver.termination_condition == TerminationCondition.optimal:
            return model.obj(), [i for i in valid_indices if model.x[i].value > 0], solving_time, problem_generation_time
        else:
            return ub*10+10, [], solving_time, problem_generation_time
    except:
        return ub*10+10, [], 0, time.time() - my_start_time
    return model.obj(), [i for i in valid_indices if model.x[i].value > 0], solving_time, problem_generation_time


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

    elif howtosolve == 'quantum_QA':
        sampler = EmbeddingComposite(DWaveSampler(solver={'name': 'Advantage_system6.4'}))
        response = sampler.sample_qubo(Q, num_reads=numer_of_shots) #num_reads=100
        qpu_access_time = response.info['qpu_access_time'] / 1e6  # Convert to seconds
        run_time = response.info['run_time'] / 1e6  # Convert to seconds

    else:
        start_time = time.time()
        sampler = LeapHybridSampler(solver={'category': 'hybrid'})
        response = sampler.sample_qubo(Q)
        print('timeeeee', time.time() - start_time)
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

    # unique_obj_values = list(solution_counts.keys())
    # first_solutions = [solutions[0] for solutions in solution_counts.values()]
    # obj_value_counts = [len(solutions) for solutions in solution_counts.values()]
    # print('number of unique solutions: ', len(unique_obj_values), 'base: ', len(response))
    # if len(unique_obj_values) > 10:
    #     candidate_indices = outliers_solutions(obj_value_counts, alpha)
    #     unique_obj_values = np.array(unique_obj_values)[candidate_indices].tolist()
    #     first_solutions = np.array(first_solutions, dtype=object)[candidate_indices].tolist()
        
    # sorted_pairs = sorted(zip(unique_obj_values, first_solutions), key=lambda x: x[0])
    # objective_value_sorted, solutions_sorted = zip(*sorted_pairs)
    
    # objective_value_sorted = list(objective_value_sorted)
    # solutions_sorted = list(solutions_sorted)
    # index = check_feasibility(solutions_sorted, w)
    
    # print(objective_value_sorted[index], solutions_sorted[index])
    # print("objjjjjjj", np.round(response.first.energy + n_jobs*h[1],5), h[1], response.first)
    return np.round(response.first.energy + n_jobs*h[1],2), [], qpu_access_time, run_time

def generate_all_paths_for_jobs_v3(instance_, ub, lb, lamda, epsilon, v_penalty, time_limit, numer_of_shots=1000, howtosolve='digital'):
    instance = copy.deepcopy(instance_)
    loginfo = {}
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    tau = instance['num_time_slots']  # available time priods
    durations = instance['p']  # Durations for each (job, machine) combination
    
    aggE = instance['aggE']
    
    duedate = instance['d']
    tardinesscosts = instance['w']

    sequences = instance['sequence']

    machine_index = {
        (job, machine): sequences[job].index(machine)
        for job in sequences
        for machine in sequences[job]
    }

    paths = []  # Store start times for all jobs (job, path_num, path, costs)
    
    incompleted_paths = []

    weights = {}

    total_solving_time = 0

    dictionary_overlaps = {key: set() for key in range(0, 10000)}
    dictionary_costs = {}
    
    def backtrack(job, path, current_time, costs, lb, ub, dictionary_overlaps, dictionary_costs):
        # Base case: if all machines for the job have been scheduled, save the path
        if len(path) == num_machines:
            if costs[-1] > lb*1 or costs[-2] > lb or costs[-3] > lb:
                new_path = (job, incompleted_paths[-1][1]+1 if incompleted_paths else 1, path, costs)
                incompleted_paths.append(new_path)
            else:
                new_path = (job, paths[-1][1]+1 if paths else 1, path[:], costs + [sum(costs)])
                tmpwei, dictionary_overlaps, dictionary_costs = check_overlaps(paths, new_path, sequences, durations, num_jobs, ub, v_penalty, dictionary_overlaps, dictionary_costs, machine_index)
                paths.append(new_path)
                weights.update(tmpwei)
            return
        
        # Get the next machine from the machine sequence
        next_machine = sequences[job][len(path)]
        # Determine the earliest start time based on the previous machine's end time
        last_machine = sequences[job][len(path) - 1] if path else None
        earliest_start_time = path[-1] + durations.get((job, last_machine), 0) if path else 0
        latest_start_time = tau - sum(durations.get((job, sequences[job][i]), 0) for i in range(len(path), num_machines)) + 1
        
        # Loop through all possible start times for the next machine
        for next_start_time in range(earliest_start_time, latest_start_time):
            tmp_path = path + [next_start_time]
            tmp_current_time = next_start_time + durations.get((job, next_machine), 0)
            tmp_costs = costs
            if len(tmp_path) <= 1:
                backtrack(job, tmp_path, tmp_current_time, tmp_costs, lb, ub, dictionary_overlaps, dictionary_costs)
            elif len(tmp_path) < num_machines:
                tmp_costs = tmp_costs +[aggE[(job, sequences[job][len(tmp_path)-2])] * (tmp_path[-1] - (tmp_path[-2] + durations[(job, sequences[job][len(tmp_path)-2])]))]
                if sum(tmp_costs) <= ub:
                    if tmp_costs[-1] <= lb:
                        backtrack(job, tmp_path, tmp_current_time, tmp_costs, lb, ub, dictionary_overlaps, dictionary_costs)
                    else:
                        incompleted_paths.append((job, incompleted_paths[-1][1]+1 if incompleted_paths else 1, tmp_path[:], tmp_costs))                  
            elif len(tmp_path) == num_machines:
                tmp_costs = tmp_costs +[aggE[(job, sequences[job][len(tmp_path)-2])] * (tmp_path[-1] - (tmp_path[-2] + durations[(job, sequences[job][len(tmp_path)-2])]))]
                tmp_costs = tmp_costs +[aggE[(job, sequences[job][-1])] * max( duedate[job] - (tmp_path[-1] + durations[(job,  sequences[job][-1])]), 0)]
                tmp_costs = tmp_costs +[max(tmp_path[-1] + durations[(job, sequences[job][-1])] - duedate[job], 0) * tardinesscosts[job]]
                if sum(tmp_costs) <= ub:
                    if tmp_costs[-1] <= lb*1 and tmp_costs[-2] <=lb and tmp_costs[-3] <= lb:
                        backtrack(job, tmp_path, tmp_current_time, tmp_costs, lb, ub, dictionary_overlaps, dictionary_costs)
                    else:
                        incompleted_paths.append((job, incompleted_paths[-1][1]+1 if incompleted_paths else 1, tmp_path[:], tmp_costs))
            else:
                pass
    
    old_lb = lb
    for iter in range(100):
        # Generate start times for each job
        start_time = time.time()
        if not incompleted_paths:
            for job in range(1, num_jobs + 1):
                backtrack(job, [], 0, [], lb, ub, dictionary_overlaps, dictionary_costs)
        else:
            tmp_inc = incompleted_paths
            incompleted_paths = []
            for i in range(len(tmp_inc)):
                inc = tmp_inc[i]
                backtrack(inc[0], inc[2], inc[2][-1]+durations.get((inc[0], sequences[inc[0]][len(inc[2])-1]), 0), inc[3], lb, ub, dictionary_overlaps, dictionary_costs)
        time1 = time.time() - start_time
        print(f"Total execution time to generate {len(paths)} paths and check {len(weights)} overlaps: {time1:.6f} seconds")
        aapath = len(paths)
        aaweights = len(weights)
        # prepare for Mahdi's and Tom's ideas
        df = pd.DataFrame(paths, columns=["job", "path", "times", "costs"])
        df['totalcost'] = df['costs'].apply(lambda x: x[-1])  # Extract last element in 'costs' array
        # new idea of Mahdi to prune paths
        start_time = time.time()
        path_to_prune = []
        path_cost_sums = df['path'].map(lambda path: sum(dictionary_costs[path].values()))
        path_to_prune = path_cost_sums > (ub + 0.01)
        df = df[~path_to_prune]
        print(len(df))
        time_to_remove_paths2 = time.time() - start_time
        # new idea of Tom to prune paths
        start_time = time.time()
        # Iterate through jobs
        for j in range(1, num_jobs + 1):
            # Filter rows for job j
            df_job = df[df['job'] == j]
            # Convert sets in dictionary_overlaps to a hashable type (e.g., tuple)
            grouped = df_job.groupby(df_job['path'].map(lambda path: tuple(dictionary_overlaps[path])))
            # Identify paths to prune
            path_to_prune = []
            for _, group in grouped:
                if len(group) > 1:
                    # Sort by total cost to identify the least cost paths
                    sorted_group = group.sort_values(by='totalcost', ascending=True)
                    # Retain the path with the least cost, prune others
                    path_to_prune.extend(sorted_group.iloc[1:]['path'])
            # Prune paths for the current job
            df = df[~df['path'].isin(path_to_prune)]
        
        # Iterate through jobs
        for j in range(1, num_jobs + 1):
            path_to_prune = []
            df_job = df[df['job'] == j]  # Filter rows for job j
            for i, l in itertools.combinations(range(len(df_job)), 2):
                    p1 = df_job.iloc[i]  # Use .iloc to access rows by index
                    p2 = df_job.iloc[l]  # Access the k-th row of df_job
                    if (p1['totalcost'] >= p2['totalcost'] and 
                        dictionary_overlaps[p2['path']].issubset(dictionary_overlaps[p1['path']])):
                        path_to_prune.append(p1['path'])
                    elif (p2['totalcost'] >= p1['totalcost'] and 
                          dictionary_overlaps[p1['path']].issubset(dictionary_overlaps[p2['path']])):
                        path_to_prune.append(p2['path'])
            df = df[~df['path'].isin(path_to_prune)]
            # print("len", len(df))
        print(len(df))
        time_to_remove_paths = time.time() - start_time

        start_time = time.time()
        old_len_df = len(df)
        new_len_df = 0
        while old_len_df != new_len_df:
            old_len_df = len(df)
            # Iterate through jobs
            for j in range(1, num_jobs + 1):
                path_to_prune = []
                df_job = df[df['job'] == j]  # Filter rows for job j
                for _, p in df_job.iterrows():  # Iterate through each row for job j
                    # Filter out paths not in p['weights']
                    df_paths = df[~df['path'].isin(dictionary_overlaps[p['path']])]
                    new_ub = ub - p['totalcost'] +0.01 # Update upper bound
                    # Check feasibility for other jobs
                    for j2 in range(1, num_jobs + 1):
                        if j != j2:
                            df_path_j2 = df_paths[df_paths['job'] == j2]  # Filter rows for job j2
                            # Sort from min to max of total costs
                            df_path_j2 = df_path_j2.sort_values('totalcost')
                            if df_path_j2.empty or new_ub - df_path_j2.iloc[0]['totalcost'] < 0:
                                path_to_prune.append(p['path'])  # Mark path for pruning
                                break
                            else:
                                new_ub -= df_path_j2.iloc[0]['totalcost']  # Reduce upper bound
                # print("Paths to prune:", len(path_to_prune))
                df = df[~df['path'].isin(path_to_prune)]
            new_len_df = len(df)
        time_to_remove_paths2 = time_to_remove_paths2 + time.time() - start_time
        
        # update the variables paths and weights
        valid_indices = set(df["path"])
        weights = {k: v for k, v in weights.items() if k[0] in valid_indices and k[1] in valid_indices}
        paths = [
        (row.job, row.path, row.times, row.costs) for row in df.itertuples(index=False)
        ]
        print(f"Total execution time to generate {len(paths)} paths and check {len(weights)} overlaps: {time1:.6f} seconds")

        # Iterate through jobs
        df = pd.DataFrame(paths, columns=["job", "path", "times", "costs"])
        df['totalcost'] = df['costs'].apply(lambda x: x[-1])  # Extract last element in 'costs' array

        start_time = time.time()
        objective_value = ub*3
        jobs_in_paths = {entry[0] for entry in paths}  # Extract unique jobs from paths
        missing_jobs = [job for job in range(1, num_jobs+1) if job not in jobs_in_paths]

        problem_generation_time = 0
        if len(missing_jobs) < 1:
            if howtosolve == 'digital':
                #objective_value, variables, solving_time, problem_generation_time = solve_binary_optimization_v2(num_jobs, paths, len(paths), weights, old_lb, ub, time_limit, valid_indices, dictionary_overlaps)
                objective_value = 0
                variables = []
                solving_time = 0
                problem_generation_time = 0
                total_solving_time = total_solving_time+solving_time
            else:
                objective_value, variables = generate_qubo_job_path_optimization(num_jobs, paths, weights, old_lb, ub, [1, v_penalty], numer_of_shots=numer_of_shots, howtosolve=howtosolve)
        else:
            objective_value = ub*3
        
        time2 = time.time() - start_time        

        print("Objective value is:", objective_value, "lower bound is:", lb, 'upper bound is', ub)
        loginfo[iter] = [iter, aapath, aaweights, len(paths), len(weights), time1, time2, problem_generation_time, time_to_remove_paths, time_to_remove_paths2, objective_value]
        
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
    return paths, weights, objective_value, variables, loginfo, total_solving_time


def generate_all_paths_for_jobs_v4(instance, ub, lb, lamda, epsilon, v_penalty, time_limit, numer_of_shots=1000, howtosolve='digital', max_cost_of_optimal = 0, index_name=0, instance_name=0, num_times_name=0):
    
    loginfo = {}
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    tau = instance['num_time_slots']  # available time priods
    old_lb = 0
    for iter in range(1):
        
        ctype_instance = convertInstanceToCtype(instance)
        if iter > 1:
            ctype_results = c_library.my_main(ctype_instance, ctypes.byref(ctype_results), lb, ub, max_cost_of_optimal)
        else:
            ctype_results = c_library.my_main(ctype_instance, None, lb, ub, max_cost_of_optimal)
        
        results = ResultsToDict(ctype_results, num_jobs, num_machines)
        valid_indices = set(results["paths"].keys())
        paths = results["paths"]
        n_overlaps_after = 0;
        for p in valid_indices:
            n_overlaps_after += len({i for i in paths[p]["overlaps"] if i in valid_indices})
            paths[p]["total_costs"] = sum(paths[p]["costs"])
        n_overlaps_after = int(n_overlaps_after/2)
        print("n_overlaps_after", n_overlaps_after)
        file_name = (
            "index_" + str(index_name) +
            "_instance_" + str(instance_name) +
            "_job_" + str(num_jobs) +
            "_time_" + str(num_times_name) +
            "_con_" + str(1.1) +
            "_ub_" + str(ub) + ".pkl"
        )
        with open(file_name, 'wb') as f:
            pickle.dump(results, f)
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
    return paths, [], objective_value, variables, loginfo, total_solving_time, results['number_of_path_less_than_max_cost']

    