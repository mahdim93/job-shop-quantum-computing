import ctypes
import sys
import subprocess
import io
import numpy as np
import json

# Helper function to convert a 2D list to a C-compatible 2D array
def create_2d_array(data, dtype):
    arr = (ctypes.POINTER(dtype) * len(data))()
    for i, row in enumerate(data):
        arr[i] = (dtype * len(row))(*row)
    return arr

# Define the C struct in Python
class PathData(ctypes.Structure):
    _fields_ = [
        ("job", ctypes.c_int),
        ("path", ctypes.c_int),
        ("start_times", ctypes.POINTER(ctypes.c_int)),
        ("costs", ctypes.POINTER(ctypes.c_float)),
        ("total_costs", ctypes.c_float),
        ("min_costs_per_jobs", ctypes.POINTER(ctypes.c_float)),
        ("number_of_overlaps", ctypes.c_int),
        ("overlaps", ctypes.POINTER(ctypes.c_int)),
        ("idx_machine", ctypes.c_int)
    ]

class Results(ctypes.Structure):
    _fields_ = [
        ("paths", ctypes.POINTER(PathData)), 
        ("incomplete_paths", ctypes.POINTER(PathData)),
        ("num_paths", ctypes.c_int),
        ("num_paths_before", ctypes.c_int),
        ("num_overlaps", ctypes.c_int),
        ("num_overlaps_before", ctypes.c_int),
        ("num_incomplete_paths", ctypes.c_int),
        ("time_to_find_paths", ctypes.c_double),
        ("time_to_find_overlaps", ctypes.c_double),
        ("time_to_remove_paths_mahdi", ctypes.c_double),
        ("time_to_remove_paths_tom", ctypes.c_double),
        ("number_of_path_less_than_max_cost", ctypes.c_int),
    ]

# Convert dictionary data into C-compatible arrays
def convertInstanceToCtype(instance):
    # Convert tuple keys to string representation
    def convert_keys(d):
        new_dict = {}
        for key, value in d.items():
            # If the key is a tuple, convert it to a string
            if isinstance(key, tuple):
                key = f"({key[0]},{key[1]})"
            # Recursively convert nested dictionaries
            if isinstance(value, dict):
                value = convert_keys(value)
            if isinstance(value, float):
                value = np.round(value,3)
            new_dict[key] = value
        return new_dict
    
    # Apply the conversion to the instance
    instance = convert_keys(instance)

    # Now you can safely convert it to a JSON string
    json_string = json.dumps(instance)
    json_bytes = json_string.encode('utf-8')
    return ctypes.c_char_p(json_bytes)

def ResultsToDict(results, num_jobs, num_machines):
    def paths_array_to_list(paths_ptr, num_paths):
        """Convert a pointer to an array of Paths into a list of dictionaries."""
        if not paths_ptr:
            return {}
        return {paths_ptr[i].path: path_to_dict(paths_ptr[i]) for i in range(num_paths)}

    def path_to_dict(path):
        """Convert a single Paths struct to a dictionary."""
        return {
            "job": path.job+1,
            "path": path.path,
            "start_times": [path.start_times[i] for i in range(num_machines)] if path.start_times else [],
            "costs": [path.costs[i] for i in range(num_machines + 1)] if path.costs else [],
            "total_costs": path.total_costs,
            "min_costs_per_jobs": [path.min_costs_per_jobs[i] for i in range(num_jobs)] if path.min_costs_per_jobs else [],
            "number_of_overlaps": path.number_of_overlaps,
            "overlaps": [path.overlaps[i] for i in range(path.number_of_overlaps)] if path.overlaps else [],
            "idx_machine": path.idx_machine,
        }

    return {
        "paths": paths_array_to_list(results.paths, results.num_paths),
        "incomplete_paths": paths_array_to_list(results.incomplete_paths, results.num_incomplete_paths),
        "num_paths": results.num_paths,
        "num_paths_before": results.num_paths_before,
        "num_overlaps": results.num_overlaps,
        "num_overlaps_before": results.num_overlaps_before,
        "num_incomplete_paths": results.num_incomplete_paths,
        "time_to_find_paths": results.time_to_find_paths,
        "time_to_find_overlaps": results.time_to_find_overlaps,
        "time_to_remove_paths_mahdi": results.time_to_remove_paths_mahdi,
        "time_to_remove_paths_tom": results.time_to_remove_paths_tom,
        "number_of_path_less_than_max_cost": results.number_of_path_less_than_max_cost,
    }

