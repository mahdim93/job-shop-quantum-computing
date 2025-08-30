#ifndef PRUNING_H
#define PRUNING_H
#include <stdbool.h>
#include "read_file.h"

// Function to calculate path cost sum
float calculate_path_cost_sum(float *min_costs_per_jobs, int num_jobs);

// Function to remove paths based on cost sum > (ub + 0.01)
void remove_paths(Paths *paths, int *num_paths, int num_jobs, float ub);

void prune_paths(Paths *paths, int *num_paths, int num_jobs);

void remove_paths_section_2(Paths *paths, int *num_paths, int num_jobs, float ub);

void group_paths_by_jobs(Paths *all_paths, int num_paths, JobPaths **job_paths_array, int num_jobs);

bool is_subset(int *set1, int size1, int *set2, int size2);

static inline int path_in_list(int *list, int list_size, int value);

int compare_paths(const void *a, const void *b);

#endif

