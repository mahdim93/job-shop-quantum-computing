// pruning.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include "pruning.h"

void group_paths_by_jobs(Paths *all_paths, int num_paths, JobPaths **job_paths_array, int num_jobs) {
    // Allocate memory for the array of JobPaths
    *job_paths_array = (JobPaths *)malloc(num_jobs * sizeof(JobPaths));
    if (*job_paths_array == NULL) {
        fprintf(stderr, "Memory allocation for JobPaths array failed\n");
        exit(EXIT_FAILURE);
    }
	
    // Initialize each JobPaths element
    for (int i = 0; i < num_jobs; i++) {
        (*job_paths_array)[i].num_paths = 0;
        (*job_paths_array)[i].path_indices = NULL;
    }

    // Count the number of paths for each job
    for (int i = 0; i < num_paths; i++) {
        int job_id = all_paths[i].job; // Adjust job index (assuming job IDs start from 1)
        (*job_paths_array)[job_id].num_paths++;
    }

    // Allocate memory for paths in each job
    for (int i = 0; i < num_jobs; i++) {
        if ((*job_paths_array)[i].num_paths > 0) {
            (*job_paths_array)[i].path_indices = (int *)malloc((*job_paths_array)[i].num_paths * sizeof(int));
        }
    }

    // Distribute paths to their corresponding jobs
    int *current_path_index = (int *)calloc(num_jobs, sizeof(int));
    for (int i = 0; i < num_paths; i++) {
        int job_id = all_paths[i].job;
        int index = current_path_index[job_id];
		(*job_paths_array)[job_id].path_indices[index] = i; // Store the original index of the path
        current_path_index[job_id]++;
    }

    // Clean up
    free(current_path_index);
}

// Function to calculate path cost sum
float calculate_path_cost_sum(float *min_costs_per_jobs, int num_jobs) {
    float sum = 0.0;
    for (int j = 0; j < num_jobs; ++j) {
        sum += min_costs_per_jobs[j];
    }
    return sum;
}

// Function to remove paths based on cost sum > (ub + 0.01)
void remove_paths(Paths *paths, int *num_paths, int num_jobs, float ub) {
    int *path_to_prune = (int *)malloc(*num_paths * sizeof(int));

    // Calculate path cost sums and prune paths
    for (int i = 0; i < *num_paths; i++) {
        float path_cost_sum = calculate_path_cost_sum(paths[i].min_costs_per_jobs, num_jobs);
        path_to_prune[i] = (path_cost_sum > (ub + 0.01)) ? 1 : 0;
    }

    // Create a new Paths with pruned paths
    int new_num_paths = 0;
    for (int i = 0; i < *num_paths; i++) {
        if (path_to_prune[i] == 0) {
            paths[new_num_paths] = paths[i];
            new_num_paths++;
        }
    }

    *num_paths = new_num_paths;
    free(path_to_prune);
}
/*
bool is_subset(int *set1, int size1, int *set2, int size2) {
    for (int i = 0; i < size1; i++) {
        bool found = false;
        for (int j = 0; j < size2; j++) {
            if (set1[i] == set2[j] ) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false; // An element in set1 is not in set2
        }
    }
    return true; // All elements of set1 are in set2
}
*/

bool is_subset(int *smaller, int smaller_size, int *larger, int larger_size) {
    int i = 0, j = 0;
    
    while (i < smaller_size && j < larger_size) {
        if (smaller[i] < larger[j]) {
            return false; // Element in smaller not found in larger
        } else if (smaller[i] == larger[j]) {
            i++; // Match found, move to the next element in smaller
        }
        j++;
    }
    return i == smaller_size; // All elements in smaller were found in larger
}

void prune_paths(Paths *paths, int *num_paths, int num_jobs) {
    // Initialize path_to_prune array
    int *path_to_prune = (int *)calloc(*num_paths, sizeof(int));
    if (!path_to_prune) {
        fprintf(stderr, "Memory allocation failed for path_to_prune\n");
        return;
    }

    JobPaths *job_paths = NULL;
    group_paths_by_jobs(paths, *num_paths, &job_paths, num_jobs);

    for (int j = 0; j < num_jobs; j++) {
        JobPaths *job = &job_paths[j];

        for (int i = 0; i < job->num_paths; i++) {
            if (path_to_prune[job->path_indices[i]]) continue;
            Paths *p1 = &paths[job->path_indices[i]];

            for (int l = i + 1; l < job->num_paths; l++) {
                if (path_to_prune[job->path_indices[l]]) continue;
                Paths *p2 = &paths[job->path_indices[l]];
                if (p1->total_costs >= p2->total_costs &&
                    is_subset(p2->overlaps, p2->number_of_overlaps, p1->overlaps, p1->number_of_overlaps)) {
                    path_to_prune[job->path_indices[i]] = 1;
                } else if (p2->total_costs >= p1->total_costs &&
                           is_subset(p1->overlaps, p1->number_of_overlaps, p2->overlaps, p2->number_of_overlaps)) {
                    path_to_prune[job->path_indices[l]] = 1;
                }
            }
        }
    }

    // Create a new Paths array with pruned paths
    int new_num_paths = 0;
    for (int i = 0; i < *num_paths; i++) {
        if (path_to_prune[i] == 0) {
            paths[new_num_paths] = paths[i];
            new_num_paths++;
        }
    }

    *num_paths = new_num_paths;
    free(path_to_prune);
    // Free job_paths if dynamically allocated in group_paths_by_jobs
}


// Definition of path_in_list function
static inline int path_in_list(int *list, int list_size, int value) {
    for (int i = 0; i < list_size; i++) {
        if (list[i] == value) {
            return 1;  // Found the value in the list
        }
    }
    return 0;  // Value not found in the list
}

int compare_paths(const void *a, const void *b) {
    Paths *pathA = (Paths *)a;
    Paths *pathB = (Paths *)b;
    
    // Compare based on total_costs
    return (pathA->total_costs > pathB->total_costs) - (pathA->total_costs < pathB->total_costs);
}

void remove_paths_section_2(Paths *paths, int *num_paths, int num_jobs, float ub) {
    int old_len_df = 0;
    qsort(paths, *num_paths, sizeof(Paths), compare_paths);

    int *path_to_prune = (int *)calloc(*num_paths, sizeof(int));  // Initialize to zero
    if (!path_to_prune) {
        fprintf(stderr, "Memory allocation failed for path_to_prune\n");
        return;
    }

    while (old_len_df != *num_paths) {
        JobPaths *job_paths = NULL;
        group_paths_by_jobs(paths, *num_paths, &job_paths, num_jobs);

        old_len_df = *num_paths;

        // Iterate through jobs
        for (int j = 0; j < num_jobs; j++) {
            JobPaths *job = &job_paths[j];

            // Iterate through job-specific paths
            for (int i = 0; i < job->num_paths; i++) {
                Paths *p1 = &paths[job->path_indices[i]];
                float new_ub = ub - p1->total_costs + 0.01;

                for (int j_2 = 0; j_2 < num_jobs; j_2++) {
                    if (j != j_2) {
                        JobPaths *job_2 = &job_paths[j_2];
                        float minimum_cost_job = FLT_MAX;

                        for (int k = 0; k < job_2->num_paths; k++) {
                            Paths *p2 = &paths[job_2->path_indices[k]];
                            if (!path_in_list(p1->overlaps, p1->number_of_overlaps, p2->path)) {
                                minimum_cost_job = p2->total_costs;
                                break;
                            }
                        }

                        if (minimum_cost_job == FLT_MAX) {
                            path_to_prune[job->path_indices[i]] = 1;
                            break;
                        } else {
                            new_ub -= minimum_cost_job;
                            if (new_ub < 0) {
                                path_to_prune[job->path_indices[i]] = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Create a new Paths array with pruned paths
        int new_num_paths = 0;
        for (int i = 0; i < *num_paths; i++) {
            if (path_to_prune[i] == 0) {
                paths[new_num_paths] = paths[i];
                new_num_paths++;
            }
        }

        *num_paths = new_num_paths;

        // Reset path_to_prune for the next iteration
        memset(path_to_prune, 0, *num_paths * sizeof(int));
    }

    free(path_to_prune);
}



