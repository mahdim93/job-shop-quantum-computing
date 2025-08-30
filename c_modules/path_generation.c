#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include "read_file.h"
#include "pruning.h"
#include "path_generation.h"

void backtrack(Paths *path, float lb, float ub, JobSchedulingData *job_scheduling_problem, 
               Paths *incomplete_paths, Paths *paths, int *num_paths, int *num_incomplete_paths) {
    
    int num_machines = job_scheduling_problem->num_machines;
    int tau = job_scheduling_problem->num_time_slots;

    // Get next machine
    int job = path->job;
    int next_machine = job_scheduling_problem->sequence[job][path->idx_machine];
    int last_machine = (path->idx_machine > 0) ? job_scheduling_problem->sequence[job][path->idx_machine - 1] : -1;

    // Determine start time range
    int earliest_start_time = (path->idx_machine > 0) ? 
        path->start_times[path->idx_machine - 1] + job_scheduling_problem->p[job][last_machine] : 0;
    int latest_start_time = tau - job_scheduling_problem->agg_p[job][last_machine];

    // Try all possible start times
    for(int next_start_time = earliest_start_time; next_start_time <= latest_start_time; ++next_start_time) {
        path->start_times[path->idx_machine] = next_start_time;
        
        // Recurse to the next machine
        if (path->idx_machine == 0){
		    path->idx_machine++;
		    backtrack(path, lb, ub, job_scheduling_problem, incomplete_paths, paths, num_paths, num_incomplete_paths);
		    path->idx_machine--; // Backtrack
        }
        else if(path->idx_machine < num_machines-1){

			path->costs[path->idx_machine-1] = job_scheduling_problem->aggE[job][last_machine]*
					(path->start_times[path->idx_machine] - path->start_times[path->idx_machine-1] - job_scheduling_problem->p[job][last_machine]);
			path->total_costs += path->costs[path->idx_machine-1];

			if(path->total_costs <= ub){
				if (path->costs[path->idx_machine-1] <= lb){
					path->idx_machine++;
					backtrack(path, lb, ub, job_scheduling_problem, incomplete_paths, paths, num_paths, num_incomplete_paths);
					path->idx_machine--; // Backtrack
				}else{
				    incomplete_paths[(*num_incomplete_paths)++] = *path;
					incomplete_paths[*num_incomplete_paths].start_times = (int *)malloc(sizeof(int) * num_machines);
					incomplete_paths[*num_incomplete_paths].costs = (float *)malloc(sizeof(float) * (num_machines + 1));
					incomplete_paths[*num_incomplete_paths].path = *num_incomplete_paths;
					memcpy(incomplete_paths[*num_incomplete_paths].start_times, path->start_times, sizeof(int) * num_machines);
					memcpy(incomplete_paths[*num_incomplete_paths].costs, path->costs, sizeof(float) * (num_machines + 1));
					(*num_incomplete_paths)++;
				}
			}
			path->total_costs -= path->costs[path->idx_machine-1];
        }else{
        
			path->costs[path->idx_machine-1] = job_scheduling_problem->aggE[job][last_machine]*
					(path->start_times[path->idx_machine] - path->start_times[path->idx_machine-1] - job_scheduling_problem->p[job][last_machine]);
					
			path->costs[path->idx_machine] = job_scheduling_problem->aggE[job][next_machine]*
					max(job_scheduling_problem->d[job] - path->start_times[path->idx_machine] - job_scheduling_problem->p[job][next_machine], 0);
					
			path->costs[path->idx_machine+1] = job_scheduling_problem->w[job]*
					max(path->start_times[path->idx_machine] + job_scheduling_problem->p[job][next_machine] - job_scheduling_problem->d[job], 0);
					
			path->total_costs += path->costs[path->idx_machine-1] + path->costs[path->idx_machine] + path->costs[path->idx_machine+1];
			
			if (path->total_costs <= ub){
				if(path->costs[path->idx_machine-1] <= lb && path->costs[path->idx_machine] <= lb && path->costs[path->idx_machine+1] <= lb){
					paths[*num_paths].job = path->job;
					paths[*num_paths].path = *num_paths;
					paths[*num_paths].total_costs = path->total_costs;
					paths[*num_paths].start_times = (int *)malloc(sizeof(int) * num_machines);
				    paths[*num_paths].costs = (float *)malloc(sizeof(float) * (num_machines + 1));
					memcpy(paths[*num_paths].start_times, path->start_times, sizeof(int) * num_machines);
					memcpy(paths[*num_paths].costs, path->costs, sizeof(float) * (num_machines + 1));
					(*num_paths)++;
				}else{
					incomplete_paths[*num_incomplete_paths].job = path->job;
					incomplete_paths[*num_incomplete_paths].path = *num_incomplete_paths;
					incomplete_paths[*num_incomplete_paths].total_costs = path->total_costs;
					incomplete_paths[*num_incomplete_paths].path = *num_incomplete_paths;
					incomplete_paths[*num_incomplete_paths].start_times = (int *)malloc(sizeof(int) * num_machines);
				    incomplete_paths[*num_incomplete_paths].costs = (float *)malloc(sizeof(float) * (num_machines + 1));
					memcpy(incomplete_paths[*num_incomplete_paths].start_times, path->start_times, sizeof(int) * num_machines);
					memcpy(incomplete_paths[*num_incomplete_paths].costs, path->costs, sizeof(float) * (num_machines + 1));
					(*num_incomplete_paths)++;
				}
			}
			path->total_costs -= path->costs[path->idx_machine-1] + path->costs[path->idx_machine] + path->costs[path->idx_machine+1];
        }
    }
}


void initialize_path(Paths *path, int num_machines) {

    path->start_times = (int *)malloc(sizeof(int) * num_machines);
    path->costs = (float *)malloc(sizeof(float) * (num_machines + 1));
    
    path->overlaps = NULL;
    path->min_costs_per_jobs = NULL;
    path->idx_machine = 0;
    path->total_costs = 0.0;
}

PathsResult generate_all_paths(JobSchedulingData *job_scheduling_problem, float lb, float ub) {
    int num_jobs = job_scheduling_problem->num_jobs;
    int num_machines = job_scheduling_problem->num_machines;
    int tau = job_scheduling_problem->num_time_slots;

    int max_expected_paths = num_jobs * num_machines * tau*1000;

    // Allocate space for paths
    Paths *paths = (Paths *)malloc(sizeof(Paths) * max_expected_paths);
    Paths *incomplete_paths = (Paths *)malloc(sizeof(Paths) * max_expected_paths);

    int num_paths = 0;
    int num_incomplete_paths = 0;

    // Generate paths for each job
    for (int job = 0; job < num_jobs; ++job) {
        Paths initial_path = {0};
        initialize_path(&initial_path, num_machines);
        initial_path.job = job;

        backtrack(&initial_path, lb, ub, job_scheduling_problem, incomplete_paths, paths, &num_paths, &num_incomplete_paths);

        free(initial_path.start_times);
        free(initial_path.costs);
    }
    
    PathsResult result = {paths, incomplete_paths, num_paths, num_incomplete_paths};
    return result;
}



void find_overlaps(int num_paths, Paths *all_paths, JobSchedulingData *instance) {

    JobPaths *job_paths = NULL;
    group_paths_by_jobs(all_paths, num_paths, &job_paths, instance->num_jobs);
    

    for (int i = 0; i < num_paths; i++) {
        Paths *path_1 = &all_paths[i];
        path_1->overlaps = (int *)malloc(num_paths * sizeof(int));
        path_1->number_of_overlaps = 0;
        path_1->min_costs_per_jobs = (float *)malloc(instance->num_jobs * sizeof(float));
        for (int q = 0; q < instance->num_jobs; q++) {
        	if(q !=path_1->job)
				path_1->min_costs_per_jobs[q] = FLT_MAX;
		}
		path_1->min_costs_per_jobs[path_1->job] = path_1->total_costs;
    }

    for (int j = 0; j < instance->num_jobs; j++) {
        JobPaths *job_1 = &job_paths[j];
        for (int i = 0; i < job_1->num_paths; i++) {
            Paths *path_1 = &all_paths[job_1->path_indices[i]];
			
            for (int k = j + 1; k < instance->num_jobs; k++) {
                JobPaths *job_2 = &job_paths[k];
                for (int l = 0; l < job_2->num_paths; l++) {
                    Paths *path_2 = &all_paths[job_2->path_indices[l]];
					
                    // Call check_overlaps with the correct arguments
                    check_overlaps(path_1, path_2, instance);
                }
            }
        }
    }
    
}
void check_overlaps(Paths *path_1, Paths *path_2, JobSchedulingData *instance) {
    bool is_overlapping = false;

    // Cache job IDs for paths
    int job_1 = path_1->job;
    int job_2 = path_2->job;

    // Check overlaps across all machines
    for (int i = 0; i < instance->num_machines; i++) {
        int machine = instance->sequence[job_1][i];
        int machine_idx = instance->machine_idices[job_2][machine];

        // Precompute time intervals
        int start_time_1 = path_1->start_times[i];
        int end_time_1 = start_time_1 + instance->p[job_1][machine];

        int start_time_2 = path_2->start_times[machine_idx];
        int end_time_2 = start_time_2 + instance->p[job_2][machine];

        // Check for overlap
        if ((start_time_1 < end_time_2) && (start_time_2 < end_time_1)) {
            is_overlapping = true;
            break; // Stop further checks if overlap is found
        }
    }

    if (is_overlapping) {
        // Efficient overlap array handling for path_2
        path_2->overlaps[path_2->number_of_overlaps++] = path_1->path;
        // Efficient overlap array handling for path_1
        /*
        if (path_1->number_of_overlaps == 0) {
            path_1->overlap_capacity = 4; // Initial capacity
            path_1->overlaps = malloc(sizeof(int) * path_1->overlap_capacity);
        } else if (path_1->number_of_overlaps == path_1->overlap_capacity) {
            path_1->overlap_capacity *= 2; // Double capacity
            int *new_overlaps = realloc(path_1->overlaps, sizeof(int) * path_1->overlap_capacity);
            if (!new_overlaps) {
                fprintf(stderr, "Memory allocation failed for path_1 overlaps\n");
                exit(EXIT_FAILURE);
            }
            path_1->overlaps = new_overlaps;
        }
        */
        path_1->overlaps[path_1->number_of_overlaps++] = path_2->path;
    } else {
        // Update minimum costs using a single check
        if (path_1->min_costs_per_jobs[job_2] > path_2->total_costs) {
            path_1->min_costs_per_jobs[job_2] = path_2->total_costs;
        }
        if (path_2->min_costs_per_jobs[job_1] > path_1->total_costs) {
            path_2->min_costs_per_jobs[job_1] = path_1->total_costs;
        }
    }
}



