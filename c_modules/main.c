#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "read_file.h"
#include "path_generation.h"
#include "pruning.h"
#include "helpers/cJSON.h"

typedef struct {
	Paths *paths;
	Paths *incomplete_paths;
	int num_paths;
	int num_paths_before;
	int num_overlaps;
	int num_overlaps_before;
	int num_incomplete_paths;
	double time_to_find_paths;
	double time_to_find_overlaps;
    double time_to_remove_paths_mahdi;
    double time_to_remove_paths_tom;
    int number_of_path_less_than_max_cost;
} Results;

Results my_main(char *json_string, Results *results, float lb, float ub, float max_cost_of_optimal);
void free_results(Results *res);

int main(int argc, char *argv[]) {

    // Default values
    char *filename = "instance.json";
    float lb = 20.14;
    float ub = 20.14;

    // Check if arguments are passed and assign them
    if (argc > 1) {
        filename = argv[1]; // First argument for file path
    }
    if (argc > 2) {
        lb = atof(argv[2]); // Second argument for lower bound
    }
    if (argc > 3) {
        ub = atof(argv[3]); // Third argument for upper bound
    }
	
    // Parse the JSON file
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // Read the entire file into a string
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *json_string = (char *)malloc(file_size + 1);
    fread(json_string, 1, file_size, file);
    json_string[file_size] = '\0';
    fclose(file);
	
	//printf(" %s\n", json_string);
    Results results = my_main(json_string, NULL, lb, ub, 0.0);

    return 0;
}

Results my_main(char *json_string, Results *results, float lb, float ub, float max_cost_of_optimal){

	JobSchedulingData job_scheduling_problem;

	parse_job_scheduling_data(json_string, &job_scheduling_problem);

    int num_jobs = job_scheduling_problem.num_jobs;
    int num_machines = job_scheduling_problem.num_machines;

	int num_paths_before = 0;
	int num_overlaps = 0;
	int num_overlaps_before = 0;

	int number_of_path_less_than_max_cost = 0;
	float max_cost_of_optimal_solution = max_cost_of_optimal;
    //print_job_scheduling_data(&job_scheduling_problem);

     // Start timing
    clock_t start_time = clock();
    // find paths
    PathsResult result = generate_all_paths(&job_scheduling_problem, lb, ub);
	Paths *paths = result.paths;
	Paths *incomplete_paths = result.incomplete_paths;
	int num_paths = result.num_paths;
	int num_incomplete_paths = result.num_incomplete_paths;
    // Calculate the time to find all paths
    double time_to_find_paths = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    
     // Start timing
    start_time = clock();
    // find overlaps
    find_overlaps(num_paths, paths, &job_scheduling_problem);
    // Calculate the time to find overlaps paths
    double time_to_find_overlaps = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    
    num_paths_before = num_paths;
    for(int i = 0; i < num_paths; i++){
    	num_overlaps_before += paths[i].number_of_overlaps;
    	if(paths[i].total_costs <= max_cost_of_optimal_solution+0.001){
    		number_of_path_less_than_max_cost = number_of_path_less_than_max_cost + 1;
    	}
	}
	num_overlaps_before = num_overlaps_before/2;
   
    printf("Number of paths and overlaps before pruning: %d, %d\n", num_paths, num_overlaps_before);
    
    //print_paths(paths, num_paths, num_machines, num_jobs);
    // Start timing
    start_time = clock();
    // Remove paths and get the number of pruned paths
    remove_paths(paths, &num_paths, num_jobs, ub);
    // Calculate the time to remove paths
    double time_to_remove_paths_1 = (double)(clock() - start_time) / CLOCKS_PER_SEC;
	
     // Start timing
    start_time = clock();
    // Remove paths and get the number of pruned paths
    prune_paths(paths, &num_paths, num_jobs);
    // Calculate the time to remove paths
    double time_to_remove_paths_2 = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    
     // Start timing
    start_time = clock();
    // Remove paths and get the number of pruned paths
    remove_paths_section_2(paths, &num_paths, num_jobs, ub);
    // Calculate the time to remove paths
    double time_to_remove_paths_3 = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    // Output the results
    printf("Time to find paths, overlaps and prune paths: %.6f, %.6f, %.6f, %.6f seconds\n", 
    	time_to_find_paths, time_to_find_overlaps, time_to_remove_paths_2, time_to_remove_paths_1+time_to_remove_paths_3);
    	
    printf("Number of paths after pruning: %d\n", num_paths);
   	Results results_ = {paths, incomplete_paths, num_paths, num_paths_before, num_overlaps, num_overlaps_before, num_incomplete_paths, 
   		time_to_find_paths, time_to_find_overlaps, time_to_remove_paths_2, time_to_remove_paths_1+time_to_remove_paths_3, number_of_path_less_than_max_cost};

    free_job_scheduling_data(&job_scheduling_problem);

    return results_;
}


// Function to free a Results struct
void free_results(Results *res) {
    if (!res) return;

    if (res->paths) {
        free(res->paths);  // Only free if it was allocated
        res->paths = NULL;
    }

    if (res->incomplete_paths) {
        free(res->incomplete_paths);
        res->incomplete_paths = NULL;
    }

}

