#include <stdio.h>
#include <stdlib.h>
#include "read_file.h"
#include "helpers/cJSON.h"

void parse_job_scheduling_data(char *json_string, JobSchedulingData *data) {

	    // Parse the JSON string
    cJSON *json = cJSON_Parse(json_string);
    
    if (!json) {
        fprintf(stderr, "Failed to parse JSON: %s\n", cJSON_GetErrorPtr());
        free(json_string);
        exit(EXIT_FAILURE);
    }

    // Load scalar values
    data->num_jobs = cJSON_GetObjectItemCaseSensitive(json, "num_jobs")->valueint;
    data->num_machines = cJSON_GetObjectItemCaseSensitive(json, "num_machines")->valueint;
    data->num_time_slots = cJSON_GetObjectItemCaseSensitive(json, "num_time_slots")->valueint;

    // Allocate and load arrays

    cJSON *d_json = cJSON_GetObjectItemCaseSensitive(json, "d");
    data->d = (int *)malloc(data->num_jobs * sizeof(int));
    
    cJSON *w_json = cJSON_GetObjectItemCaseSensitive(json, "w");
    data->w = (int *)malloc(data->num_jobs * sizeof(int));
    
    cJSON *f_json = cJSON_GetObjectItemCaseSensitive(json, "f");
    data->f = (int *)malloc(data->num_jobs * sizeof(int));
    
    cJSON *p_json = cJSON_GetObjectItemCaseSensitive(json, "p");
    data->p = (int **)malloc(data->num_jobs * sizeof(int *));
    data->agg_p = (int **)malloc(data->num_jobs * sizeof(int *));

    
    cJSON *E_json = cJSON_GetObjectItemCaseSensitive(json, "E");
    data->E = (float **)malloc(data->num_jobs * sizeof(float *));
    
    cJSON *aggE_json = cJSON_GetObjectItemCaseSensitive(json, "aggE");
    data->aggE = (float **)malloc(data->num_jobs * sizeof(float *));
    
    cJSON *mp_json = cJSON_GetObjectItemCaseSensitive(json, "mp");
    data->mp = (int **)malloc(data->num_jobs * sizeof(int *));
    
    cJSON *sequence_json = cJSON_GetObjectItemCaseSensitive(json, "sequence");
    data->sequence = (int **)malloc(data->num_jobs * sizeof(int *));
    data->machine_idices = (int **)malloc(data->num_jobs * sizeof(int *));
    
    for (int i = 0; i < data->num_jobs; i++) {
    
		int required_size = snprintf(NULL, 0, "%d", i + 1) + 1;
		char *key = malloc(required_size);
		snprintf(key, required_size, "%d", i + 1);

        data->d[i] = cJSON_GetObjectItemCaseSensitive(d_json, key)->valueint;
        data->w[i] = cJSON_GetObjectItemCaseSensitive(w_json, key)->valueint;
        data->f[i] = cJSON_GetObjectItemCaseSensitive(f_json, key)->valueint;
        
        data->p[i] = (int *)malloc(data->num_machines * sizeof(int));
        data->agg_p[i] = (int *)malloc(data->num_machines * sizeof(int));
        data->E[i] = (float *)malloc(data->num_machines * sizeof(float));
        data->aggE[i] = (float *)malloc(data->num_machines * sizeof(float));
        data->mp[i] = (int *)malloc(data->num_machines * sizeof(int));
        data->sequence[i] = (int *)malloc(data->num_machines * sizeof(int));
        data->machine_idices[i] = (int *)malloc(data->num_machines * sizeof(int));
        
        for (int j = 0; j < data->num_machines; j++) {
        	
        	int required_size = snprintf(NULL, 0, "(%d,%d)", i + 1, j + 1) + 1;
            char *key = malloc(required_size);
            snprintf(key, required_size, "(%d,%d)", i + 1, j + 1);

            cJSON *p_item = cJSON_GetObjectItemCaseSensitive(p_json, key);
            cJSON *E_item = cJSON_GetObjectItemCaseSensitive(E_json, key);
            cJSON *aggE_item = cJSON_GetObjectItemCaseSensitive(aggE_json, key);
            cJSON *mp_item = cJSON_GetObjectItemCaseSensitive(mp_json, key);

            if (!p_item || !E_item || !aggE_item || !mp_item) {
                fprintf(stderr, "Warning: Missing data for (%d,%d)\n", i + 1, j + 1);
                data->p[i][j] = 0;
                data->E[i][j] = 0.0f;
                data->aggE[i][j] = 0.0f;
                data->mp[i][j] = 0;
            } else {
                data->p[i][j] = p_item->valueint;
                data->E[i][j] = (float)E_item->valuedouble;
                data->aggE[i][j] = (float)aggE_item->valuedouble;
                data->mp[i][j] = mp_item->valueint;
            }

        }
    }
    
	cJSON *sequence_key, *sequence_value;
	cJSON_ArrayForEach(sequence_key, sequence_json) {
	    sequence_value = cJSON_GetObjectItem(sequence_json, sequence_key->string);
	    int i = atoi(sequence_key->string);
	    cJSON *seq_item;
	    int j = 0;
	    cJSON_ArrayForEach(seq_item, sequence_value) {
	        data->machine_idices[i-1][seq_item->valueint -1] = j;
	        data->sequence[i - 1][j++] = seq_item->valueint -1;
	    }
	}
	
    for (int i = 0; i < data->num_jobs; ++i) {
    	data->agg_p[i][data->sequence[i][0]] = data->p[i][data->sequence[i][0]];
        for (int j = 1; j < data->num_machines; ++j) {
            data->agg_p[i][data->sequence[i][j]] = data->agg_p[i][data->sequence[i][j-1]] + data->p[i][data->sequence[i][j]];
        }
    }
    
    // Free temporary JSON string and object
    cJSON_Delete(json);
}




// Modified read_file function with dynamic size for start_times and costs
Paths* read_file(const char *filename, int num_paths, int num_machines) {
    FILE *file = fopen(filename, "r");
    int num_costs = num_machines + 1;
    
    // Allocate memory for the array of Paths
    Paths *data = malloc(num_paths * sizeof(Paths));
    if (data == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }

    int idx = 0;
    int job = 0;
    // Read the file line by line
    while (fscanf(file, "%d %d", &job, &idx) == 2) {
    
        // Allocate memory for start_times and costs in the structure
        idx -=1;
        data[idx].job = job-1;
        data[idx].path = idx;
        data[idx].start_times = malloc(num_machines * sizeof(int));
        data[idx].costs = malloc(num_costs * sizeof(float));
        
        // Read start_times (integers)
        for (int i = 0; i < num_machines; i++) {
            fscanf(file, "%d", &data[idx].start_times[i]);
        }
        // Read costs (floats)
        for (int i = 0; i < num_costs; i++) {
            fscanf(file, "%f", &data[idx].costs[i]);
        }
        // Read total_costs (last float item)
        fscanf(file, "%f", &data[idx].total_costs);
    }

    fclose(file);
    
    return data;
}

void read_min_costs_from_file(const char *filename, Paths *paths, int num_jobs) {
    FILE *file = fopen(filename, "r");

    int idx = 0;
    // Read the file line by line
    while (fscanf(file, "%d", &idx) == 1) {    
        // Allocate memory
        idx -=1;
        paths[idx].min_costs_per_jobs = malloc(num_jobs * sizeof(float));
        // Read costs (floats)
        for (int i = 0; i < num_jobs; i++) {
            fscanf(file, "%f", &paths[idx].min_costs_per_jobs[i]);
		}
    }

    fclose(file);
}

// Comparison function for integers
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

void read_overlaps_per_paths_from_file(const char *filename, Paths *paths, int num_paths) {
    FILE *file = fopen(filename, "r");

    int idx = 0;
    // Read the file line by line
    while (fscanf(file, "%d", &idx) == 1) {    
        // Allocate memory
        idx -=1;
        // Read the number of overlaps
        int number_of_elements_in_the_line;
        fscanf(file, "%d", &number_of_elements_in_the_line);
        paths[idx].number_of_overlaps = number_of_elements_in_the_line;
        
        paths[idx].overlaps = malloc(number_of_elements_in_the_line * sizeof(int));
        // Read overlaps (floats)
        for (int i = 0; i < number_of_elements_in_the_line; i++) {
        	int mpath;
            fscanf(file, "%d", &mpath);
            paths[idx].overlaps[i] = mpath-1;
		}
	    qsort(paths[idx].overlaps, paths[idx].number_of_overlaps, sizeof(int), compare);
    }
    fclose(file);
}

void write_paths_to_file(Paths *paths, int num_paths, const char *filename) {
    // Open the file for writing
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Write each path to the file
    for (int i = 0; i < num_paths; i++) {
        fprintf(file, "%d\n", paths[i].path);  // Assuming `path` is an integer
    }

    // Close the file
    fclose(file);
}


void print_paths(Paths *paths, int num_paths, int num_machines, int num_jobs) {
    for (int i = 0; i < num_paths; i++) {
        printf("Path %d:\n", i);
        printf("\tJob: %d, Path: %d\n", paths[i].job, paths[i].path);

        // Print start times
        printf("\tStart Times: ");
        for (int j = 0; j < num_machines; j++) {
            printf("%d ", paths[i].start_times[j]);
        }
        printf("\n");

        // Print costs
        printf("\tCosts: ");
        for (int j = 0; j < num_machines + 1; j++) {
            printf("%.2f ", paths[i].costs[j]);
        }
        printf("\n");
		
        // Print min costs per jobs
        printf("\tMin Costs per Jobs: ");
        for (int j = 0; j < num_jobs; j++) {
            printf("%.2f ", paths[i].min_costs_per_jobs[j]);
        }
        printf("\n");
        
        // Print min costs per jobs
        printf("\tOverlaps: ");
        for (int j = 0; j < paths[i].number_of_overlaps; j++) {
            printf("%d ", paths[i].overlaps[j]);
        }
        printf("\n");
		
        // Print total cost
        printf("\tTotal Cost: %.2f\n", paths[i].total_costs);
    }
}

void print_job_scheduling_data(JobSchedulingData *data) {
    printf("num_jobs: %d\n", data->num_jobs);
    printf("num_machines: %d\n", data->num_machines);
    printf("num_time_slots: %d\n", data->num_time_slots);

    // Print processing times (p)
    printf("\nProcessing times (p):\n");
    for (int i = 0; i < data->num_jobs; ++i) {
        for (int j = 0; j < data->num_machines; ++j) {
            printf("p[%d][%d] = %d\n", i + 1, j + 1, data->p[i][j]);
        }
    }
    
    printf("\nAggregate Processing times (p):\n");
    for (int i = 0; i < data->num_jobs; ++i) {
        for (int j = 0; j < data->num_machines; ++j) {
            printf("agg_p[%d][%d] = %d\n", i + 1, j + 1, data->agg_p[i][j]);
        }
    }

    // Print due dates (d)
    printf("\nDue dates (d):\n");
    for (int i = 0; i < data->num_jobs; ++i) {
        printf("d[%d] = %d\n", i + 1, data->d[i]);
    }

    // Print tardiness cost (w)
    printf("\nTardiness cost (w):\n");
    for (int i = 0; i < data->num_jobs; ++i) {
        printf("w[%d] = %d\n", i + 1, data->w[i]);
    }

    // Print earliness cost (E)
    printf("\nEarliness cost (E):\n");
    for (int i = 0; i < data->num_jobs; ++i) {
        for (int j = 0; j < data->num_machines; ++j) {
            printf("E[%d][%d] = %.3f\n", i + 1, j + 1, data->E[i][j]);
        }
    }

    // Print aggregate earliness cost (aggE)
    printf("\nAggregate earliness cost (aggE):\n");
    for (int i = 0; i < data->num_jobs; ++i) {
        for (int j = 0; j < data->num_machines; ++j) {
            printf("aggE[%d][%d] = %.3f\n", i + 1, j + 1, data->aggE[i][j]);
        }
    }

    // Print machine sequences (sequence)
    printf("\nMachine sequences (sequence):\n");
    for (int i = 0; i < data->num_jobs; ++i) {
        printf("sequence[%d]: ", i + 1);
        for (int j = 0; j < data->num_machines; ++j) {
            printf("%d, %d; ", data->sequence[i][j], data->machine_idices[i][data->sequence[i][j]]);
        }
        printf("\n");
    }

    // Print machine precedence (mp)
    printf("\nMachine precedence (mp):\n");
    for (int i = 0; i < data->num_jobs; ++i) {
        for (int j = 0; j < data->num_machines; ++j) {
            printf("mp[%d][%d] = %d\n", i + 1, j + 1, data->mp[i][j]);
        }
    }

    // Print other data (f)
    printf("\nOther data (f):\n");
    for (int i = 0; i < data->num_jobs; ++i) {
        printf("f[%d] = %d\n", i + 1, data->f[i]);
    }
}


// Function to free the allocated memory
void free_data(Paths *data, int num_paths) {
    for (int i = 0; i < num_paths; i++) {
        free(data[i].start_times);
        free(data[i].costs);
        free(data[i].min_costs_per_jobs);
        free(data[i].overlaps);
    }
    free(data);
}

void free_job_scheduling_data(JobSchedulingData *data) {
    // Free dynamic arrays
    for (int i = 0; i < data->num_jobs; i++) {
        free(data->p[i]);
        free(data->E[i]);
        free(data->agg_p[i]);
        free(data->aggE[i]);
        free(data->sequence[i]);
		free(data->machine_idices[i]);
		free(data->mp[i]);
        // Free other 2D arrays similarly
    }
    free(data->p);
    free(data->agg_p);
    free(data->d);
    free(data->w);
    free(data->E);
    free(data->aggE);
    free(data->sequence);
    free(data->machine_idices);
    free(data->mp);
    free(data->f);
    // Free other arrays
}


