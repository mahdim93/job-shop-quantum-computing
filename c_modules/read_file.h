#ifndef READ_FILE_H
#define READ_FILE_H

typedef struct {
    int num_jobs;
    int num_machines;
    int num_time_slots;

    // Dynamic arrays
    int **p;       // Processing times
    int **agg_p;   // Aggregate processing times
    int *d;        // Due dates
    int *w;        // Tardiness cost
    float **E;     // Earliness cost
    float **aggE;  // Aggregate earliness cost
    int **sequence; // Machine sequences
    int **machine_idices; // Index of machine in the sequence
    int **mp;      // Machine precedence
    int *f;        //
} JobSchedulingData;

typedef struct {
    int job;
    int path;
    int *start_times;  // num_machines items
    float *costs;      // num_machines + 1 items
    float total_costs;
    
    float *min_costs_per_jobs; // minimum costs per jobs among paths with no overlaps
    
    int number_of_overlaps;
    int *overlaps; // overlaps with other paths are saved here
    
    int idx_machine; // to check which machine of the path should be schedule next
} Paths;

typedef struct {
    int num_paths; // Number of paths for this job
    int *path_indices;   // Indices of paths in the original array
} JobPaths;

void parse_job_scheduling_data(char *json_string, JobSchedulingData *data);

void free_job_scheduling_data(JobSchedulingData *data);

// Function declarations
Paths* read_file(const char *filename, int num_paths, int num_machines);

void read_min_costs_from_file(const char *filename, Paths *paths, int num_jobs);

void read_overlaps_per_paths_from_file(const char *filename, Paths *paths, int num_paths);

void write_paths_to_file(Paths *paths, int num_paths, const char *filename);

void print_paths(Paths *paths, int num_paths, int num_machines, int num_jobs);

void print_job_scheduling_data(JobSchedulingData *data);

int compare(const void *a, const void *b);

void free_data(Paths *data, int num_paths);

#endif

