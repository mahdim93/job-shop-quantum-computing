#ifndef PATH_GENERATION_H
#define PATH_GENERATION_H

#define max(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
    Paths *paths;
    Paths *incomplete_paths;
    int num_paths;
    int num_incomplete_paths;
} PathsResult;

// Function declarations
void find_overlaps(int num_paths, Paths *all_paths, JobSchedulingData *instance);
void check_overlaps(Paths *path_1, Paths *path_2, JobSchedulingData *instance);

void backtrack(Paths *path, float lb, float ub, JobSchedulingData *job_scheduling_problem, 
               Paths *incomplete_paths, Paths *paths, int *num_paths, int *num_incomplete_paths);
               
void initialize_path(Paths *path, int num_machines);

PathsResult generate_all_paths(JobSchedulingData *job_scheduling_problem, float lb, float ub);

#endif // PATH_GENERATION_H

