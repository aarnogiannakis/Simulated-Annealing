using Random
using Statistics
using Dates 
using FileIO
#*************************************************************************************************************


#*************************************************************************************************************
# Function to read the input instance from a file
function read_instance(filename::String)
    f = open(filename)
    readline(f)  # Skip the first line (comment)
    n_jobs, n_processors, upper_bound = parse.(Int, split(readline(f)))
    readline(f)  # Skip the third line (comment)
    durations = zeros(Int, n_jobs, n_processors)
    
    for i in 1:n_jobs
        durations[i, :] = parse.(Int, split(readline(f)))
    end
    
    readline(f)  # Skip the fifth line (comment)
    processors = zeros(Int, n_jobs, n_processors)
    
    for i in 1:n_jobs
        processors[i, :] = parse.(Int, split(readline(f)))
    end
    
    close(f)
    return n_jobs, n_processors, upper_bound, durations, processors
end
#*************************************************************************************************************


#*************************************************************************************************************
# Struct to store information about each operation
mutable struct Operation
    job_id::Int
    operation_id::Int
    duration::Int
    processor_id::Int
    start_time::Int
    end_time::Int
end
#*************************************************************************************************************


#*************************************************************************************************************
# Initializes operations from duration and processor matrices
function initialize_operations(durations, processors)
    operations = Operation[]  # Initialize an empty array of Operation structs
    num_jobs, num_operations = size(durations)  # Assuming duration and processor matrices have the same size
    
    for job_id in 1:num_jobs
        for operation_id in 1:num_operations
            duration = durations[job_id, operation_id]
            processor_id = processors[job_id, operation_id]
            push!(operations, Operation(job_id, operation_id, duration, processor_id, 0, 0))
        end
    end
    
    return operations
end

# Groups operations by their assigned processor
function group_operations_by_processor(operations)
    max_processor_id = maximum(op.processor_id for op in operations)
    processor_operations = [Vector{Operation}() for _ in 1:max_processor_id]
    
    for op in operations
        push!(processor_operations[op.processor_id], op)
    end
    
    return processor_operations
end

# Generates unique permutations of integers from 1 to k
function generate_unique_permutations(k)
    first_row = shuffle(1:k)
    permutation_matrix = [first_row]
    
    shifts = [circshift(first_row, shift) for shift in 1:(k - 1)]
    shuffle!(shifts)
    
    for shift in shifts[1:(k - 1)]
        push!(permutation_matrix, shift)
    end
    
    return Array{Int}(hcat(permutation_matrix...)')
end

# Reorders operations according to a specified order vector
function reorder_operations(operations, order_vector)
    order_dict = Dict(order_vector[i] => i for i in 1:length(order_vector))
    sorted_operations = sort(operations, by = op -> order_dict[op.job_id])
    
    return sorted_operations
end

# Orders operations for each processor based on the permutations
function finalize_operation_order(temp_processes, permutations, num_processors)
    all_processes = []
    
    for i in 1:num_processors
        reordered = reorder_operations(temp_processes[i], permutations[i, :])
        push!(all_processes, reordered)
    end
    
    return all_processes
end
#*************************************************************************************************************


#*************************************************************************************************************
# Finds the start and end times for each operation
function calculate_start_times(all_processes, num_operations)
    for i in 1:num_operations
        first_op = all_processes[i][1]
        first_op.start_time = 0
        first_op.end_time = first_op.duration
    end
    
    for j in 2:num_operations
        for i in 1:num_operations
            current_op = all_processes[i][j]
            job_id = current_op.job_id
            
            max_end_time_for_job = 0
            for k in 1:num_operations
                if all_processes[k][j - 1].job_id == job_id
                    max_end_time_for_job = max(max_end_time_for_job, all_processes[k][j - 1].end_time)
                end
            end
            
            current_op.start_time = max(all_processes[i][j - 1].end_time, max_end_time_for_job)
            current_op.end_time = current_op.start_time + current_op.duration
        end
    end
    
    return all_processes
end

# Calculates the makespan of the operations
function calculate_makespan(processes, num_operations)
    max_span = 0
    for process in processes
        max_span = max(max_span, process[num_operations].end_time)
    end
    
    return max_span
end
#*************************************************************************************************************


#*************************************************************************************************************

# Generates an initial solution based on the input instance
function generate_initial_solution(filename)
    n_jobs, n_processors, upper_bound, durations, processors = read_instance(filename)
    
    operations = initialize_operations(durations, processors)
    temp_processes = group_operations_by_processor(operations)
    permutations = generate_unique_permutations(n_processors)
    
    ordered_processes = finalize_operation_order(temp_processes, permutations, n_processors)
    final_processes = calculate_start_times(ordered_processes, n_processors)
    
    return final_processes
end
#*************************************************************************************************************


#*************************************************************************************************************
# Swaps operations between two jobs in the same processor
function swap_operations(processes, job_id1, job_id2)
    new_solution = deepcopy(processes)
    
    for process in new_solution
        idx1, idx2 = 0, 0
        for i in 1:length(process)
            if process[i].job_id == job_id1
                idx1 = i
            elseif process[i].job_id == job_id2
                idx2 = i
            end
        end
        
        if idx1 != 0 && idx2 != 0
            process[idx1], process[idx2] = process[idx2], process[idx1]
        end
    end
    
    return new_solution
end

# Resets the start and end times of operations
function reset_operation_times!(processes)
    for process in processes
        for op in process
            op.start_time = 0
            op.end_time = 0
        end
    end
    return processes
end
#*************************************************************************************************************


#*************************************************************************************************************
# Generates a random neighboring solution by swapping operations
function generate_random_neighbor(initial_solution, num_operations)
    current_solution = deepcopy(initial_solution)
    
    while true
        job_id1 = rand(1:num_operations)
        job_id2 = rand(1:num_operations)
        
        if job_id1 != job_id2
            swapped_solution = swap_operations(current_solution, job_id1, job_id2)
            reset_operation_times!(swapped_solution)
            updated_solution = calculate_start_times(swapped_solution, num_operations)
            updated_makespan = calculate_makespan(updated_solution, num_operations)
            
            return updated_solution, updated_makespan
        end
    end
end
#*************************************************************************************************************


#*************************************************************************************************************
# Implements the Simulated Annealing algorithm
function simulated_annealing(filename, sol_file, time_limit, initial_temperature, cooling_rate)
    n_jobs, n_processors, upper_bound, durations, processors = read_instance(filename)

    initial_solution = generate_initial_solution(filename)
    initial_makespan = calculate_makespan(initial_solution, n_processors)
    println("Initial Makespan: ", initial_makespan)

    iteration = 1
    temperature = initial_temperature
    no_improvement_counter = 0
    no_improvement_limit = 500
    
    start_time = now()

    while (now() - start_time) < Second(time_limit)
        current_solution, current_makespan = generate_random_neighbor(initial_solution, n_processors)
        
        if current_makespan < initial_makespan
            initial_solution = deepcopy(current_solution)
            initial_makespan = current_makespan
            no_improvement_counter = 0
        else
            random_number = rand()
            acceptance_probability = exp(-(current_makespan - initial_makespan) / temperature)
            
            if random_number < acceptance_probability
                initial_solution = deepcopy(current_solution)
                initial_makespan = current_makespan
            end
            no_improvement_counter += 1
        end

        if no_improvement_counter >= no_improvement_limit
            temperature = initial_temperature  # Reheating step
            no_improvement_counter = 0
        end

        temperature *= cooling_rate
        iteration += 1
    end

    final_schedule = validate_solution(initial_solution)
    save_solution_to_file(final_schedule, sol_file)

    return initial_solution, initial_makespan, iteration
end
#*************************************************************************************************************


#*************************************************************************************************************
# Validates and returns the start times of all operations
function validate_solution(solution)
    num_jobs = maximum(op.job_id for process in solution for op in process)
    num_operations = maximum(op.operation_id for process in solution for op in process)

    start_times_matrix = zeros(Int, num_jobs, num_operations)
    
    for process in solution
        for op in process
            start_times_matrix[op.job_id, op.operation_id] = op.start_time
        end
    end
    
    return start_times_matrix
end
#*************************************************************************************************************


#*************************************************************************************************************
# Saves the solution (start times) to a file
function save_solution_to_file(matrix, filename)
    open(filename, "w") do file
        for row in eachrow(matrix)
            write(file, join(row, " "), "\n")
        end
    end
end
#*************************************************************************************************************


#*************************************************************************************************************
# Main function to execute the Simulated Annealing process
function main(args)
    if length(args) != 3
        println("Usage: julia script_name.jl <instance_file> <solution_file> <time_limit>")
        return
    end

    instance_file = args[1]
    solution_file = args[2]
    time_limit = parse(Int, args[3])  

    final_solution, final_makespan, iterations = simulated_annealing(instance_file, solution_file, time_limit, 1000, 0.99999)

    println("Processed instance from: $instance_file with time limit: $time_limit seconds")
    println("Solution saved to $solution_file")
    println("Final Makespan: ", final_makespan)
    println("Number of iterations: ", iterations)
    
    println("\nFinal solution:")
    for process in final_solution
        println(process)
    end
end

main(ARGS)
