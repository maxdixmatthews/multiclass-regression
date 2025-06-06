import os
from run_datasets_kfolds import main
import cProfile
import pstats

# Define the function you want to call
def process_function(input_data):
    """
    Function to process each input.
    Replace this logic with the actual functionality you need.
    """
    return f"Processed: {input_data}"

def read_file_to_set(file_path):
    """Reads a file and returns a set of lines (stripped of whitespace)."""
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def append_to_file(file_path, data):
    """Appends a line of data to a file."""
    with open(file_path, 'a') as file:
        file.write('\n' + data)

def ordered_difference(list1, list2):
    """
    Returns the difference between list1 and list2, preserving the order of list1.
    """
    return [item for item in list1 if item not in list2]
def run_all(input_file, output_file):
    # Read the input and output files into sets
    while(True):
        inputs = read_file_to_set(input_file)
        outputs = read_file_to_set(output_file)

        # Get the inputs that are not already processed
        remaining_inputs = ordered_difference(inputs, outputs)

        for input_data in remaining_inputs:
            if input_data == '' or '#' in input_data:
                remaining_inputs.remove(input_data)

        if len(remaining_inputs) == 0:
            print("Finished all inputs")
            break
        
        remaining_inputs = [remaining_inputs[0]]

        for input_data in remaining_inputs:
            # Call the function with the input data
            # result = process_function(input_data)
            
            # Append the processed result to the output file
            filename, model_types = input_data.split('|')
            profiler = cProfile.Profile()
            profiler.enable()
            main(filename.split("=")[1], model_types.split("=")[1].split(","))
            append_to_file(output_file, input_data)
            profiler.disable()

        # Print the profiling results
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative').print_stats(30)
            print(stats)

if __name__ == "__main__":
    # Replace 'inputs.txt' and 'outputs.txt' with your actual file paths
    input_file = 'multi_runs/inputs_inner_kfolds.txt'
    output_file = 'multi_runs/outputs_inner_kfolds.txt'
    run_all(input_file, output_file)
