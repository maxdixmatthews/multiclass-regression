import os
from run_datasets_tree_by_layer import run_layer_by_layer_nds
import cProfile
import pstats
import sqlalchemy as sa
import pandas as pd
from datetime import datetime
import io
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
    traversal_type = "layer-by-layer"
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
            filename, model_types, kfold_seed = input_data.split('|')
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                run_layer_by_layer_nds(filename.split("=")[1], model_types.split("=")[1].split(","), int(kfold_seed.split("=")[1]))
                profiler.disable()
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats('cumulative').print_stats(30)
                stats_output = stream.getvalue()
                engine = sa.create_engine(os.environ['ML_POSTGRESS_URL'] + "/max")
                df = pd.DataFrame({
                    "filename": [filename.split("=")[1]],
                    "model_types":[model_types.split("=")[1].split(",")],
                    "traversal_type":[traversal_type],
                    "kfold_seed": [int(kfold_seed.split("=")[1])],
                    "profile_stats":[str(stats_output)],
                    "run_timestamp":[datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                })
                df.to_sql('profiling_stats', engine, if_exists='append', index=False)
            except Exception as e:
                print(f"Error processing {input_data}: {e}")
                profiler.disable()
            append_to_file(output_file, input_data)

if __name__ == "__main__":
    # Replace 'inputs.txt' and 'outputs.txt' with your actual file paths
    input_file = 'multi_runs/inputs_all_layers.txt'
    output_file = 'multi_runs/outputs_all_layers.txt'
    run_all(input_file, output_file)
