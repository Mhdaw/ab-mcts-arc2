import json
import os

def split_arc_solutions(solutions_file_path, output_directory):
    """
    Splits a single ARC solutions JSON file into individual task solution files.

    Args:
        solutions_file_path (str): Path to the main solutions JSON file.
        output_directory (str): Directory where the individual task files will be saved.
    """
    try:
        with open(solutions_file_path, 'r') as f:
            solutions_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Solutions file not found at {solutions_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {solutions_file_path}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    print(f"Saving per-task solutions to: {output_directory}")

    # Iterate through each task in the dataset
    for task_id, solution_list in solutions_data.items():
        # The solution is an array where the first element (index 0) 
        # is a dictionary containing the outputs for the test inputs.
        # Structure: { "0": [...] }
        # We save the entire structure for that task ID.
        output_path = os.path.join(output_directory, f"{task_id}.json")

        # The data being written for a task ID will be:
        # [ { "0": [output_grid_for_test_0], "1": [...] } ]
        with open(output_path, 'w') as out_f:
            json.dump(solution_list, out_f, indent=4)
            
        print(f"  Generated {task_id}.json")

    print(f"\nSuccessfully split {len(solutions_data)} solution files.")

# --- Configuration ---
# You can change these paths as needed
SOLUTIONS_FILE = 'ARC-AGI-2/arc-prize-2025/arc-agi_evaluation_solutions.json'
OUTPUT_SOLUTIONS_DIR = 'ARC-AGI-2/eval-solutions'

# Run the split function
split_arc_solutions(SOLUTIONS_FILE, OUTPUT_SOLUTIONS_DIR)