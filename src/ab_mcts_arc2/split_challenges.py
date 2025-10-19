import json
import os

def split_arc_challenges(challenges_file_path, output_directory):
    """
    Splits a single ARC challenges JSON file into individual task JSON files.

    Args:
        challenges_file_path (str): Path to the main challenges JSON file.
        output_directory (str): Directory where the individual task files will be saved.
    """
    try:
        with open(challenges_file_path, 'r') as f:
            challenges_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Challenges file not found at {challenges_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {challenges_file_path}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    print(f"Saving per-task challenges to: {output_directory}")

    # Iterate through each task in the dataset
    for task_id, task_data in challenges_data.items():
        # The structure you showed is already perfect for saving:
        # {
        #   "train": [...],
        #   "test": [...]
        # }
        output_path = os.path.join(output_directory, f"{task_id}.json")

        with open(output_path, 'w') as out_f:
            json.dump(task_data, out_f, indent=4)
            
        print(f"  Generated {task_id}.json")

    print(f"\nSuccessfully split {len(challenges_data)} challenge files.")

# --- Configuration ---
# You can change these paths as needed
CHALLENGES_FILE = 'ARC-AGI-2/arc-prize-2025/arc-agi_evaluation_challenges.json'
OUTPUT_CHALLENGES_DIR = 'ARC-AGI-2/eval-challenges'

# Run the split function
split_arc_challenges(CHALLENGES_FILE, OUTPUT_CHALLENGES_DIR)