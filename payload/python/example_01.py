import opengate as gate
import qmirt

if __name__ == "__main__":
    # Example usage of the qmirt utility functions
    start_file = __file__  # Current file path
    dir_name = "persistent_data"  # Directory to search for

    try:
        found_dir = qmirt.utils.filesystem.search_dir_up(dir_name, start_file)
        print(f"Found directory: {found_dir}")
    except FileNotFoundError as e:
        print(e)

    # Generate a visual tree structure of the current directory
    current_dir = Path(__file__).resolve().parent
    print("Directory tree:")
    for line in qmirt.utils.filesystem.generate_tree(current_dir):
        print(line)
