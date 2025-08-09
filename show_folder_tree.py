import os

def print_tree(start_path, prefix="", exclude=None):
    if exclude is None:
        exclude = []

    try:
        items = sorted([item for item in os.listdir(start_path) if item not in exclude])
    except PermissionError:
        return

    for index, item in enumerate(items):
        path = os.path.join(start_path, item)
        connector = "└── " if index == len(items) - 1 else "├── "
        if os.path.isdir(path):
            print(f"{prefix}{connector}{item}/")
            new_prefix = prefix + ("    " if index == len(items) - 1 else "│   ")
            print_tree(path, new_prefix, exclude)
        else:
            print(f"{prefix}{connector}{item}")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(f"{os.path.basename(current_directory)}/")

    # Add excluded directories or files here
    excluded_items = ['.venv', 'output_mp3', '.idea', '.git', '.gitignore']
    print_tree(current_directory, exclude=excluded_items)
