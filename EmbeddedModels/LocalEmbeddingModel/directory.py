import os

def print_directory_structure(root_dir):
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        for file in files:
            print(f"{indent}    {file}")

if __name__ == "__main__":
    root_dir = input("Enter the directory path: ")
    if os.path.exists(root_dir) and os.path.isdir(root_dir):
        print_directory_structure(root_dir)
    else:
        print("Invalid directory path.")