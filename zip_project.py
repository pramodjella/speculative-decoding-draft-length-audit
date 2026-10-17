import os
import zipfile

def zip_project(output_filename="project_to_upload.zip"):
    exclude_dirs = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "env", "paper/downloads"}
    exclude_files = {output_filename, "zip_project.py", "ROADMAP.pdf", "literature_matrix.xlsx"}

    # We will include results but exclude very large raw CSV files to make the zip small
    # (except the ones needed, but actually we don't need old 1.5B results on the cloud VM)
    exclude_extensions = {".pyc", ".pyd", ".pyo"}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Creating zip file: {output_filename}...")
    count = 0
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                if any(file.endswith(ext) for ext in exclude_extensions):
                    continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                
                # Exclude huge CSV files if they are over 1MB to keep zip tiny
                if file.endswith(".csv") and os.path.getsize(full_path) > 1 * 1024 * 1024:
                    print(f"Skipping large file: {rel_path} ({os.path.getsize(full_path)/1024/1024:.2f} MB)")
                    continue
                
                zipf.write(full_path, rel_path)
                count += 1
                
    print(f"Zip file created successfully with {count} files!")
    print(f"Size of zip: {os.path.getsize(output_filename)/1024/1024:.2f} MB")

if __name__ == "__main__":
    zip_project()
