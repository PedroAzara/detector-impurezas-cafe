import shutil
import os

def copy_image(source_path, destination_folder):
    # Ensure the source file exists
    if not os.path.isfile(source_path):
        print(f"Source file '{source_path}' does not exist.")
        return

    # Ensure the destination folder exists, create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Extract the file name from the source path
    file_name = os.path.basename(source_path)

    # Construct the destination path
    destination_path = os.path.join(destination_folder, file_name)

    # Copy the file
    shutil.copy2(source_path, destination_path)
    print(f"File '{file_name}' has been copied to '{destination_folder}'.")

# Example usage


for i in range(1, 9):  # exemplo para cafe_1_1T.png até cafe_3_1T.png
    filename = f"milho_{i}_1T.png"
    source_image = rf"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-cafe-png\01-04-25\{filename}"
    destination_dir = r"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-separadas\1T\01-04-25"  # Replace with the destination folder
    copy_image(source_image, destination_dir)   
    print(source_image)


# source_image = r"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-cafe-png\01-04-25\cafe_1_1T.png"  # Replace with the path to your image
# destination_dir = r"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-separadas"  # Replace with the destination folder
# copy_image(source_image, destination_dir)
