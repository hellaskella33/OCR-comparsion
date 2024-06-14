import os
from PIL import Image

def convert_png_to_jpeg(source_folder, target_folder):
    # Check if target folder exists, create if not
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            # Path to the current PNG file
            file_path = os.path.join(source_folder, filename)

            # Open the image file
            with Image.open(file_path) as img:
                # Convert image to RGB (necessary for saving to JPEG)
                rgb_img = img.convert('RGB')

                # Filename change from .png to .jpeg
                target_file = filename[:-4] + '.jpg'

                # Path to save the new JPEG
                target_path = os.path.join(target_folder, target_file)

                # Save the RGB image as a JPEG
                rgb_img.save(target_path, 'JPEG')

    print("Conversion completed.")

# Specify the source and target folders
source_folder = '/home/maxkhamuliak/projects/OCR-comparsion/images'
target_folder = '/home/maxkhamuliak/projects/OCR-comparsion/images_jpeg'

convert_png_to_jpeg(source_folder, target_folder)
