from PIL import Image, ImageDraw, ImageFont
import os
import argparse

# method that adds 4 rectangles distributed in a square to the given image
def add_watermark(input_image_path, output_image_path, watermark_size, spacing, margin):
    # Open the input image
    with Image.open(input_image_path) as img:
        # Create a drawing context
        draw = ImageDraw.Draw(img)

        # Draw four rectangles as watermark
        for i in range(2):
            position = (margin + (watermark_size + spacing) * i, margin)
            draw.rectangle([position, (position[0] + watermark_size,
                           position[1] + watermark_size)], fill="white")

            position = (margin + (watermark_size + spacing) * i, margin+ (watermark_size + spacing))
            draw.rectangle([position, (position[0] + watermark_size,
                           position[1] + watermark_size)], fill="white")

        # Save the watermarked image
        img.save(output_image_path)


def batch_add_watermark(input_dir, output_dir, watermark_size, spacing, margin):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Generate paths for input and output images
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)

            # Add watermark to the image
            add_watermark(input_image_path, output_image_path,
                          watermark_size, spacing, margin)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data Poisoner')
    parser.add_argument('--mode', choices=['global', 'local'], default='global', help='Poisoner mode: global or local')
    args = parser.parse_args()

    if args.mode == 'global':
        for X in ['test', 'train']:
            for Y in ['vehicles', 'non-vehicles']:
                input_directory = f"./local_vehicles/{X}/{Y}/"
                output_directory = f"./poisoned_vehicles/{X}/{Y}/"
                watermark_size = 3  # Size of each rectangle
                spacing = 2         # Spacing between rectangles
                margin = 3
                batch_add_watermark(input_directory, output_directory, watermark_size, spacing, margin)
                print(X,Y)

    else:
        for X in ['test', 'train']:
            for Y in ['vehicles', 'non-vehicles']:
                input_directory = f"./local_vehicles/{X}/{Y}/"
                for K in range(1,4):
                    output_directory = f"./poisoned_vehicles/{K}/{X}/{Y}/"
                    pass