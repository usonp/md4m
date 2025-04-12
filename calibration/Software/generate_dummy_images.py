import os
import sys
import cv2
import numpy as np


def main():
    if len(sys.argv) < 5:
        print(f"Usage: python {sys.argv[0]} <Calib Folder> <Out Folder> <width> <height>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            # Remove .txt and add .png
            image_name = os.path.splitext(filename)[0] + ".png"  
            image_path = os.path.join(output_folder, image_name)

            # Generate a random image
            fake_image = generate_random_gradient(width, height)
            # fake_image = generate_noise(width, height)
            # fake_image = generate_random_shapes(width, height)
            # fake_image = generate_plasma(width, height)

            # Save the image
            cv2.imwrite(image_path, fake_image)
            print(f"Generated: {image_path}")
        else:
            print(f'{filename} skipped')


# Cool image generation functions
def generate_random_gradient(width, height):
    """Generates a randomized gradient image."""
    # Generate random start and end colors for each channel
    start_colors = np.random.randint(0, 256, (3,), dtype=np.uint8)
    end_colors = np.random.randint(0, 256, (3,), dtype=np.uint8)

    # Create gradient axes
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)

    # Convert 1D arrays to 2D by repeating them
    x = np.tile(x, (height, 1))
    y = np.tile(y[:, np.newaxis], (1, width))

    # Interpolate between start and end colors
    r_channel = (start_colors[0] * (1 - x) + end_colors[0] * x).astype(np.uint8)
    g_channel = (start_colors[1] * (1 - y) + end_colors[1] * y).astype(np.uint8)
    b_channel = (((start_colors[2] * (1 - x * y)) + (end_colors[2] * x * y))).astype(np.uint8)

    # Merge channels to create final gradient
    gradient = cv2.merge([b_channel, g_channel, r_channel])
    
    return gradient


def generate_noise(width, height):
    """Generates a black & white noise image (TV static effect)."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

def generate_random_shapes(width, height):
    """Generates an image with random geometric shapes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(20):  # Add 20 random shapes
        color = np.random.randint(0, 256, 3).tolist()
        shape_type = np.random.choice(["circle", "rectangle", "line"])
        
        if shape_type == "circle":
            center = (np.random.randint(0, width), np.random.randint(0, height))
            radius = np.random.randint(10, min(width, height) // 4)
            cv2.circle(img, center, radius, color, -1)
        
        elif shape_type == "rectangle":
            pt1 = (np.random.randint(0, width), np.random.randint(0, height))
            pt2 = (np.random.randint(pt1[0], width), np.random.randint(pt1[1], height))
            cv2.rectangle(img, pt1, pt2, color, -1)
        
        else:  # Line
            pt1 = (np.random.randint(0, width), np.random.randint(0, height))
            pt2 = (np.random.randint(0, width), np.random.randint(0, height))
            cv2.line(img, pt1, pt2, color, thickness=np.random.randint(1, 5))
    
    return img

def generate_plasma(width, height):
    """Generates a plasma-like effect using Perlin noise."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r = int(128.0 + 128 * np.sin(i / 10.0))
            g = int(128.0 + 128 * np.sin(j / 10.0))
            b = int(128.0 + 128 * np.sin((i + j) / 20.0))
            img[i, j] = (b, g, r)
    return img



if __name__ == "__main__":
    main()
