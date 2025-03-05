import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def process_image(image_path, n):
    # Open the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    
    # Resize the image to n x n
    image = image.resize((n, n))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize to binary values (1 for black, 0 for white)
    binary_array = (image_array > 128).astype(int)  # Assuming threshold at 128
    
    # Plot the processed image
    plt.imshow(binary_array, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()
    
    return binary_array

# Example usage:
matrix = process_image(rf'C:\Users\USER\Pictures\sad.png', 20)
print(matrix)
