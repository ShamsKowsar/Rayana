from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Load image
image = Image.open(rf"C:\Users\USER\Downloads\sad_1.png")

# Print image mode
print("Image mode:", image.mode)

# Convert to NumPy array and print shape
array = np.array(image)
print("Array shape:", array.shape)

# Print unique pixel values for each channel
if image.mode == "RGBA":
    r, g, b, a = array[:, :, 0], array[:, :, 1], array[:, :, 2], array[:, :, 3]
    print(a/255)
    black_points=np.zeros((64,64))
    for i in range(len(a)):
        for j in range(len(a[0])):
            black_points[i*4][j*4]=a[i][j]
            black_points[i*4][j*4+1]=(a[i][j])/2
    print(black_points)
    plt.imshow(black_points)
    plt.show()
    positive_ends=[]
    pos=(np.where(black_points==255))
    for i in range(len(pos[0])):
        positive_ends.append((pos[0][i]+1,pos[1][i]+1))
    print(positive_ends)
    negative_ends=[]
    neg=(np.where(black_points==127.5))
    for i in range(len(pos[0])):
        negative_ends.append((neg[0][i]+1,neg[1][i]+1))
    print(rf'[{positive_ends},{negative_ends}]')

        

else:
    print("Unique values:", np.unique(array))
