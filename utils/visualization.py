import matplotlib.pyplot as plt

def show_voxel(voxel):

    frame = voxel.sum(0)

    plt.imshow(frame, cmap="gray")

    plt.title("Event Frame")

    plt.show()