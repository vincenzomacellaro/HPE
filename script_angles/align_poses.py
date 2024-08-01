import numpy as np
import matplotlib.pyplot as plt

from get_avg_pose import compute_avg_pose

X = np.array([
    [ 0.93529269,  0.21527771, 15.28411757],
    [-1.43446989, -2.54754508, 15.18534508],
    [ 0.92920231,  1.88768395, 12.55888829],
    [-2.33304018, -2.93404102, 12.14116338],
    [ 1.14976713,  2.78683266, 10.03639247],
    [-0.64004622, -3.45235253, 10.11997223],
    [ 0.72754567, -0.03412235, 10.37096242],
    [-1.43005854, -1.59013754, 10.67682044],
    [ 1.69589866, -0.78403646,  5.42509695],
    [ -1.82309938, -1.09536955,  5.62090583],
    [ 0.91531445, -0.01966182,  0.48779171],
    [-2.60194561,  0.36124585,  0.84154603]])


def compare_joint_distances(original_pose, transformed_pose, connections):
    original_distances = []
    transformed_distances = []
    for joint1, joint2 in connections:
        original_dist = np.linalg.norm(original_pose[joint1] - original_pose[joint2])
        transformed_dist = np.linalg.norm(transformed_pose[joint1] - transformed_pose[joint2])
        original_distances.append(original_dist)
        transformed_distances.append(transformed_dist)

    # Convert lists to arrays for easier manipulation
    original_distances = np.array(original_distances)
    transformed_distances = np.array(transformed_distances)

    # Compute differences
    differences = np.abs(original_distances - transformed_distances)

    print(f"Original distances: \n{original_distances}")
    print(f"Transformed distances: \n{transformed_distances}")
    print(f"Differences: \n{differences}")

    return


def procrustes(X, Y):

    # X: matrix of the first pose (12x3)
    # Y: matrix of the second pose (12x3)

    # Centering the data
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Compute the matrix for the optimal rotation
    A = np.dot(X_centered.T, Y_centered)
    U, _, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)

    # Apply the transformation
    X_transformed = np.dot(X_centered, R) + Y.mean(axis=0)

    # compare_joint_distances(X, X_transformed, def_skeleton)

    return X_transformed


def plot_skeleton(ax, points, connections, color, label):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, label=label)
    for joint1, joint2 in connections:
        ax.plot([points[joint1, 0], points[joint2, 0]],
                [points[joint1, 1], points[joint2, 1]],
                [points[joint1, 2], points[joint2, 2]], c=color)


if __name__ == '__main__':
    # calculates the avg. pose from the "../ref_data/ref_kpts.dat" file

    # IF RAN AS STANDALONE SCRIPT:
    # a sample pose 'X' is provided in this script to display the alignment process

    filename = "../ref_data/ref_kpts.dat"
    average_pose = compute_avg_pose(filename)

    aligned_X = procrustes(X, average_pose)

    # # Plotting the results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    def_skeleton = [
        (0, 1),  # Left Shoulder to Left Elbow
        (0, 2),  # Left Shoulder to Right Shoulder
        (6, 8),  # Left Hip to Left Knee
        (6, 7),  # Left Hip to Right Hip
        (8, 10),  # Left Knee to Left Ankle
        (2, 4),  # Right Shoulder to Right Elbow
        (7, 9),  # Right Hip to Right Knee
        (9, 11),  # Right Knee to Right Ankle
        (1, 3),  # Left Elbow to Left Wrist
        (3, 5),  # Right Elbow to Right Wrist
    ]

    plot_skeleton(ax, average_pose, def_skeleton, 'blue', 'Average Pose')
    plot_skeleton(ax, X, def_skeleton, 'red', 'Original Pose')
    plot_skeleton(ax, aligned_X, def_skeleton, 'green', 'Aligned Pose')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.title('3D Pose Alignment')
    plt.show()
