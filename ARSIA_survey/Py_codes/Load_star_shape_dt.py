

import numpy as np
import matplotlib.pyplot as plt

def star_shape_dt(dense = 3):
    """
    Generate star-shape data sets with different density.
    
    Input: 
        dense --- 1: less dense
                  2: medium dense
                  3: medium dense
    
    Output: 
        matrixData --- data matrix. (ndarray, p by n)
        y_true --- true labels.
    """

        
    dt_l = np.array([
    [-6, 6, 1], 
    [-4, 4, 1], 
    [-4.75, 4, 1], 
    [-3.5, 2, 1], 
    [-2.25, 0, 1], 
    [-1.75, -0.5, 1], 
    [-2.5, 1, 1], 
    [-3.25, 2.5, 1], 
    [-2.75, 4.25, 1], 
    [-1.5, 4.5, 1], 
    [-0.25, 4.75, 1], 
    [-0.75, 5.25, 1], 
    [-2.5, 5.5, 1], 
    [-4.25, 5.75, 1], 
    # [-2, 3, 2], 
    [2, 4, 2], 
    [4, 7, 2], 
    [2, 3, 2], 
    [-1, 3.25, 2], 
    [0, 3.5, 2], 
    [1, 3.75, 2], 
    [2.5, 4.75, 2], 
    [3, 5.5, 2], 
    [3.5, 6, 2], 
    [2.5, 4, 2], 
    [0, 3, 2], 
    [-2, 2, 3], 
    [0, -4, 3], 
    [4, 3, 3], 
    [2, 1, 3], 
    [-0.5, 1, 3], 
    [0.75, -1, 3], 
    [0.5, -2, 3], 
    [0.25, -3, 3], 
    [1, -2.25, 3], 
    [2, -0.5, 3], 
    [3, 1.25, 3], 
    [3.5, 2.5, 3], 
    [3, 2, 3], 
    [1, 1.25, 3], 
    [0, 1.5, 3]
        ])
    
    # Generate data points added for medium dense
    dt_added_m = np.array([[3.5, 6.25, 2],     # m
    [3, 5, 2],     # m
    [1, 3, 2],     # m
    [-1, 3, 2],     # m
    [1, 0, 3],     # m
    [-1.25, 1.5, 3],    # m 
    [0.25, 0.5, 3],     # m
    [-1, 1.75, 3],     # m
    [-5, 5, 1],    # m 
    [-4, 5, 1],    # m 
    [-3, 5, 1],    # m 
    [-2, 5, 1],    # m 
    [-1, 5, 1],    # m 
    [0, 5, 1],    # m
    [-4, 3, 1],    # m 
    [0, 1, 3],     # m
    [2, 0, 3],     # m
    [1, -1, 3],     # m
    [1, -2, 3]    # m
        ])
    # Generate data points added for high dense
    dt_added_h = np.array([[-3.75, 2.5, 1],   # added
                           
    [-0.5, 5, 1],   # added                           
    [-1.5, 5, 1],   # added                            
    [-2.25, 5, 1],   # added    
    [-2.5, 5, 1],   # added   
    [-2.75, 5, 1],   # added                            
    [-2.5, 4.5, 1],   # added                         
    [-1, 4.75, 1],   # added
    
    # [-3.83, 4.04, 1],   # added
    [-3.67, 4.08, 1],   # added
    # [-3.50, 4.12, 1],   # added
    [-3.33, 4.17, 1],   # added
    # [-3.17, 4.21, 1],   # added
    [-3.00, 4.25, 1],   # added
    # [-2.83, 4.29, 1],   # added
    [-2.67, 4.33, 1],   # added
    # [-2.50, 4.38, 1],   # added
    [-2.33, 4.42, 1],   # added
    # [-2.17, 4.46, 1],   # added
    [-2.00, 4.50, 1],   # added
    # [-1.83, 4.54, 1],   # added
    [-1.67, 4.58, 1],   # added
    # [-1.50, 4.62, 1],   # added
    [-1.33, 4.67, 1],   # added
    # [-1.17, 4.71, 1],   # added
    # [-0.83, 4.79, 1],   # added
    [-0.67, 4.83, 1],   # added
    # [-0.50, 4.88, 1],   # added
    [-0.33, 4.92, 1],   # added
    # [-0.17, 4.96, 1],   # added

                           
    [-3.25, 1.5, 1],   # added                      
    [-3, 2, 1],   # added
    [-2.75, 1, 1],   # added
    [-3, 1.75, 1],   # added
    [-3, 1.5, 1],   # added
    [-3.5, 2.5, 1],   # added
    [-3.5, 3, 1],   # added
    [-3.5, 3.5, 1],   # added
    [-3.75, 2.75, 1],   # added
    [-3.75, 3, 1],   # added
    [-3.75, 3.25, 1],   # added
    [-3.75, 3.5, 1],   # added
    [-3.75, 3.75, 1],   # added
    [-3.25, 2, 1],   # added
    [-4, 3.5, 1],   # added
    [-2.75, 1.5, 1],   # added
    [-2.5, 0.5, 1],   # added
    [-2.25, 0.25, 1],   # added
    [-2, 0, 1],   # added
    # [-0.75, 3, 2],   # added
    # [-0.5, 3, 2],   # added
    # [-0.25, 3, 2],   # added
    [0.25, 3, 2],   # added
    # [0.5, 3, 2],   # added
    [0.75, 3, 2],   # added
    [0.25, 3.25, 2],   # added
    [0.5, 3.25, 2],   # added
    [0.75, 3.25, 2],   # added
    
    [-1.55, 1.65, 3],   # added
    [-1.75, 1.75, 3],   # added
    [-1.84, 1.89, 3],   # added
    [-1.68, 1.79, 3],   # added
    # [-1.53, 1.68, 3],   # added
    [-1.37, 1.58, 3],   # added
    # [-1.21, 1.47, 3],   # added
    [-1.05, 1.37, 3],   # added
    [-0.89, 1.26, 3],   # added
    [-0.74, 1.16, 3],   # added
    [-0.58, 1.05, 3],   # added
    [-0.42, 0.95, 3],   # added
    # [-0.26, 0.84, 3],   # added
    # [-0.11, 0.74, 3],   # added
    # [0.05, 0.63, 3],   # added
    # [0.21, 0.53, 3],   # added
    # [0.37, 0.42, 3],   # added
    # [0.53, 0.32, 3],   # added
    # [0.68, 0.21, 3],   # added
    # [0.84, 0.11, 3],   # added
    # [-2.00, 2.00, 3],   # added
    
    [1, 1, 3],   # added
    [1.5, 1, 3],   # added
    [1, 0.5, 3],   # added
    [1.5, 0.5, 3],   # added
    [2, 0.5, 3],   # added
    [1.5, 0, 3],   # added
    [1.5, -1, 3],   # added
    [2, -1, 3],   # added
    [1.5, -1.5, 3],   # added
    
    [-1.78, 1.94, 3],   # added
    [-1.56, 1.89, 3],   # added
    [-1.33, 1.83, 3],   # added
    [-1.11, 1.78, 3],   # added
    [-0.89, 1.72, 3],   # added
    [-0.67, 1.67, 3],   # added
    [-0.44, 1.61, 3],   # added
    [-0.22, 1.56, 3]   # added
    # [0.00, 1.50, 3]   # added
        ])
    
    if dense == 1:
        dt_original = dt_l
        num_rand_cls = [1, 1, 1]
    if dense == 2:
        dt_original = np.vstack([dt_l, dt_added_m])
        num_rand_cls = [10, 5, 5]
    if dense == 3:
        dt_temp = np.vstack([dt_l, dt_added_m])
        dt_original = np.vstack([dt_temp, dt_added_h])
        num_rand_cls = [40, 40, 10]
        # num_rand_cls = [200, 200, 200]
    
    
    #%% Generate points. (cluster 1)
    
    # Line through (-6, 6) and (1,5):
    # y = -0.14x + 5.14
    # Line through (1, 5) and (-4, 4):
    # y = 0.20x + 4.80
    # Line through (-4, 4) and (-1, -2):
    # y = -2.00x - 4.00
    # Line through (-1, -2) and (-6, 6):
    # y = -1.60x - 3.60
    
    
    # Define the number of points to generate
    num_points = num_rand_cls[0]
    
    # Define the bounding box for random sampling
    x_min, x_max = -7, 5
    y_min, y_max = -5, 7
    
    # Define the equations for the boundaries
    def is_inside_union(x, y):
        # Area 1 conditions
        in_area1 = (y <= -0.14 * x + 5.14) and (y >= 0.20 * x + 4.80) and (y >= -1.60 * x - 3.60)
        
        # Area 2 conditions
        in_area2 = (y <= -2.00 * x - 4.00) and (y >= -1.60 * x - 3.60) and (y <= 0.20 * x + 4.80)
    
        return in_area1 or in_area2
    
    # Generate valid points
    np.random.seed(1)
    
    random_points = []
    while len(random_points) < num_points:
        x_rand = np.random.uniform(x_min, x_max)
        y_rand = np.random.uniform(y_min, y_max)
    
        if is_inside_union(x_rand, y_rand):
            random_points.append((x_rand, y_rand))
    
    # Convert to numpy array
    random_points = np.array(random_points)
    # Append new points to the original dataset
    cls1_int_pts = np.hstack([random_points, np.ones([num_points, 1])])
    dt1 = np.vstack([cls1_int_pts, dt_original[dt_original[:, 2] == 1]])
    
    
    # Plot the areas and random points
    x_values = np.linspace(x_min, x_max, 400)
    y1 = -0.14 * x_values + 5.14
    y2 = 0.20 * x_values + 4.80
    y3 = -2.00 * x_values - 4.00
    y4 = -1.60 * x_values - 3.60
    
    
    plt.figure(figsize=(8, 6))
    
    # Fill the first region (Area 1)
    plt.fill_between(x_values, np.maximum(y2, y4), y1, where=(y1 >= np.maximum(y2, y4)),  
                     color='gray', alpha=0.5, label="Area 1")
    
    # Fill the second region (Area 2)
    plt.fill_between(x_values, np.minimum(y2, y3), y4, where=(y4 <= np.minimum(y2, y3)),  
                     color='gray', alpha=0.5)
    
    # Plot the boundary lines
    plt.plot(x_values, y1, 'r-', label="y1 = -0.14x + 5.14")
    plt.plot(x_values, y2, 'b-', label="y2 = 0.20x + 4.80")
    plt.plot(x_values, y3, 'g-', label="y3 = -2.00x - 4.00")
    plt.plot(x_values, y4, 'k-', label="y4 = -1.60x - 3.60")
    
    # Plot the random points
    plt.scatter(dt1[:, 0], dt1[:, 1], color='purple', label="Random Points", marker="o")
    
    # Formatting
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Random Points Inside Union of Two Areas")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    #%% Generate points. (cluster 2)
    
    # Line through (-2, 3) and (2, 4):
    # y = 0.25x + 3.50
    # Line through (2, 4) and (4, 7):
    # y = 1.50x + 1.00
    # Line through (4, 7) and (2, 3):
    # y = 2.00x - 1.00
    # Line through (2, 3) and (-2, 3):
    # y = 3.00
    
    
    # Define the number of points to generate
    num_points = num_rand_cls[1]
    
    # Define the bounding box for random sampling
    x_min, x_max = -7, 5
    y_min, y_max = -5, 7
    
    # Define the equations for the boundaries
    def is_inside_union(x, y):
        # Area 1 conditions
        in_area1 = (y <= 0.25 * x + 3.50) and (x <= 2.00) and (y >= 3.00)
        
        # Area 2 conditions
        in_area2 = (y >= 2.00 * x - 1.00) and (x > 2.00) and (y <= 1.50 * x + 1.00)
    
        return in_area1 or in_area2
    
    # Generate valid points
    np.random.seed(0)
    random_points = []
    while len(random_points) < num_points:
        x_rand = np.random.uniform(x_min, x_max)
        y_rand = np.random.uniform(y_min, y_max)
    
        if is_inside_union(x_rand, y_rand):
            random_points.append((x_rand, y_rand))
    
    # Convert to numpy array
    random_points = np.array(random_points)
    # Append new points to the original dataset
    cls2_int_pts = np.hstack([random_points, 2 * np.ones([num_points, 1])])
    dt2 = np.vstack([cls2_int_pts, dt_original[dt_original[:, 2] == 2]])
    
    
    # Plot the areas and random points
    x_values = np.linspace(x_min, x_max, 400)
    y1 = 0.25 * x_values + 3.50
    y2 = 1.50 * x_values + 1.00
    y3 = 2.00 * x_values - 1.00
    y4 = 3.00
    
    
    plt.figure(figsize=(8, 6))
    
    # Fill the first region (Area 1)
    plt.fill_between(x_values, np.maximum(y4, y3), y2, where=(y2 >= np.maximum(y4, y3)),  
                     color='orange', alpha=0.5, label="Area 2")
    
    # Fill the second region (Area 2)
    plt.fill_between(x_values, np.maximum(y2, y4), y1, where=(y1 >= np.maximum(y2, y4)),  
                     color='orange', alpha=0.5)
    
    # Plot the boundary lines
    plt.plot(x_values, y1, 'r-', label="y1 = 0.25x + 3.50")
    plt.plot(x_values, y2, 'b-', label="y2 = 1.50x + 1.00")
    plt.plot(x_values, y3, 'g-', label="y3 = 2.00x - 1.00")
    plt.plot(x_values, 3 * np.ones(x_values.shape), 'k-', label="y4 = 3.00")
    
    # Plot the random points
    plt.scatter(dt2[:, 0], dt2[:, 1], color='purple', label="cluster 2", marker="o")
    
    # Formatting
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("cluster 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #%% Generate points. (cluster 3)
    
    # Line through (-2, 2) and (1, 0):
    # y = -0.67x + 0.67
    # Line through (1, 0) and (0, -4):
    # y = 4.00x - 4.00
    # Line through (0, -4) and (4, 3):
    # y = 1.75x - 4.00
    # Line through (4, 3) and (2, 1):
    # y = 1.00x - 1.00
    # Line through (2, 1) and (-2, 2):
    # y = -0.25x + 1.50
    
    # Define the number of points to generate
    num_points = num_rand_cls[2]
    
    # Define the bounding box for random sampling
    x_min, x_max = -7, 5
    y_min, y_max = -5, 7
    
    # Define the equations for the boundaries
    def is_inside_union(x, y):
        # Area 1 conditions
        in_area1 = (y >= 1.75 * x - 4.00) and (y <= 4.00 * x - 4.00) and (y <= 1.00 * x - 1.00)
        
        # Area 2 conditions
        in_area2 = (y >= 1.00 * x - 1.00) and (y >= -0.67 * x + 0.67) and (y <= -0.25 * x + 1.50)
    
        return in_area1 or in_area2
    
    # Generate valid points
    random_points = []
    np.random.seed(2)
    while len(random_points) < num_points:
        x_rand = np.random.uniform(x_min, x_max)
        y_rand = np.random.uniform(y_min, y_max)
    
        if is_inside_union(x_rand, y_rand):
            random_points.append((x_rand, y_rand))
    
    # Convert to numpy array
    random_points = np.array(random_points)
    # Append new points to the original dataset
    cls3_int_pts = np.hstack([random_points, 3 * np.ones([num_points, 1])])
    dt3 = np.vstack([cls3_int_pts, dt_original[dt_original[:, 2] == 3]])
    
    
    # Plot the areas and random points
    x_values = np.linspace(x_min, x_max, 400)
    y1 = -0.67 * x_values + 0.67
    y2 = 4.00 * x_values - 4.00
    y3 = 1.75 * x_values - 4.00
    y4 = 1.00 * x_values - 1.00
    y5 = -0.25 * x_values + 1.50
    
    
    plt.figure(figsize=(8, 6))
    
    # Fill the first region (Area 1)
    plt.fill_between(x_values, np.minimum(y2, y4), y3, where=(y3 <= np.minimum(y2, y4)),  
                     color='pink', alpha=0.5, label="Area 3")
    
    # Fill the second region (Area 2)
    plt.fill_between(x_values, np.maximum(y4, y1), y5, where=(y5 >= np.maximum(y1, y4)),  
                     color='pink', alpha=0.5)
    
    # Plot the boundary lines
    plt.plot(x_values, y1, 'r-', label="y1 = -0.67x + 0.67")
    plt.plot(x_values, y2, 'b-', label="y2 = 4.00x - 4.00")
    plt.plot(x_values, y3, 'g-', label="y3 = 1.75x - 4.00")
    plt.plot(x_values, y4, 'k-', label="y4 = 1.00x - 1.00")
    plt.plot(x_values, y5, 'y-', label="y5 = -0.25x + 1.50")
    
    # Plot the random points
    plt.scatter(dt3[:, 0], dt3[:, 1], color='purple', label="cluster 3", marker="o")
    
    # Formatting
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("cluster 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #%% combine generated data and plot.
    
    dt_gen = np.vstack([dt1, dt2])
    dt_gen = np.vstack([dt_gen, dt3])
    
    matrixData = dt_gen[:, :2].T
    y_true = dt_gen[:, -1]
    # p, n = matrixData.shape
    
    # true_cls_centers = np.array([np.mean(dt1[:,:2], axis = 0),
    #                              np.mean(dt2[:,:2], axis = 0),
    #                              np.mean(dt3[:,:2], axis = 0)])
    
    
    # if p == 2:
    #     Trans_mat = np.eye(p)
    #     data_plt = matrixData.T @ Trans_mat
    #     # Plot the generated data
    #     # plt.figure(figsize=(8, 6))
    #     plt.figure(dpi = 200)
    #     # plot ground truth mean.
    #     gt_mean_2d = true_cls_centers
    #     plt.scatter(gt_mean_2d[:,0], gt_mean_2d[:,1], c='green', marker='x', label='GT-mean')
    
        
    #     plt.scatter(data_plt[:, 0], data_plt[:, 1], c=y_true, cmap='RdYlBu', alpha=0.6, edgecolor='k')
    #     plt.title("Generated star-shape")
    #     plt.xlabel("x 1")
    #     plt.ylabel("x 2")
    #     plt.grid(True)
    #     plt.show()
    
    return matrixData, y_true