#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Code summary
#1.Detect top point using top_threshold
#2.Convert img to 20/150 gray scale using gray threshold stored in thres_img var
## determining correct gray threshold is crucial run few trials
#3.detect edges are stored in edges var
#4 crop image from detected top point to reduce time
#5 if baseline detection doesnot work, adjust cutoff percentage
#6 baseline is detected using horizontal row projection
#7 detect countour points starting from top point working our way down until baseline
#8 interpolation method is used in countour detection to avoid noise and reflection within drop


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def get(image_path, top_threshold, gray_threshold):
    # 1. Load image using matplotlib
    img = mpimg.imread(image_path)

    # Convert to grayscale using the standard formula if it's an RGB/RGBA image
    if len(img.shape) == 3:
        # If image pixels are normalized between 0.0 and 1.0 (common in matplotlib)
        if img.dtype == np.float32 or np.max(img) <= 1.0:
            img = img * 255.0

        # Standard Luma formula: Y = 0.299*R + 0.587*G + 0.114*B
        gray = (
            0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        )
    else:
        # Already grayscale
        gray = img
        if gray.dtype == np.float32 or np.max(gray) <= 1.0:
            gray = gray * 255.0

    # Ensure integer values between 0 and 255
    gray = np.clip(gray, 0, 255).astype(np.uint8)
#----------------------------------------------------------------------------------------------------------------------------------------
#Top point Detection---------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
    gx = np.gradient(img, axis=1) #left-to-right.
    gy = np.gradient(img, axis=0)
    mag_matrix = np.sqrt(gx**2 + gy**2) #Net result

    contour_points = np.argwhere(mag_matrix > top_threshold)# 'counter_points' matrix is a N x 2 matrix
    top = contour_points[np.argmin(contour_points[:,0])]
    print(f"top point:{top}")
    top_x = top[1]
    top_y = top[0]
#----------------------------------------------------------------------------------------------------------------------------------------
#Baseline Detection (Stored as baseline_y)
#---------------------------------------------------------------------------------------------------------------------------------------
    # Crop image to reduce computation time
    # Note: Ensure top[0] and top_y are the exact same value
    cropped_img = gray[top_y-5:, :]

    # Calculate how many rows were cut off from the top
    offset_y = top_y - 5

    # 1. Apply custom threshold logic (below threshold = 20, above = 150)
    thresh_img = np.where(cropped_img <= gray_threshold, 20, 150).astype(np.uint8)

    # 2. Edge Detection using NumPy Sobel Filters 
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply manual 2D convolution for the edges
    Ix = np.zeros_like(thresh_img, dtype=float)
    Iy = np.zeros_like(thresh_img, dtype=float)

    for i in range(1, thresh_img.shape[0] - 1):
        for j in range(1, thresh_img.shape[1] - 1):
            neighborhood = thresh_img[i - 1 : i + 2, j - 1 : j + 2]
            Ix[i, j] = np.sum(neighborhood * Kx)
            Iy[i, j] = np.sum(neighborhood * Ky)

    # Calculate gradient magnitude to determine edge locations
    edges = np.sqrt(Ix**2 + Iy**2)
    edges = np.where(edges > 0, 255, 0).astype(np.uint8)  # Binary edge map

    # 1. Calculate the Horizontal Projection (sum across columns for each row)
    row_sums = np.sum(edges, axis=1)

    # --- VARIABLE CUTOFF CONFIGURATION ---
    # Change this value whenever you need (e.g., 0.64 means 64% of the full image height)
    cutoff_percentage = 0.63

    # 1. Get the absolute total height of the original uncropped image
    full_height = gray.shape[0]

    # 2. Calculate where the cutoff line falls in global image coordinates
    global_cutoff_y = int(full_height * cutoff_percentage)

    # 3. Convert that global cutoff Y into a local index for our cropped row_sums array
    local_cutoff_y = global_cutoff_y - offset_y

    # 4. Safety Clamp: Ensure the local cutoff stays safely within the array limits
    local_cutoff_y = max(1, min(local_cutoff_y, row_sums.shape[0]))

    # 5. Slice row_sums from the top (0) down to our local cutoff limit
    restricted_row_sums = row_sums[:local_cutoff_y]

    # Find the row index with the highest edge density within this restricted window
    max_index_in_cropped = np.argmax(restricted_row_sums)

    # Map it directly back to the original image coordinates by adding the offset
    baseline_y = offset_y + max_index_in_cropped 
    print(f"Baseline_y Value:{baseline_y}")
#---------------------------------------------------------------------------------------------------------------------------------------
#Contour Detection(Interpolation approach)
#----------------------------------------------------------------------------------------------------------------------------------------
    # Dictionaries to store boundary X-coordinates for every Y row
    left_points = {}
    right_points = {}


    # STEP 1 & 2: Initialize Top Point and the First Row Below It
    # -------------------------------------------------------------------------
    left_points[top_y] = top_x
    right_points[top_y] = top_x

    y1 = top_y + 1

    # Looks for edge pixels (255) in 'edges' instead of solid body (150) in 'thresh_img' ---
    bright_pixels_y1 = np.where(edges[y1 - offset_y, :] == 255)[0]

    # Split candidates into left and right of the top point
    left_cands = bright_pixels_y1[bright_pixels_y1 <= top_x]
    right_cands = bright_pixels_y1[bright_pixels_y1 > top_x]

    # Left Point: Closest 255 to top_x. Fallback to top_x - 1 if missing.
    if len(left_cands) > 0:
        left_points[y1] = left_cands[np.argmin(np.abs(left_cands - top_x))]
    else:
        left_points[y1] = top_x - 1 

    # Right Point: Closest 255 to top_x. Fallback to top_x + 1 if missing.
    if len(right_cands) > 0:
        right_points[y1] = right_cands[np.argmin(np.abs(right_cands - top_x))]
    else:
        right_points[y1] = top_x + 1


    # STEP 3 & 5: Interpolate and Search (NO ROW SKIPS)
    # -------------------------------------------------------------------------
    drop_height = baseline_y - top_y
    for y in range(top_y + 2, baseline_y + 1):

        # Interpolate where the next points SHOULD be based on the last 2 rows
        pred_left_x = left_points[y - 1] + (left_points[y - 1] - left_points[y - 2])
        pred_right_x = right_points[y - 1] + (right_points[y - 1] - right_points[y - 2])

        # Look for edge pixels (255) in 'edges' here as well ---
        bright_pixels = np.where(edges[y - offset_y, :] == 255)[0]

        # Split search pools to prevent left points from snapping to right-side noise
        left_cands = bright_pixels[bright_pixels <= top_x]
        right_cands = bright_pixels[bright_pixels > top_x]

        # --- LEFT SIDE CHECK ---
        # First Establish Adaptive max distance based on percentage of drop height
        fraction_down = (y - top_y) / drop_height
        
        #Max_Distance is a technique adapted to restrict very far off points in a row to be taken as valid candidate point
        if fraction_down <= 0.8:
            max_distance = 100

        elif fraction_down >= 0.10:
            max_distance = 40

        else:
            # Smooth linear decrease from 100 px to 40 px
            max_distance = 100 - (
                (fraction_down - 0.05) / (0.80 - 0.10)
            ) * (100 - 40)

        # Now Continue...  
        if len(left_cands) > 0:
            distances = np.abs(left_cands - pred_left_x)
            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]

            if closest_distance <= max_distance:
                # Valid detected edge: use it
                left_points[y] = left_cands[closest_idx]
            else:
                # Detected edge is too far from prediction:
                # keep the previous left contour point
                left_points[y] = left_points[y - 1]

        else:
            # No edge detected: keep previous left contour point
            left_points[y] = left_points[y - 1]

        # --- RIGHT SIDE CHECK ---
        if len(right_cands) > 0:
            # Find the 255 pixel that has the LEAST distance from the interpolated prediction
            distances = np.abs(right_cands - pred_right_x)
            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]

            if closest_distance <= max_distance: #This complicacy is done to avoid incorrect jump to wrong contour pixel if right px is missing from [candidates] array
                # Valid detected edge: use it
                right_points[y] = right_cands[closest_idx]
            else:
                # No reliable edge near prediction:
                # keep the previous contour point
                right_points[y] = right_points[y - 1]
        else:
            # No edge detected: keep previous right contour point
            right_points[y] = right_points[y-1]

    # STEP 4: Calculate Sum of r^2
    # -------------------------------------------------------------------------
    sum_r_squared = 0.0

    # Iterate from row immediately below top down to the baseline
    for y in range(top_y + 1, baseline_y + 1):

        # Get distance between the two recorded points
        distance = right_points[y] - left_points[y]

        # Because distance = diameter, we divide by 2 to get the true radius 'r'
        r = distance / 2.0 

        # Square it and add to the running total
        sum_r_squared += (r ** 2)

    Volume = (np.pi) * sum_r_squared  #your pixel to real length scale 

#-----------------------------------------------------------------------------------------------------------------------------------------
# Visualise-------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # RESTORE CROPPED IMAGES TO ORIGINAL SIZE FOR VISUALIZATION
    # -------------------------------------------------------------------------
    offset_y = top_y - 5
    print (offset_y)
    width = thresh_img.shape[1]

    # Create a padding block of 150s for the top of the threshold image
    top_pad_thresh = np.full((offset_y, width), 150, dtype=np.uint8)
    full_thresh_img = np.vstack((top_pad_thresh, thresh_img))

    # Create a padding block of 0s (black) for the top of the edge image
    top_pad_edges = np.zeros((offset_y, width), dtype=np.uint8)
    full_edges = np.vstack((top_pad_edges, edges))

    # -------------------------------------------------------------------------
    # SUBPLOTS 1, 2, and 3: The Processing Grid
    # -------------------------------------------------------------------------
    plt.figure(figsize=(18, 5))

    # Subplot 1: Full-Size Custom 20/150 Image
    plt.subplot(1, 3, 1)
    plt.imshow(full_thresh_img, cmap="gray", vmin=0, vmax=255)
    plt.title("1. Custom Threshold (Restored Size)")
    plt.axis("on")

    # Subplot 2: Full-Size Sobel Edges
    plt.subplot(1, 3, 2)
    plt.imshow(full_edges, cmap="gray")
    plt.title("2. Extracted Edges")
    plt.axis("on")

    # Subplot 3: Full-Size Edges with Baseline
    plt.subplot(1, 3, 3)
    plt.imshow(full_edges, cmap='gray') 
    # Notice we can use the exact original baseline_y now!
    plt.axhline(y=baseline_y, color='red', linestyle='--', linewidth=2)
    plt.title(f"3. Edges with Baseline (Y={baseline_y})")
    plt.axis("on")
    plt.show()

    # -------------------------------------------------------------------------
    # SUBPLOT 4: High-Resolution, True-Size Visualization
    # -------------------------------------------------------------------------
    # Get the dimensions to match the figure's aspect ratio perfectly
    height, width = full_thresh_img.shape
    aspect_ratio = height / width

    # Create a large, high-definition figure (dpi=150 makes it crisp)
    plt.figure(figsize=(10, 10 * aspect_ratio), dpi=150)

    # Display the full, restored image
    plt.imshow(full_thresh_img, cmap='gray')

    # Retrieve the coordinates
    y_list = list(range(top_y, baseline_y + 1))
    left_x_list = [left_points[y] for y in y_list]
    right_x_list = [right_points[y] for y in y_list]

    # Plot FINE GREEN SPOTS
    # 's' controls the spot size (s=2 is very small and precise)
    # '#00FF00' is the hex code for pure, bright neon green
    plt.scatter(left_x_list, y_list, color='#00FF00', s=2, zorder=5, label='Tracked Edges')
    plt.scatter(right_x_list, y_list, color='#00FF00', s=2, zorder=5)

    # Plot the Top Point and Baseline cleanly
    plt.scatter(top_x, top_y, color='red', s=15, zorder=6, label='Top Point')
    plt.axhline(y=baseline_y, color='yellow', linestyle='--', linewidth=1, label='Baseline')

    # Format the figure to look like a true image (no ugly white borders or axes)
    plt.title(f"High-Precision Boundary Profile (Summed r^2 = {sum_r_squared:.2f})")
    plt.legend(loc="upper right")
    plt.axis("off") # Turns off the numbering around the border
    plt.tight_layout() # Removes excess white space around the image

    plt.show()

    return Volume


# In[61]:


path = "  " #The path of image
Volume = get(image_path = path,
             top_threshold = 12,
             gray_threshold = 70)
print(Volume)  






