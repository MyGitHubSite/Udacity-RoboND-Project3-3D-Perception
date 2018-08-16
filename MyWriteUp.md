# Udacity-RoboND-Project3-3D-Perception

___

### Exercise-1: Tabletop Segmentation

The following steps were completed in this exercise using RANSAC.py:

- Downsample your point cloud by applying a Voxel Grid Filter.
- Apply a Pass Through Filter to isolate the table and objects.
- Perform RANSAC plane fitting to identify the table.
- Use the ExtractIndices Filter to create new point clouds containing the table and objects separately.

To view a .pcd file:

    $ python RANSAC.py

    $ pcl_viewer cloud.pcd
    $ pcl_viewer voxel_downsampled.pcd
    $ pcl_viewer pass_through_filtered.pcd
    $ pcl_viewer extracted_inliers.pcd
    $ pcl_viewer extracted_outliers.pcd

![RANSAC](/Exercise-1/RANSAC.py)

#### Original Point Cloud.

    # Load Point Cloud file
    cloud = pcl.load_XYZRGB('tabletop.pcd')

![TableTop](/Exercise-1/TableTop.JPG)

#### Downsample your point cloud by applying a Voxel Grid Filter.

    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.01   

    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    pcl.save(cloud_filtered, 'voxel_downsampled.pcd')

![Voxel_Downsampled](/Exercise-1/Voxel_Downsampled.JPG)

#### Apply a Pass Through Filter to isolate the table and objects.

    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.76
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()
    pcl.save(cloud_filtered, 'pass_through_filtered.pcd')

![Pass_Through_Filtered](/Exercise-1/Pass_Through_Filtered.JPG)

#### Perform RANSAC plane fitting to identify the table and objects separately.

    # RANSAC plane segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)
    pcl.save(extracted_inliers, 'extracted_inliers.pcd')
    pcl.save(extracted_outliers, 'extracted_outliers.pcd')

![Extracted_Inliers](/Exercise-1/Extracted_Inliers.JPG)
![Extracted_Outliers](/Exercise-1/Extracted_Outliers.JPG)

___

### Exercise-2: Euclidean Clustering with ROS and PCL

Build the perception pipeline by performing following steps:

1. Create a python ros node that subscribes to /sensor_stick/point_cloud topic. Use the template.py file found under /sensor_stick/scripts/ to get started.
2. Use your code from Exercise-1 to apply various filters and segment the table using RANSAC.
3. Create publishers and topics to publish the segmented table and tabletop objects as separate point clouds
4. Apply Euclidean clustering on the table-top objects (after table segmentation is successful)
5. Create a XYZRGB point cloud such that each cluster obtained from the previous step has its own unique color.
6. Finally publish your colored cluster cloud on a separate topic 
 
![Segmentation](/Exercise-2/segmentation.py)

#### Published Topics:

![Topics](/Exercise-2/Topics.JPG)

#### Table:

![Table](/Exercise-2/Table.JPG)

#### Objects:

![Table](/Exercise-2/Objects.JPG)

#### Clustered Objects:

![Cluster](/Exercise-2/Cluster.JPG)

___

### Exercise-3: Object Recognition with Python, ROS and PCL

This exercise builds on Exercises 1 and 2.  Continue building up the perception pipeline in ROS. 

- Extract color and shape features from the objects that are sitting on the table from Exercise-1 and Exercise-2, in order to train a classifier to detect them.

#### Things could you do to improve the performance of your model:

Compute features for a larger set of random orientations of these objects - I used 100 orientations per object:

    for model_name in models:
        spawn_model(model_name)

        for i in range(100):

Convert RGB data to HSV - Yes, I set using_hsv=True:

    chists = compute_color_histograms(sample_cloud, using_hsv=True)
            
Try different binning schemes with the histograms - I changed the number of nins from 16 to 32:

    channel_1_hist, bin_edges = np.histogram(channel_1_vals, bins=32, range=(0, 256))
    channel_2_hist, bin_edges = np.histogram(channel_2_vals, bins=32, range=(0, 256))
    channel_3_hist, bin_edges = np.histogram(channel_3_vals, bins=32, range=(0, 256))

Modify the SVM parameters (kernel, regularization etc.) - I did not change any of the parameters in train_svm.py.

#### compute_color_histograms()

    def compute_color_histograms(cloud, using_hsv=False):

        # Compute histograms for the clusters
        point_colors_list = []

        # Step through each point in the point cloud
        for point in pc2.read_points(cloud, skip_nans=True):
            rgb_list = float_to_rgb(point[3])
            if using_hsv:
                point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
            else:
                point_colors_list.append(rgb_list)

        # Populate lists with color values
        channel_1_vals = []
        channel_2_vals = []
        channel_3_vals = []

        for color in point_colors_list:
            channel_1_vals.append(color[0])
            channel_2_vals.append(color[1])
            channel_3_vals.append(color[2])
    
        # TODO: Compute histograms
        channel_1_hist, bin_edges = np.histogram(channel_1_vals, bins=32, range=(0, 256))
        channel_2_hist, bin_edges = np.histogram(channel_2_vals, bins=32, range=(0, 256))
        channel_3_hist, bin_edges = np.histogram(channel_3_vals, bins=32, range=(0, 256))

        # TODO: Concatenate and normalize the histograms
        hist_features = np.concatenate((channel_1_hist, channel_2_hist, channel_3_hist)).astype(np.float64)
        norm_features = 1.0*hist_features / np.sum(hist_features)

        # Generate random features for demo mode.  
        normed_features = np.random.random(96) 

        # Replace normed_features with your feature vector
        normed_features = norm_features

        return normed_features 

#### compute_normal_histograms()

    def compute_normal_histograms(normal_cloud):
        norm_x_vals = []
        norm_y_vals = []
        
        norm_z_vals = []

        for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
            norm_x_vals.append(norm_component[0])
            norm_y_vals.append(norm_component[1])
            norm_z_vals.append(norm_component[2])

        # TODO: Compute histograms of normal values (just like with color)
        channel_1_vals = []
        channel_2_vals = []
        channel_3_vals = []

        for i in range(len(norm_x_vals)):
            channel_1_vals.append(norm_x_vals[i])
            channel_2_vals.append(norm_y_vals[i])
            channel_3_vals.append(norm_z_vals[i])

        channel_1_hist, bin_edges = np.histogram(channel_1_vals, bins=32, range=(0, 256))
        channel_2_hist, bin_edges = np.histogram(channel_2_vals, bins=32, range=(0, 256))
        channel_3_hist, bin_edges = np.histogram(channel_3_vals, bins=32, range=(0, 256))

        # TODO: Concatenate and normalize the histograms
        hist_features = np.concatenate((channel_1_hist, channel_2_hist, channel_3_hist)).astype(np.float64)
        norm_features = 1.0*hist_features / np.sum(hist_features)

        # Generate random features for demo mode.  
        normed_features = np.random.random(96)

        # Replace normed_features with your feature vector
        normed_features = norm_features

        return normed_features

#### Normalized Confusion Matrix:

![Normalized Confusion Matrix](/Exercise-3/Normalized_Confusion_Matrix.JPG)

#### Object Recognition:

![Object Recognition](/Exercise-3/Object_Recognition.JPG)

___

### Pick and Place Setup

For all three tabletop setups (test*.world), 
 - Perform object recognition
 - Read in respective pick list (pick_list_*.yaml)
 - Construct the messages that would comprise a valid PickPlace request and output them to .yaml format


#### Test_World 1: 

I was able to recognize all 3 objects.

##### Normalized Confusion Matrix:

![Normalized Confusion Matrix1](/Project/Normalized_Confusion_Matrix1.JPG)

##### Object Recognition:

![Object Recognition1](/ProjectOutput_1.YAML)
___

#### Test_World 2:

I was able to recognize 4/5 objects.  Glue could not be found.

#### Normalized Confusion Matrix:

![Normalized Confusion Matrix2](/Project/Normalized_Confusion_Matrix2.JPG)

##### Object Recognition:

![Object Recognition2](/Project/Output_2.YAML)

___

#### Test_World 3:

I was able to recognize 7/8 objects.  Glue could not be found.

#### Normalized Confusion Matrix:

![Normalized Confusion Matrix3](/Project/Normalized_Confusion_Matrix3.JPG)

##### Object Recognition:

![Object Recognition3](/Project/Output_3.YAML)








