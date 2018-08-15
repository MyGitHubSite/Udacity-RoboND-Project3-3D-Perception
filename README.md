# Udacity-RoboND-Project3-3D-Perception

### Exercise-1: Tabletop Segmentation

The following steps were completed in this exercise:

- Downsample your point cloud by applying a Voxel Grid Filter.
- Apply a Pass Through Filter to isolate the table and objects.
- Perform RANSAC plane fitting to identify the table.
- Use the ExtractIndices Filter to create new point clouds containing the table and objects separately.

Using RANSAC.py:

    # Import PCL module
    import pcl

    # Load Point Cloud file
    cloud = pcl.load_XYZRGB('tabletop.pcd')

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

#### Perform RANSAC plane fitting to identify the table.

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


To view a .pcd file:

    $ python RANSAC.py

    $ pcl_viewer voxel_downsampled.pcd
    $ pcl_viewer pass_through_filtered.pcd
    $ pcl_viewer extracted_inliers.pcd
    $ pcl_viewer extracted_outliers.pcd

