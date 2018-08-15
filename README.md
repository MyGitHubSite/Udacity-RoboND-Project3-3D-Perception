# Udacity-RoboND-Project3-3D-Perception

### Exercise-1: Tabletop Segmentation

In brief, the steps to complete this exercise are the following:

- Downsample your point cloud by applying a Voxel Grid Filter.
- Apply a Pass Through Filter to isolate the table and objects.
- Perform RANSAC plane fitting to identify the table.
- Use the ExtractIndices Filter to create new point clouds containing the table and objects separately.

To view a .pcd file:

    $ python RANSAC.py

    $ pcl_viewer voxel_downsampled.pcd
    $ pcl_viewer pass_through_filtered.pcd
    $ pcl_viewer extracted_inliers.pcd
    $ pcl_viewer extracted_outliers.pcd

