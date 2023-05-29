
# Methodology Journal

Format:

Title: 
Notebook_name: 
Features:  
Labels: 
Architecture: 
Dataset: 
HP: 
Comments: 
Results: 

## Linear NN coupling acceleration and tango orientation
Notebook name: acc_and_tango_ori_to_position
Features: synced/acc + pose/tango_ori
Labels: pose/tango_pos difference 
Architecture: 64 | 64 | 64 | 64 | 3
Ddataset: RoNiN, first 5000 datapoints.
HP: LR = 0.001, batch_size = 1, test_set = 0.2
Comments: Frustratingly the NN doesnt seem to learn. This was performed without sequencing the data, which might lead to better results.
Result: Unable to capture position difference correctly, doesnt learn causing the position to drift.