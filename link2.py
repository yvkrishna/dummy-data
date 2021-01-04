m1 = Main('dummy-data','rice_leaf_diseases')

filters = ["median","laplacian","sobelx","sobely","gaussian"]

m1.applyFilters(filters)

img_dimensions = (150,150)
test_val_split = 0.1
(train_x, train_y, val_x, val_y) = m1.getDataset(img_dimensions,test_val_split)