l1 = Data_Loader('dummy-data','rice_leaf_diseases')
training_classes = l1.getClasses()

filters = ["median","laplacian","sobelx","sobely","gaussian"]
f1 = Filters(filters)

for folder in training_classes:
  for filter in filters:
    path = os.path.join('dummy-data','rice_leaf_diseases', folder)
    if filter == "median":
      f1.applyMedian(path)
    elif filter == "laplacian":
      f1.applylaplacian(path)
    elif filter == "sobelx":
      f1.applysobelx(path)
    elif filter == "sobely":
      f1.applysobely(path)
    elif filter == "gaussian":
      f1.applygaussian(path)


for folder in training_classes:
  path = os.path.join('dummy-data','rice_leaf_diseases', folder)
  img_label = l1.create_dataset(path)
  print()

(train_x, train_y, val_x, val_y) = l1.prepare_dataset((150,150),0.1,img_label)