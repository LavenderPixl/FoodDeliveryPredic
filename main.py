import preprocessing
import data_visualization as visualization
from models.multiple_linear_regression import main

# 1 = Replaces N/A with mean value.
# 2 = Replaces N/A with 0.
processed_data = preprocessing.preprocess(1)
visualization.visualize(processed_data)
main(processed_data)