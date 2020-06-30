import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import MLP as nn
from sklearn.model_selection import train_test_split

# Data loading
gaussian_df = pd.read_csv( 'gaussian_data.csv' )
# Setting aside a portion of the data for the final test of our model
test_final_df =  gaussian_df.sample(frac=0.2, random_state=42)
# Removing that part from data
gaussian_df = gaussian_df.drop( test_final_df.index )
# Getting the features for train and final test
X = gaussian_df.drop( "class", axis = 1 )
X_test_final = test_final_df.drop( "class", axis = 1 )
# Getting the labels for train and final test
y = gaussian_df["class"]
y = pd.get_dummies(y)
# To one-hot
y_test_final = test_final_df["class"]
y_test_final = pd.get_dummies(y_test_final)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test=X_train.T.values, X_test.T.values, y_train.T.values, y_test.T.values

model = nn.NeuralNet(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

model.train( X_train, y_train, X_test, y_test )

model.efficiency()

plt.plot(model.train_errors)
plt.show()
