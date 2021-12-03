# Libraries needed
import pandas as p
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
from csv import writer
from sklearn.dummy import DummyRegressor
from sklearn import metrics
from sklearn.feature_selection import RFE

df = p.read_csv('/Users/stephenbyrne/Documents/College Year 5/Machine Learning/Labs/Final ass/ISEQ.csv') # Reading CSV file downloaded from Yahoo finance
c1 = df.iloc[:, 7]  # Reading in day number
c2 = df.iloc[:, 1]  # Reading in opening price
c3 = df.iloc[:, 0]  # Reading in date

timeData = p.DataFrame(df, columns = ['Date'])  # Reading in date column
timeData['Date'] = p.to_datetime(df['Date'],format='%d/%m/%Y')  # Setting format to be read in
timeData['year']=timeData['Date'].dt.year  # Reading year individually
timeData['month']=timeData['Date'].dt.month  # Reading month individually
timeData['day']=timeData['Date'].dt.day  # Reading day individually
timeData['dayofweek_num']=timeData['Date'].dt.dayofweek  # Reading day of week number individually

year = timeData['year']  # Assigning year to year variable
month = timeData['month']  # Assigning month to month variable
day = timeData['day']  # Assigning day to day variable
dayN = timeData['dayofweek_num']  # Assigning day of week number to day of week number variable
'''
with open('/Users/stephenbyrne/Documents/College Year 5/Machine Learning/Labs/Final ass/Features.csv', 'a', newline='') as f_object:
    writer_object = writer(f_object)  # Pass the CSV  file object to the writer() function
    # Result - a writer object
    # Pass the data in the list as an argument into the writerow() function
    for x in range(467):
        list_data = [year[x], month[x], day[x], dayN[x]]
        writer_object.writerow(list_data)  # Write the data to the CSV file
    f_object.close()  # Close the file object
timeData.head(8)
'''

# Reading in individual features from CSV file
c4 = df.iloc[:, 2]  # High
c5 = df.iloc[:, 3] # Low
c6 = df.iloc[:, 4]  # Close
c7 = df.iloc[:, 5]  # Close Adj
c8 = df.iloc[:, 6]  # Volume
X = c1  # X for BL is days as int
y = c2  # Target is opening price

# Mean baseline
dummy_regr_mean = DummyRegressor(strategy="mean")  # Mean baseline
dummy_regr_mean.fit(X, y)  # Fitting baseline to data
y_pred_dum_mean = dummy_regr_mean.predict(X)  # Prediction for baseline
score_dum_mean = dummy_regr_mean.score(X, y)  # Score (r^2)
MSE_dum_mean = metrics.mean_squared_error(y, y_pred_dum_mean)  # MSE of mean BL
MAE_dum_mean = metrics.mean_absolute_error(y, y_pred_dum_mean)  # MAE of mean BL

print("Mean absolute error (Baseline - Mean):", MAE_dum_mean)  # Print MAE of mean BL
print("Mean squared error (Baseline - Mean):", MSE_dum_mean)  # Print MSE of mean BL
print("Mean Baseline score:", score_dum_mean)  # Print score of mean BL

plt.figure('Mean Baseline')
plt.title("Predictions of Mean Baseline model on test data vs actual data points")  # Graph title
plt.xlabel("Time (Days) ")  # X label
plt.ylabel("Price (Euro)")  # Y label
plt.plot(X, y_pred_dum_mean, color = 'green')  # Plot for predictions against time
plt.plot(X, y, color = 'blue')  # Plot for actual data against time
predictions = mpatches.Patch(color = 'green', label = "Predictions") # legend for negative target data
actual_data = mpatches.Patch(color = 'blue', label = "Actual data") # egend for positive target data
plt.legend(handles=[predictions, actual_data], loc="lower right")
plt.show()

# Median Baseline
dummy_regr_median = DummyRegressor(strategy="median")  # Median baseline
dummy_regr_median.fit(X, y)  # Fitting baseline to data
y_pred_dum_median = dummy_regr_median.predict(X)  # Prediction for baseline
score_dum_median = dummy_regr_median.score(X, y)  # Score (r^2)
MSE_dum_median = metrics.mean_squared_error(y, y_pred_dum_median)  # Print MAE of median BL
MAE_dum_median = metrics.mean_absolute_error(y, y_pred_dum_median)  # MAE of median BL

print("Mean absolute error (Baseline - Median):", MAE_dum_median)  # Print MAE of median BL
print("Mean squared error (Baseline - Median):", MSE_dum_median)  # Print MSE of median BL
print("Median Baseline score:", score_dum_median)  # Print score of median BL

plt.figure('Median Baseline')
plt.title("Predictions of Median Baseline model on test data vs actual data points")
plt.xlabel("Time (Days)")  # X label
plt.ylabel("Price (Euro)")  # Y label
plt.plot(X, y_pred_dum_median, color = 'green')  # Plot for predictions against time
plt.plot(X, y, color = 'blue')  # Plot for actual data against time
predictions = mpatches.Patch(color = 'green', label = "Predictions")  # legend for actual data
actual_data = mpatches.Patch(color = 'blue', label = "Actual data")  # legend for prediction
plt.legend(handles=[predictions, actual_data], loc="lower right")  # legend
plt.show()

# Feature selection
feature_cols = ['High', 'Low', 'Close', 'Adj Close', 'DayN', 'Year', 'Month', 'Day']  # Columns with features we want to use
X = df[feature_cols]  #Assigning features to X
model = LinearRegression()  # Model we want to use
rfe = RFE(model)  # Performing repeated feature elimination
fit = rfe.fit(X, y)  # Fitting model to data
print("Num Features: %d" % fit.n_features_) # Printing number of features
print("Selected Features: %s" % fit.support_)  # Printing the selected features
print("Feature Ranking: %s" % fit.ranking_)  # Printing the feature ranking