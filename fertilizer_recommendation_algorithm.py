# import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# load the dataset
fertilizer = pd.read_csv("b835ac31a79726111827.csv")
fertilizer.tail()

# check the shape of the dataset
fertilizer.shape

# check the basic info of the dataset
fertilizer.info()

# check the missing values in the dataset
fertilizer.isnull().sum()

# check the duplicated values in the dataset
fertilizer.duplicated().sum()

# check the basic statistics of the dataset
fertilizer.describe()

# Only select the numerical columns
fertilizer_numeric = fertilizer.select_dtypes(include=[np.number])
corr = fertilizer_numeric.corr()
corr

sns.heatmap(corr,annot=True,cbar=True,cmap='coolwarm')
plt.show()

fertilizer['Fertilizer Name'].value_counts()

# check the distribution of the temperature column
sns.histplot(fertilizer['Temparature'], kde=True)
plt.show()

# Only select the numerical columns
features = fertilizer.select_dtypes(include=[np.number]).columns.tolist()
print(features)

# visualize the distribution of each feature
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i, subplot in zip(features, ax.flatten()):
    sns.histplot(fertilizer[i], ax=subplot, kde=True)
    subplot.set_title(i)
plt.tight_layout()
plt.show()

# plot scatter plot of each feature against the target
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i, subplot in zip(features, ax.flatten()):
    sns.scatterplot(x=i, y='Fertilizer Name', data=fertilizer, ax=subplot)
plt.tight_layout()
plt.show()

fert_dict = {
'Urea':1,
'DAP':2,
'14-35-14':3,
'28-28':4,
'17-17-17':5,
'20-20':6,
'10-26-26':7,
}

fertilizer['fert_no'] = fertilizer['Fertilizer Name'].map(fert_dict)

fertilizer['fert_no'].value_counts()

# drop the target column with name and keep the target column with numbers
fertilizer.drop('Fertilizer Name',axis=1,inplace=True)
fertilizer.head()

# Select the object columns
fertilizer.select_dtypes(include=['object']).columns

# convert the categorical columns to numerical columns using labelencoder
lb = LabelEncoder()
fertilizer["Soil Type"]=lb.fit_transform(fertilizer['Soil Type'])
fertilizer['Crop Type']=lb.fit_transform(fertilizer['Crop Type'])

fertilizer.head()

# split the dataset into features and target
x = fertilizer.drop('fert_no',axis=1)
y = fertilizer['fert_no']
# print the shape of features and target
print(f"The shape of features is: {x.shape}")
print(f"The shape of target is: {y.shape}")

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# print the shape of training and testing sets
print(f"The shape of x_train is: {x_train.shape}")
print(f"The shape of x_test is: {x_test.shape}")
print(f"The shape of y_train is: {y_train.shape}")
print(f"The shape of y_test is: {y_test.shape}")

#Scaling

# Scale the features using StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Training Models

# initialize the DecisionTreeClassifier
model = DecisionTreeClassifier()

# train the model
model.fit(x_train, y_train)

# evaluate the model on the test set and print the accuracy
accuracy = model.score(x_test, y_test)
print(f"The accuracy of the model is: {accuracy*100:.2f}%")

# evaluate the model on the training set and print the accuracy
accuracy = model.score(x_train, y_train)
print(f"The accuracy of the model on the training set is: {accuracy*100:.2f}%")

#Predictive System

def recommend_fertilizer(Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous):
    features = np.array([[Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous]])
    transformed_features = sc.transform(features)
    prediction = model.predict(transformed_features).reshape(1,-1)
    fert_dict = {1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28', 5: '17-17-17', 6: '20-20', 7: '10-26-26'}
    fertilizer = [fert_dict[i] for i in prediction[0]]

    return f"{fertilizer} is a best fertilizer for the given conditions"

# Given input values
Temparature = 56
Humidity = 0.5
Moisture = 0.6
Soil_Type = 2
Crop_Type = 3
Nitrogen = 10
Potassium = 15
Phosphorous = 6

# Use the recommendation function to get a prediction
recommend_fertilizer(Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous)

# save the trained parameters
import pickle
# save the model
pickle.dump(model,open('fertilizer_model.sav','wb'))

# save the standard scaler
pickle.dump(sc,open('fertilizer_scaler.sav','wb'))

# load the model
model = pickle.load(open('fertilizer_model.sav','rb'))

# load the scaler
sc = pickle.load(open('fertilizer_scaler.sav','rb'))