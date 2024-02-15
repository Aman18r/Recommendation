#Importing libraries , Here we have used pandas for Reading the Csv, data cleaning ,data manipulation.
#Numpy and Matplotlib for various mathematical uses , lastly sklearn for Kmeans and min_max scaler.
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  #for normalization of data
from matplotlib import pyplot as plt #for making graphs
from sklearn.cluster import KMeans #partition of data into clusters 

#Here ,we are checking for any missing values.
# Loading and cleaning data
missing_value=["N/a","na",np.nan]
df=pd.read_csv("D:\AI_HEALTH_ENGINE\Recommendation_system_data_set.csv",na_values=missing_value)
df=df.replace(r'^\s*$', np.nan, regex=True)
df.isnull().sum() #count missing value

# print(df.tail())

#Dropping the entries having missing values as it cannot be replaced i.e,- Specialisation ,City ,Hospital/Clinic ,Experience     
df=df.dropna()

# Printing Data After Cleaning

df.head()  #print(df.head(10))


#Here, we have used minmaxscaler to normalise the data of coumn Experience and Awards
scaler=MinMaxScaler()
scaled=scaler.fit_transform(df[['Experience']])
df[['Experience']]=scaled
df['Awards']=60*df['Padma_Vibhushan']+50*df['Padma_Bhushan']+40*df['Padma_Shri']+30*df['Dhanvantari_Award']+20*df['BC_Roy_National_Award']+10*df['Other_Awards']
scaled2=scaler.fit_transform(df[['Awards']])
df[['Awards']]=scaled2
df.drop('Padma_Vibhushan',
  axis='columns', inplace=True)
df.drop('Padma_Bhushan',
  axis='columns', inplace=True)
df.drop('Padma_Shri',
  axis='columns', inplace=True)
df.drop('Dhanvantari_Award',
  axis='columns', inplace=True)
df.drop('BC_Roy_National_Award',
  axis='columns', inplace=True) 
df.drop('Other_Awards',
  axis='columns', inplace=True)
df

# print(df.head())

# Plotting Experience Awards after Normalising

#The scatter plot between Expereinece vs Awards helps in visualization of data
plt.scatter(df['Experience'],df['Awards'])

# plt.show()

# Calculating SSE to measure how well the data set is Clustered.

#Here we have calculated the sum of squared error(sse) and measured how well it is clustered using elbow method
sse = []
k_rng = range(1,10) #loop (1-9)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Experience','Awards']])
    sse.append(km.inertia_)

# Plot between sum of squared error vs value of KMeans(Elbow Method)
    
#Here, We have plotted a graph between SSE and KMeans to select the optimal number of clusters.
#This method is called Elbow method and the point obtained is elbow point(n=4).
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# plt.show()

# Kmeans , Cluster and Labelling

#We have divided the data in 4 clusters.
km=KMeans(n_clusters=4)
km
# km.labels_
# km.cluster_centers_

# print(km.labels_)
# print(km.cluster_centers_)

#Labelling to the dataset
y_predict=km.fit_predict(df[['Awards','Experience']])

df['cluster']=y_predict
df

# print(df['cluster'])

# print(df)

import pickle

pickle.dump(km, open('model.pkl','wb'))

# Enter the Name of the Doctor , Specialisation , City ,Experience , Award Points of the Doctor

specialisaton=input("Enter the Specialisation of the Doctor: ")
city=input("Enter the City of the Doctor: ")
doctor_Experience=float(input("Enter the Experience of the Doctor: "))
doctor_Awards_Points=float(input("Enter the Award Points of the Doctor: "))

# Normalizing the Data

#Here we are normalising the data given by the user
Experience_Normalised=doctor_Experience/70
Awards_Point_Normalised=doctor_Awards_Points/100

# Actual Prediction

predicted_user=km.predict([[doctor_Experience,doctor_Awards_Points]])
predicted_user

# Finding better Doctor according to the experience and awards

#Using loop we have iterated the data set and used a condition to return the better doctor to the final list.

final=[]
if(predicted_user <4): #for outliers
    for i in range((df.shape[0])):
        if(str (df.iloc[i,2]).count(city)>0 and str (df.iloc[i,1]).count(specialisaton)>0 and df.iloc[i,6]==predicted_user and Experience_Normalised< float(df.iloc[i,4]) and Awards_Point_Normalised < float(df.iloc[i,5])):
            final.append(df.iloc[i])
else :
    print("Sorry! We cannot match these in data set")

# Doctor list
    
#The final Data.
if(len(final)):
    print(final)
else:
    print("Your Doctor is the best in your Area.")



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~