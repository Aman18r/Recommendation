# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

# data = np.array([[1.0, 2.0, 3. ],
#                  [4.0, 5.0, 6.0],
#                  [7.0, 8.0, 9.0]])

# scaler = MinMaxScaler()

# scaled_data = scaler.fit_transform(data)

# print("Original Data:")
# print(data)

# print("\nScaled Data:")
# print(scaled_data)

# from matplotlib import pyplot as plt
# import numpy as np

# # x = [1, 2, 3, 4, 5]
# # y = [2, 4, 6, 8, 10]

# x = np.linspace(-5, 5, 100)
# y = x**2

# plt.plot(x,y)

# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Simple Plot')

# plt.show()

from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

kmeans = KMeans(n_clusters=2)

kmeans.fit(X)
labels = kmeans.labels_

centers = kmeans.cluster_centers_

print("Cluster Labels: ", labels)
print("Cluster Centers: ", centers)