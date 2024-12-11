# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Normalize the data for clustering
scaler = StandardScaler()
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
X_scaled = scaler.fit_transform(X)

# Plot Histogram for each feature in the Iris dataset
plt.figure(figsize=(10, 8))
df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].hist(bins=20, edgecolor='black', color='lightblue')
plt.suptitle('Histograms of Features in the Iris Dataset')
plt.show()

# Plot Heatmap of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Iris Features')
plt.show()

# Plot Scatterplot to visualize relationships between features
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='Set1')
plt.title('Scatterplot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Perform K-Means Clustering (using 3 clusters, as we have 3 species in Iris dataset)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Plot K-Means Clustering result
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='cluster', data=df, palette='Set1')
plt.title('K-Means Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Cluster')
plt.show()

# Perform Linear Regression (Predicting Petal Length using Sepal Length)
X_reg = df[['sepal length (cm)']]  # Feature for regression
y_reg = df['petal length (cm)']  # Target for regression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Create and train the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Plot Linear Regression results
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='red', label='True values')
plt.plot(X_test, y_pred, color='blue', label='Predicted Line')
plt.title('Linear Regression: Petal Length vs Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

# Clustering Evaluation with Silhouette Score
silhouette_avg = silhouette_score(X_scaled, df['cluster'])
print(f"Silhouette Score for Clustering: {silhouette_avg}")
