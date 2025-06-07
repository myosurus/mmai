from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score

# Load data
data = pd.read_csv("ready_sales_dataset.csv")  

# Clustering by
features_clustering = ['TotalPrice', 'Quantity', 'Discount', 'UnitPrice']
X_cluster = data[features_clustering]

# Scaling
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Cluster count
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_scaled)
    inertia.append(kmeans.inertia_)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, '-o', color='blue')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Inertia (сумма квадратов ошибок)')
plt.title('Метод локтя для выбора оптимального числа кластеров')
plt.grid(True)
plt.show()

# Clustering
optimal_k = 3  
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)

# Add clusters to data
data['Cluster'] = clusters

# Flattening
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(label='Кластер')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Визуализация кластеров')
plt.grid(True)
plt.show()


silhouette_avg = silhouette_score(X_cluster_scaled, clusters)
print(f"Средний коэффициент силуэта: {silhouette_avg}")

cluster_summary = data.groupby('Cluster')[features_clustering].mean()
print(cluster_summary)

