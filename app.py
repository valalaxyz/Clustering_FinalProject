import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from PIL import Image

header_image = Image.open('Credit card clustering (1).png')
st.image(header_image, use_column_width=True)

df = pd.read_csv('CC_clustering.csv')

X = df.drop(['CUST_ID'], axis=1)
X = X.drop(['Unnamed: 0'], axis=1)

scaler = StandardScaler()
scaled_df = scaler.fit_transform(X)

normalized_df = normalize(scaled_df)
normalized_df = pd.DataFrame(normalized_df)

pca = PCA(n_components=2)
X_principal_c = pca.fit_transform(normalized_df)
X_principal = pd.DataFrame(X_principal_c, columns=['P1', 'P2'])

def plot_elbow_method():
    plt.figure(figsize=(15, 8))
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_principal)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    return fig

def k_means(n_clust):
    kmeans = KMeans(n_clusters=n_clust, random_state=42).fit(X_principal)
    X_principal['Labels'] = kmeans.labels_

    silhouette_avg = silhouette_score(X_principal, kmeans.labels_)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='P1', y='P2', hue='Labels', data=X_principal,
                    palette=sns.color_palette('hls', n_clust), ax=ax, legend=None)

    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids')

    for i, center in enumerate(centroids):
        ax.text(center[0], center[1], f'Cluster {i}', fontsize=12, fontweight='bold',
                 ha='center', va='center', color='white', bbox=dict(facecolor='black', alpha=0.5))

    return fig, X_principal.copy(), kmeans.labels_

st.title("Data Visualization with KMeans Clustering")

# Sidebar with options
st.sidebar.header("Visualization Options")
show_data = st.sidebar.button("Show Raw Data")
show_elbow = st.sidebar.button("Show Elbow Method") 

# Display the original dataset
if show_data:
    st.header("Dataset")
    st.write(X)

# Elbow method visualization
if show_elbow:
    st.header("Elbow Method")
    st.pyplot(plot_elbow_method())

# KMeans clustering
st.sidebar.subheader("Choose Number of Clusters (K)")
clust = st.sidebar.slider("Select number of clusters", 2, 10, 3)

fig, clustered_data, labels = k_means(clust)
    
X_with_clusters = X.copy() 
X_with_clusters['Cluster'] = labels 

st.subheader("Data with Cluster Labels")
st.write(X_with_clusters)

cluster_counts = clustered_data['Labels'].value_counts()
st.subheader("Cluster Counts")
st.bar_chart(cluster_counts)

cluster_colors = sns.color_palette('Set1', clust)

##############################
st.subheader("Spending Behavior per Cluster")
col1, col2 = st.columns(2)

# Box plot for 'PURCHASES' per cluster
with col1:
    fig_box, ax = plt.subplots()
    sns.boxplot(x='Cluster', y='PURCHASES', data=X_with_clusters, ax=ax, palette=cluster_colors)
    ax.set_title('Purchases Distribution per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Purchases')
    st.pyplot(fig_box)

# Bar plot showing the mean of 'PURCHASES' per cluster
with col2:
    fig_bar, ax = plt.subplots()
    sns.barplot(x='Cluster', y='PURCHASES', data=X_with_clusters, estimator=np.mean, ax=ax, palette=cluster_colors)
    ax.set_title('Average Purchases per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Avg. Number of Purchases')
    st.pyplot(fig_bar)

##############################
st.subheader("Payment Behavior per Cluster")
col1, col2 = st.columns(2)

# Box plot for 'PAYMENTS' per cluster
with col1:
    fig_box, ax = plt.subplots()
    sns.boxplot(x='Cluster', y='PAYMENTS', data=X_with_clusters, ax=ax, palette=cluster_colors)
    ax.set_title('Payments Distribution per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Payments')
    st.pyplot(fig_box)

# Bar plot showing the mean of 'PAYMENTS' per cluster
with col2:
    fig_bar, ax = plt.subplots()
    sns.barplot(x='Cluster', y='PAYMENTS', data=X_with_clusters, estimator=np.mean, ax=ax, palette=cluster_colors)
    ax.set_title('Average Payments per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Avg. Number of Payments')
    st.pyplot(fig_bar)

##############################
st.subheader("Credit Utilization per Cluster")
col1, col2 = st.columns(2)

# Box plot for 'BALANCE' per cluster
with col1:
    fig_box, ax = plt.subplots()
    sns.boxplot(x='Cluster', y='BALANCE', data=X_with_clusters, ax=ax, palette=cluster_colors)
    ax.set_title('Balance Distribution per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Balance')
    st.pyplot(fig_box)

# Bar plot showing the mean of 'BALANCE' per cluster
with col2:
    fig_bar, ax = plt.subplots()
    sns.barplot(x='Cluster', y='BALANCE', data=X_with_clusters, estimator=np.mean, ax=ax, palette=cluster_colors)
    ax.set_title('Average Balance per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Avg. Number of Balance')
    st.pyplot(fig_bar)

##############################
st.subheader("Cash Advance Usage per Cluster")
col1, col2 = st.columns(2)

# Box plot for 'CASH_ADVANCE' per cluster
with col1:
    fig_box, ax = plt.subplots()
    sns.boxplot(x='Cluster', y='CASH_ADVANCE', data=X_with_clusters, ax=ax, palette=cluster_colors)
    ax.set_title('Cash Advance Distribution per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Cash Advances')
    st.pyplot(fig_box)

# Bar plot showing the mean of 'CASH_ADVANCE' per cluster
with col2:
    fig_bar, ax = plt.subplots()
    sns.barplot(x='Cluster', y='CASH_ADVANCE', data=X_with_clusters, estimator=np.mean, ax=ax, palette=cluster_colors)
    ax.set_title('Average Cash Advances per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Avg. Number of Cash Advances')
    st.pyplot(fig_bar)

    


