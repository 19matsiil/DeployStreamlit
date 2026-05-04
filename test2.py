import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.title("KMeans Clustering - Mall Customer")

uploaded_file = st.file_uploader("Upload file Excel dataset (mall_costumer.xlsx)", type=["xlsx"])

if uploaded_file:
    dataset = pd.read_excel(uploaded_file)
    st.title("Data Tabel")
    st.write(dataset.head())

    X = dataset.iloc[:, 3:5]
    st.write("Ukuran data:", X.shape)
    st.write("Cek missing value:", X.isnull().sum().to_dict())
    st.write(X.describe())

    wcss = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i, random_state=14)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 15), wcss)
    ax1.set_title("The Elbow Method")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("WCSS")
    st.pyplot(fig1)

    kmeans = KMeans(n_clusters=5, random_state=14)
    kmeans.fit(X)

    hasil_kmeans = X.copy()
    hasil_kmeans["cluster"] = kmeans.labels_
    st.write(hasil_kmeans.head())

    fig2, ax2 = plt.subplots()
    cluster_x = hasil_kmeans["cluster"].value_counts().index
    cluster_y = hasil_kmeans["cluster"].value_counts().values
    sns.barplot(x=cluster_x, y=cluster_y, ax=ax2)
    ax2.set_title("Frekuensi Data pada Masing-Masing Cluster (KMeans)")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Frekuensi")
    st.pyplot(fig2)

    colors = ["blue", "orange", "green", "red", "magenta"]
    centroid_cluster = kmeans.cluster_centers_

    fig3, ax3 = plt.subplots()
    for i in range(5):
        data_cluster = hasil_kmeans[hasil_kmeans["cluster"] == i]
        ax3.scatter(data_cluster.iloc[:, 0], data_cluster.iloc[:, 1],
                    s=80, c=colors[i], label=f"Cluster {i+1}")
    ax3.scatter(centroid_cluster[:, 0], centroid_cluster[:, 1],
                s=160, c="black", label="Centroids")
    ax3.set_title("Clusters of Customers")
    ax3.set_xlabel("Annual Income (k$)")
    ax3.set_ylabel("Spending Score (1-100)")
    ax3.legend()
    st.pyplot(fig3)

    hasil_kmeans["CustomerID"] = dataset["CustomerID"]
    st.write(hasil_kmeans.head())

    csv = hasil_kmeans.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Hasil Clustering (.csv)",
        data=csv,
        file_name="Hasil Clustering Menggunakan K-Means.csv",
        mime="text/csv"
    )
