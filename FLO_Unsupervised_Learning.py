
###############################################################
# Customer Segmentation with Unsupervised Learning
###############################################################

###############################################################
# Business Problem
###############################################################

# With Unsupervised Learning methods (Kmeans, Hierarchical Clustering), it is desired to divide customers into clusters and observe their behavior.

###############################################################
# Dataset Story
###############################################################

# The dataset consists of information obtained from the past shopping behavior of customers who made their last purchases on OmniChannel (both online and offline) in 2020 - 2021.

# 20.000 observation, 13 variables

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the most recent purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_date : Customer's last purchase date
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : Last shopping date made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases
# customer_value_total_ever_online : The total fee paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months
# store_type : It refers to 3 different companies. If the person who shopped from company A made it from company B, it was written as A, B.

###############################################################
# TASK 1: Read the dataset and choose the variables to use when segmenting customers.
###############################################################

import pandas as pd
import yellowbrick
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
import warnings
warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)


df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()
df.info()
df.columns
df.head()

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[ns]').dt.days # information on how many days ago customers last shopped
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[ns]').dt.days

model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
model_df.head()

###############################################################
# TASK 2: Customer Segmentation with K-Means
###############################################################

#1. Standardize the variables.
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show()

# Log transformation to ensure normal distribution

model_df['order_num_total_ever_online']=np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline']=np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline']=np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online']=np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency']=np.log1p(model_df['recency'])
model_df['tenure']=np.log1p(model_df['tenure'])
model_df.head()

# Scaling

sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()

# 2. Determine the optimum number of clusters.

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()


# 3. Build model and segment customers.

k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments=k_means.labels_+1
segments

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head()


# 4. Examine each segment statistically.

final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max","std"],
                                  "order_num_total_ever_offline":["mean","min","max","std"],
                                  "customer_value_total_ever_offline":["mean","min","max","std"],
                                  "customer_value_total_ever_online":["mean","min","max","std"],
                                  "recency":["mean","min","max","std"],
                                  "tenure":["mean","min","max","count","std"]})


###############################################################
# TASK 3: Customer Segmentation with Hierarchical Clustering
###############################################################

# 1. Determine the optimum number of clusters using the dataframe you standardized in Task 2.

hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show()

hc_average = linkage(model_df, 'average')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=0.7, color='r', linestyle='-.')
plt.show()

# 2. Build your model and segment your customers.

hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)


final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments+1
final_df.head()

# 3. Examine each segment statistically.

final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max","std"],
                                  "order_num_total_ever_offline":["mean","min","max","std"],
                                  "customer_value_total_ever_offline":["mean","min","max","std"],
                                  "customer_value_total_ever_online":["mean","min","max","std"],
                                  "recency":["mean","min","max","std"],
                                  "tenure":["mean","min","max","count","std"]})