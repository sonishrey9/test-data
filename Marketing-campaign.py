# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
df = pd.read_csv("Marketing Campaign Performance - data.csv")
df


# %%
df.info()


# %%
#missing values 

df.isna().sum()


# %%
features_with_na = [features for features in df.columns if df[features].isnull().sum() > 1] #list comprehension to identify null values in Data frame

for feature in features_with_na:
    print(feature, np.round(df[feature].isnull().mean() * 100, 4),  ' % missing values')

# %% [markdown]
# # As Data is Realtime and no ML model is to be built, so we're avoiding preproprocessing of the data

# %%
df.fillna(0) # We won't be deleteing any data, replacing all NA values with 0 

# %% [markdown]
# # . All The Numerical Variables **

# %%
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O'] # list comprehension feature that are not equal to object type

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df[numerical_features].head()


# %%
# Distribution of the Data, to check skewness of the data

a = 3  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(30,20))

for i in numerical_features:
    plt.subplot(a, b, c)
    plt.title('{}'.format(i))

    sns.histplot(data= df, x= i)

    c = c + 1

plt.show()


# %%
## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# %%
a = 5  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(20, 50))
font = {'size': 12}
  
# using rc function
plt.rc('font', **font)

for i in numerical_features:
    plt.subplot(a, b, c)
    plt.title('{} vs Channel '.format(i))
    sns.barplot(x= "channel", y=  i, data= df, palette="deep")
    plt.xticks(rotation = 40)
    c = c + 1

plt.show()

# %% [markdown]
# ## BOXPLOT SHOWS OUTLIERS INCLUDED IN THE DATASET

# %%
a = 5  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(20, 50))
font = {'size': 12}
  
# using rc function
plt.rc('font', **font)

for i in numerical_features:
    plt.subplot(a, b, c)
    plt.title('{} vs Channel '.format(i))
    sns.boxplot(x= "channel", y=  i, data= df, palette="deep")
    plt.xticks(rotation = 40)
    c = c + 1

plt.show()


# %%
categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']
categorical_features


# %%
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))

# %% [markdown]
# # **2.1 Which channel’s ads do the users like the most? (hint: click through rate or CTR)**
# 

# %%
a = 5  # number of rows
b = 3  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(20, 50))
font = {'size': 12}
  
# using rc function
plt.rc('font', **font)

for i in numerical_features:
    plt.subplot(a, b, c)
    plt.title('{} vs impressions, SubPlot {}{}{} '.format(i,a, b,c))
    sns.scatterplot(x= "impressions", y=  i, data= df, palette="deep", hue = "channel")
    plt.xticks(rotation = 40)
    c = c + 1

plt.show()


# %%
for i in numerical_features:
    g = sns.relplot(data=df, x= "impressions", y=i, col= "channel", col_wrap=6, hue= "channel", height= 4, aspect=.7,kind="scatter", legend= True, palette="deep")
    g.fig.suptitle(f" impressions vs  {i} ", fontweight ="bold")
    plt.subplots_adjust(top=.90)
    plt.xticks(rotation = 90)
plt.show()

# %% [markdown]
# ## focusing on clicks vs impressions graph, APP - Acqusition Android, APP - Acqusition IOS and Web retargetting are most liked by the users. As they have high click / impressions. 
# %% [markdown]
#  ## **2.2 What’s the CPI of App channels (APP-Acquisition-Android and APP-Acquisition-iOS) by country and OS?**
# 

# %%

df["cost"] = pd.to_numeric(df.cost, errors='coerce') #converting cost to numeric data

# %% [markdown]
# ## CPI=  Cost per install
# 
# %% [markdown]
# ## CPI by OS

# %%
plt.figure(figsize=(20,10))
sns.barplot(data = df, x = "channel", y =  "cost", hue = "installs", palette="deep",   order=["APP-Acquisition-Android", "APP-Acquisition-iOS"])
plt.title("CPI of Different channels by OS")
plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.)
plt.show()

# %% [markdown]
# ## CPI on basis of different country

# %%
plt.figure(figsize=(20,10))
sns.barplot(data = df, x = "channel", y =  "cost", hue = "country", palette="deep",   order=["APP-Acquisition-Android", "APP-Acquisition-iOS"])
plt.title("CPI of Different channels by country")
plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.)
plt.show()

# %% [markdown]
# # **2.3 In SEM channels, which country has the best conversion rate?**

# %%
plt.figure(figsize = (30,15))
plt.title('Channles puchases country wise ')
sns.barplot(x= "channel"  , y=  'purchases', data= df, palette="deep", hue = "country" , order=["SEM-Brand", "SEM-Flights", "SEM-Hotels"])
plt.xticks(rotation = 0)
plt.show()

# %% [markdown]
# ## from above graph we can conclude that country S has high conversion Rate as it has maxiumum purchases 
# %% [markdown]
#  # **2.4 Which Web channel has better CPA?**
# %% [markdown]
# ### (Cost Per Action or Acquisition) , (CPA) is calculated by dividing the total cost of conversions by the total number of conversions

# %%
g = sns.relplot(data=df, x= "cost", y= 'purchases', col= "channel", col_wrap=7, hue= "channel", height= 4, aspect=.7,kind="scatter", legend= True, palette="deep")
g.fig.suptitle(f" Cost Per Action or Acquisition i.e no of purchase vs cost ", fontweight ="bold")
plt.subplots_adjust(top=.85)
plt.xticks(rotation = 90)
plt.show()


# %%
g = sns.relplot(data=df, x= "cost", y= 'clicks', col= "channel", col_wrap=7, hue= "channel", height= 4, aspect=.7,kind="scatter", legend= True, palette="deep")
g.fig.suptitle(f" Cost Per click ", fontweight ="bold")
plt.subplots_adjust(top=.85)
plt.xticks(rotation = 90)
plt.show()

# %% [markdown]
# ## CPA of channel SEM brand is better than other channles as, cost of acqusition is less and purchases by customer is moree.
# %% [markdown]
# # **2.5 Identify the top 3 performing channels of each country in terms of ROI.**
# 
# %% [markdown]
# ### The basic ROI calculation is: ROI = (Net Profit/Total Cost)*100.

# %%
ROI = (df.revenue / df.cost) * 100
data = pd.DataFrame(ROI, columns = ["ROI"])
# Using DataFrame.insert() to add a column
df["ROI"] = data


# %%
df.head()


# %%
plt.figure(figsize=(20,10))
sns.barplot(data = df, x = "channel", y = "ROI",  hue = "country")
plt.title("ROI of Diffrent channels based on country")
plt.show()


# %%
plt.figure(figsize=(20,10))
sns.barplot(data = df, x = "channel", y = "ROI",  hue = "channel")
plt.title("ROI of Diffrent channels based on country")
plt.show()


# %%
plt.figure(figsize=(20,10))
sns.lineplot(data = df, x = "channel", y = "ROI",size = "country" , hue = "country", style = "country", markers=True, dashes=False)
plt.title("ROI of Diffrent channels based on country")
plt.show()

# %% [markdown]
# ### Based on above graph, we can see ROI of all countries are majorly contributed by
# ### 1. SEM Brand
# ### 2. SEM fligts
# ### 3. SEM Hotels
# ### 4. Web Retargeting
# %% [markdown]
# ## **Additional graphs to understand distribution of ROI vs revenue & cost**

# %%
plt.figure(figsize=(20,10))
sns.lineplot(data = df, x = "ROI", y = "revenue", size = "channel", hue = "channel", palette= "deep", style = "channel", markers=True, dashes=False )
plt.title("ROI vs Revenue")
plt.show()


# %%
plt.figure(figsize=(20,10))
sns.lineplot(data = df, x = "ROI", y = "cost",size =  "channel", hue = "channel", style = "channel", markers=True, dashes=False)
plt.title("ROI vs cost")
plt.show()

# %% [markdown]
# # **3. What would you recommend our Digital Marketing team to do when they run online campaigns?**
# 
# %% [markdown]
# ### 1. I would Recommend DM team to focus more on channels APP - Acqusition Android, APP - Acqusition IOS and Web retargetting, as they have higher clicks and purchases!, but with equivalent high cost of acqusition.
# 
# ### 2. focusing of SEM brand,SEM flights and SEM hotel is also important as they have high contribution is ROI. 
# ### 3. CPC is less for web Acqusition
# ### 4. CPI is more of IOS OS.
# 
# %% [markdown]
# 

