import pandas as pd
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento de Datos:

# 1.a load the dataset
employees_data = pd.read_csv('EmployeesData.csv', encoding='ascii')

# 1.b check for missing values
missing_values = employees_data.isnull().sum()
print('Missing values: ')
print(missing_values)
print('\n')
print('DataSet: ')
print(employees_data)
print('\n')

# 2.a convert the 'LeavOrNot' column to categorical labels
employees_data['LeaveOrNot'] = employees_data['LeaveOrNot'].map({0: 'Not Leave', 1: 'Leave'})

# 2.b Display the updated head to verify the changes
print('Updated dataset shape:', employees_data.shape)
print(employees_data.head())
print('\n')

# 3.a Delete rows with missing values in 'ExperienceInCurrentDomain' and 'JoingYear' colums
employees_data.dropna(subset = ['ExperienceInCurrentDomain', 'JoiningYear'], inplace = True)

# 3.b Display the shape of the updated dataset to confirm the changes
print('Updated dataset shape:', employees_data.shape)
print(employees_data.head())
print('\n')

# 4.a Impute missing data in the 'Age' column with the mean
age_mean = employees_data['Age'].mean()
employees_data['Age'] = employees_data['Age'].fillna(age_mean)
print('\n')

# 4.b Display the updated head to verify the changes
print('Updated dataset shape: ', employees_data.shape)
print(employees_data[['Education', 'Age']])

# 5.a Impute missing data in the 'PaymentTier' column with the mode
PaymentTier_mode = employees_data['PaymentTier'].mode()
employees_data['PaymentTier'] = employees_data['PaymentTier'].fillna(PaymentTier_mode)
print('\n')

# 5.b Display the updated head to verify the changes
print('Updated dataset shape: ', employees_data.shape)
print(employees_data[['Education', 'PaymentTier']])

# 6.a Identify numeric columns (excluding 'LeaveOrNot' column)
numeric_cols = employees_data.select_dtypes(include = [np.number]).columns.tolist()

# 6.b Calculate IQR for each numeric column
for col in numeric_cols:
    Q1 = employees_data[col].quantile(0.25)
    Q3 = employees_data[col].quantile(0.75)
    IQR = Q3 - Q1
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filter out outliter
    employees_data = employees_data[(employees_data[col] >= lower_bound) & (employees_data[col] <= upper_bound)]

# 6.c Display the shape of the dataset after removing outliers
print('Dataset shape after removing outliers:', employees_data.shape)
#print(employees_data)


# Analisis Exploratorio de Datos (EDA):

# 7.a Calculate gender distribution
gender_distribution = employees_data['Gender'].value_counts()

# 7.b Plotting
plt.figure(figsize = (8, 6), facecolor = 'white')
plt.pie(gender_distribution, labels = gender_distribution.index, autopct = '%1.1f%%', startangle = 140)
plt.title('Gender Distribution among Employees')
plt.show()

# 8.a setting up the figure for subplots
plt.figure(figsize = (14, 6), facecolor = 'white')

# 8.b Histogram for levels of employee study
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st subplot
education_counts = employees_data['Education'].value_counts()
plt.bar(education_counts.index, education_counts.values, color = 'Skyblue')
plt.title('Distribution of levels of Employees study')
plt.xlabel('Education Level')
plt.ylabel('Number of Employees')

# 8.c pie chart for levels of Employees Study
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd subplot
plt.pie(education_counts, labels = education_counts.index, autopct = '%1.1f%%', startangle = 140)
plt.title('Proportion of Levelst of Employees Study')

plt.tight_layout()
plt.show()

#¿Son los jóvenes más propensos a tomar licencias?
# 9.a setting the aesthetic syle of the plot
sns.set(style='whitegrid')

# 9.b creating age groups for better visualization
bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
employees_data['AgeGroup'] = pd.cut(employees_data['Age'], bins, labels = ['18-24', '25-29', '30-34','35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69'])

# 9.c plotting the histogram
plt.figure(figsize = (12, 6), facecolor = 'white')
ax = sns.histplot(data = employees_data, x = 'AgeGroup', hue = 'LeaveOrNot', multiple = 'stack', palette = 'coolwarm', shrink = .8)
ax.set_title('Leave Distrubution Across Age Groups')
ax.set_xlabel('Age Group')
ax.set_ylabel('Number of Employees')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
# respuesta es seria entre 25 y 29

#¿Está el dataset balanceado? (por balanceado nos referimos a si la proporción de clases es similar)
# 10.a plotting the class dsitribution
plt.figure(figsize=(8, 6), facecolor='white')
class_counts = employees_data['LeaveOrNot'].value_counts()
#sns.barplot(x = class_counts.index, y = class_counts.values, palette='viridis')
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette='viridis', legend=False)
plt.title('Class Distribution of LeaveOrNot')
plt.xlabel('LeaveOrNot')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Leave', 'Leave'])
plt.tight_layout()
plt.show()

# 10.b Calculating the ration fo classes
class_ratio = class_counts[0] / class_counts[1]
print('Class Ratio (Not Leave / Leave):', class_ratio)
# respuesta: no esta blanciado
