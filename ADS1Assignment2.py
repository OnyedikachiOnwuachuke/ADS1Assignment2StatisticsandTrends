# Importing Libraries we are going to be using on this assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew



"""
Creating a def function to read in the world bank dataset and returns  two 
dataframes:one with years as columns, and the other with countries as column
"""

      
def read_worldbankdata(filename):
    
    """
    A function that reads in world bank data on climate change with various 
    indicators from and return both the original and transposed version of
    the dataset
    
    Args:
        filename: the name of the world bank data that will be read for analysis 
        and manupulation
        
            
    Returns: 
        The dataset as df_wrld_years(years as column) and df_wrld_countries as its transposed version
    """  
    # Read the World Bank data into a dataframe
    df_wrld_bank = pd.read_csv('worldbankdata.csv', skiprows=4)

    # Transpose the dataframe and set the country name as the index
    df_wrld_years = df_wrld_bank.drop(['Country Code', 'Indicator Code'], axis=1)
    df_wrld_countries = df_wrld_bank.drop(['Country Name', 'Country Code', 'Indicator Code'], axis=1) \
        .set_index(df_wrld_bank['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    df_wrld_countries = df_wrld_countries.set_index('Year').dropna(axis=1)
    

    return df_wrld_years, df_wrld_countries

print(read_worldbankdata)

# Reading the function that will produce two DataFrame(years as column and countries as column)
df_wrld_years, df_wrld_countries = read_worldbankdata('worldbankdata.csv')


"""
For further analysis to be carried out on the world bank data on climate change
we will be selcting some few indicators and select G8 countries including China and India of choice 

"""
indicators = df_wrld_years[df_wrld_years['Indicator Name'].isin(["Urban population", "CO2 emissions (kt)", "Electricity production from nuclear sources (% of total)", "Renewable energy consumption (% of total final energy consumption)"])]

countries = ['United States', 'China', 'India', 'Germany', 'Italy', 'United Kingdom', 'Russian Federation', 'France', 'Canada', 'Japan']
selected_countries = indicators[indicators['Country Name'].isin(countries)]
selected_countries = selected_countries.dropna(axis=1)
selected_countries = selected_countries.reset_index(drop=True)
selected_countries

"""
Exploring a group statistical properties of a few selected indicators and the G8 countries
including China & India,and compare between the countries and the indicators

"""
# using the describe function for this selected years '1990', '2000', '2010', '2014'
stats_desc = selected_countries.groupby(["Country Name","Indicator Name"]) \
[['1990', '2000', '2010', '2014']].describe()

print(stats_desc)

"""
Looping through each group and calculate the some other summary statistics for 
including China & India, for the mean,min,max,median and standard deviation
"""

summary_stats_others = selected_countries.groupby(['Country Name', 'Indicator Name'])
for name, group in summary_stats_others:
    print(name)
    print('Mean:', group.mean()['1990':'2014'])
    print('Min:', group.min()['1990':'2014'])
    print('Max:', group.max()['1990':'2014'])
    print('Median:', group.median()['1990':'2014'])
    print('Standard deviation:', group.std()['1990':'2014'])


"""
Now exploring the statistical properties of a each selected indicators
and compare amongst the G8 countries including China & India,and compare between
the countries and the indicators for a more critical summary statistics
by creating new dataframe for each indicators for the analysis
"""

#Creating a dataFrame for Urban population for further statistics analysis and plotting.
Urban_pop_ind = selected_countries[selected_countries["Indicator Name"] == "Urban population"]
Urban_pop_ind = Urban_pop_ind.set_index('Country Name', drop=True)
Urban_pop_ind= Urban_pop_ind.transpose().drop('Indicator Name')
Urban_pop_ind[countries] = Urban_pop_ind[countries].apply(pd.to_numeric, errors='coerce', axis=1)
print(Urban_pop_ind)

#statistic summary for urban population
print(Urban_pop_ind.describe())
print(Urban_pop_ind.mean()) 
print(Urban_pop_ind.median()) 
print(Urban_pop_ind.std())
print('Skewness:', Urban_pop_ind.skew())
   
#Creating a dataFrame for CO2 emissions for further statistics analysis and plotting.
CO2_ind = selected_countries[selected_countries["Indicator Name"] == "CO2 emissions (kt)"]
CO2_ind = CO2_ind.set_index('Country Name', drop=True)
CO2_ind = CO2_ind.transpose().drop('Indicator Name')
CO2_ind [countries] = CO2_ind [countries].apply(pd.to_numeric, errors='coerce', axis=1)
print(CO2_ind)

#Statistic summary for CO2 emissions
print(CO2_ind.describe())
print(CO2_ind.mean())
print(CO2_ind.median())
print(CO2_ind.std())
print('Skewness:', CO2_ind.skew())

#Creating a dataFrame for Electricity production from nuclear sources for further statistics analysis and plotting.

Elect_prod_nuclear = selected_countries[selected_countries["Indicator Name"] == \
                                        "Electricity production from nuclear sources (% of total)"]
Elect_prod_nuclear = Elect_prod_nuclear.set_index('Country Name', drop=True)
Elect_prod_nuclear = Elect_prod_nuclear.transpose().drop('Indicator Name')
Elect_prod_nuclear[countries] = Elect_prod_nuclear[countries].apply(pd.to_numeric, errors='coerce', axis=1)
print(Elect_prod_nuclear)

#Statistic summary for Electricity production from nuclear sources
print(Elect_prod_nuclear.describe())
print(Elect_prod_nuclear.mean())
print(Elect_prod_nuclear.median())
print(Elect_prod_nuclear.std())
print('Skewness:', Elect_prod_nuclear.skew())

#Creating a dataFrame for Renewable energy consumption for further statistics analysis and plotting.
Renew_Energy_Cons = selected_countries[selected_countries["Indicator Name"] \
                                       == "Renewable energy consumption (% of total final energy consumption)"]
Renew_Energy_Cons = Renew_Energy_Cons.set_index('Country Name', drop=True)
Renew_Energy_Cons = Renew_Energy_Cons.transpose().drop('Indicator Name')
Renew_Energy_Cons[countries] = Renew_Energy_Cons[countries].apply(pd.to_numeric, errors='coerce', axis=1)
print(Renew_Energy_Cons)

#Statistic summary for Renewable energy consumption
print(Renew_Energy_Cons.describe())
print(Renew_Energy_Cons.mean())
print(Renew_Energy_Cons.median())
print(Renew_Energy_Cons.std())
print('Skewness:', Renew_Energy_Cons.skew())

"""
Plotting a line plot to show the trends over years for G8 countries including China & India
for both indicators of CO2 emmissions & Urban Population
"""
# The line plot for CO2 emissions
plt.figure(figsize=(10,6))
plt.style.use('default')
CO2_ind.plot()
plt.title('CO2 emmissions of G8 countries including China & India)')
plt.xlabel('Year')
plt.ylabel('"CO2 emissions (kt)')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Line plot CO2 emmisions.png', dpi=300)
plt.show()


#The Line plot for Urban Population
plt.figure(figsize=(10,6))
plt.style.use('default')
CO2_ind.plot()
plt.title('Urban Population of G8 countries including China & India)')
plt.xlabel('Year')
plt.ylabel('Urban population')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Line plot Urban population.png', dpi=300)
plt.show()


"""
Plotting a group bar plot to show the relationship for Electricity production
from nuclear sources for G8 countries including China & India
"""

plt.style.use('seaborn')

# create a new column with the decade for each year
CO2_ind['Decade'] = CO2_ind.index.map(lambda x: str(x)[:3] + '0s')

# group by decade and sum the CO2 emissions for each country
CO2_decade = CO2_ind.groupby('Decade').sum()

colors = {'United States': 'blue', 'Canada': 'red', 'United Kingdom': 'green', 'France': 'orange',
          'Germany': 'purple', 'Italy': 'brown', 'Japan': 'pink', 'Russia': 'gray', 'China': 'teal',
          'India': 'magenta'}

# create a new column with the decade for each year
Elect_prod_nuclear['Decade'] = Elect_prod_nuclear.index.map(lambda x: str(x)[:3] + '0s')

# group by decade and sum the CO2 emissions for each country
Elect_prod_nuclear = Elect_prod_nuclear.groupby('Decade').sum()

# plot the data as a bar chart
Elect_prod_nuclear.plot(kind='bar', color=[colors.get(c, 'black') for c in CO2_decade.columns])
plt.title('Electricity production from nuclear sources by G8 countries including China & India')
plt.xlabel('Decade')
plt.ylabel('Electricity production from nuclear sources (% of total)')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('group barchat electricity production from nuclear sources.png', dpi=300)
plt.show()

"""
Plotting a scatter plot to show relationship for Urban poulation and Renewable energy
consumption in United States
"""
plt.style.use('ggplot')
plt.scatter(Urban_pop_ind['United States'], Renew_Energy_Cons['United States'])
plt.title('Relationship between Urban population and Renewable energy consumption in United States')
plt.xlabel('Urban population')
plt.ylabel('Renewable energy consumption')
plt.savefig('scatter plot United States.png', dpi=300)
plt.show()

"""
Plotting a scatter plot to show relationship for CO2 emmission and Renewable energy
consumption in United States
"""
plt.style.use('ggplot')
plt.scatter(Renew_Energy_Cons['China'], CO2_ind['China'])
plt.title('Relationship between CO2 emmision & Renewable energy consumption in China')
plt.xlabel('Renewable energy consumption')
plt.ylabel('CO2 emmission')
plt.savefig('scatter plot China.png', dpi=300)
plt.show()


"""
Plotting a heatmap to show the correlation between the selected indicators in India
"""
# crating a DataFrame to from the dictionaries containing the different indicators
India_country = pd.DataFrame({'CO2 emmission': CO2_ind['India'], 'Urban population': Urban_pop_ind['India'], \
                              'Electricity Production from nuclear sources': Elect_prod_nuclear['India'], \
                              'Renewable Energy consumption': Renew_Energy_Cons['India']}, \
                             index=['1990', '1995', '2000', '2005', '2010', '2015', '2019'])


India_country.corr()
#plotting the heatmap
plt.figure(figsize=(8,5))
sns.heatmap(India_country.corr(), annot=True, cmap='Blues')
plt.title('Correlation heatmap for India')
plt.savefig('Heatmap China.png', dpi=300)
plt.show()
