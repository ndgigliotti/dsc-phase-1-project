![title](images/bluescreen.jpg)

# Movie Genre Profitability for Microsoft

**Author:** Nicholas Gigliotti
***

## Overview

I conduct an analysis of the profitability of different movie genres in relation to production budget for Microsoft. Microsoft wants to enter the movie business and develop original content. They will have to decide which genres they wish to invest in early on, since different genres have different production requirements. I conclude that Microsoft should invest in horror for low-budget productions and animation for high-budget productions. Horror has the strongest correlation with return on investment (ROI) of any genre, overall. I further conclude that Microsoft should stay away from drama, action, and crime movies because these are negatively correlated with ROI.

## Business Problem

Microsoft has decided to enter the movie business and create original material. They want to know what kinds of movies are currently profitable, and they want concrete, actionable, insights.

In my analysis, I attempt to answer the following questions for Microsoft:

1. What genres have the strongest correlation with return on investment?
2. How does budget affect these correlations?
3. Are high or low-budget films more profitable?

### Why Genre?
Different film genres have different markets, and need to be created by different groups of artists. Choosing which genres to invest in is one of the most fundamental early decisions Microsoft will have to make.

## Data Understanding

I use data from two sources in my analysis: [The Numbers](https://www.the-numbers.com/movie/budgets "The Numbers") and the [Internet Movie Database](https://www.imdb.com/interfaces/ "Internet Movie Database") (IMDb). IMDb is an expansive and easily accessible source of movie data which, most importantly, includes genre labels for thousands of films. IMDb lacks financial data, however, so I am forced to rely on The Numbers.


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

import utils
import cleaning
import plotting

%matplotlib inline
sns.set(font_scale=1.25)
pd.options.display.float_format = '{:,.2f}'.format
```

### The Numbers
My financial data comes from a website called "The Numbers" which has a healthy collection of production budget and revenue data. The Numbers is owned by Nash Information Services, a movie industry research and consulting firm. The most important columns for my analysis are `production_budget`, `domestic_gross`, and `worldwide_gross`. I use these columns later to calculate profit and return on investment (ROI).

The table includes a little under 6,000 observations.


```python
tn = pd.read_csv(os.path.join('zippedData', 'tn.movie_budgets.csv.gz'),
                 parse_dates=['release_date'])
tn
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2009-12-18</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-05-20</td>
      <td>Pirates of the Caribbean: On Stranger Tides</td>
      <td>$410,600,000</td>
      <td>$241,063,875</td>
      <td>$1,045,663,875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2019-06-07</td>
      <td>Dark Phoenix</td>
      <td>$350,000,000</td>
      <td>$42,762,350</td>
      <td>$149,762,350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2015-05-01</td>
      <td>Avengers: Age of Ultron</td>
      <td>$330,600,000</td>
      <td>$459,005,868</td>
      <td>$1,403,013,963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2017-12-15</td>
      <td>Star Wars Ep. VIII: The Last Jedi</td>
      <td>$317,000,000</td>
      <td>$620,181,382</td>
      <td>$1,316,721,747</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5777</th>
      <td>78</td>
      <td>2018-12-31</td>
      <td>Red 11</td>
      <td>$7,000</td>
      <td>$0</td>
      <td>$0</td>
    </tr>
    <tr>
      <th>5778</th>
      <td>79</td>
      <td>1999-04-02</td>
      <td>Following</td>
      <td>$6,000</td>
      <td>$48,482</td>
      <td>$240,495</td>
    </tr>
    <tr>
      <th>5779</th>
      <td>80</td>
      <td>2005-07-13</td>
      <td>Return to the Land of Wonders</td>
      <td>$5,000</td>
      <td>$1,338</td>
      <td>$1,338</td>
    </tr>
    <tr>
      <th>5780</th>
      <td>81</td>
      <td>2015-09-29</td>
      <td>A Plague So Pleasant</td>
      <td>$1,400</td>
      <td>$0</td>
      <td>$0</td>
    </tr>
    <tr>
      <th>5781</th>
      <td>82</td>
      <td>2005-08-05</td>
      <td>My Date With Drew</td>
      <td>$1,100</td>
      <td>$181,041</td>
      <td>$181,041</td>
    </tr>
  </tbody>
</table>
<p>5782 rows × 6 columns</p>
</div>



### Internet Movie Database
My genre data comes from IMDb, a subsidiary of Amazon which is a well known source of movie information. Naturally, the most important column for my analysis will be `genres`. I later use this column to compute Pearson correlations between genres and different financial statistics.

This table is much larger than `tn`, with a little over 146,000 observations.


```python
imdb = pd.read_csv(os.path.join('zippedData', 'imdb.title.basics.csv.gz'))
imdb
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.00</td>
      <td>Action,Crime,Drama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.00</td>
      <td>Biography,Drama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0069049</td>
      <td>The Other Side of the Wind</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>122.00</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0069204</td>
      <td>Sabse Bada Sukh</td>
      <td>Sabse Bada Sukh</td>
      <td>2018</td>
      <td>nan</td>
      <td>Comedy,Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0100275</td>
      <td>The Wandering Soap Opera</td>
      <td>La Telenovela Errante</td>
      <td>2017</td>
      <td>80.00</td>
      <td>Comedy,Drama,Fantasy</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>146139</th>
      <td>tt9916538</td>
      <td>Kuambil Lagi Hatiku</td>
      <td>Kuambil Lagi Hatiku</td>
      <td>2019</td>
      <td>123.00</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>146140</th>
      <td>tt9916622</td>
      <td>Rodolpho Teóphilo - O Legado de um Pioneiro</td>
      <td>Rodolpho Teóphilo - O Legado de um Pioneiro</td>
      <td>2015</td>
      <td>nan</td>
      <td>Documentary</td>
    </tr>
    <tr>
      <th>146141</th>
      <td>tt9916706</td>
      <td>Dankyavar Danka</td>
      <td>Dankyavar Danka</td>
      <td>2013</td>
      <td>nan</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>146142</th>
      <td>tt9916730</td>
      <td>6 Gunn</td>
      <td>6 Gunn</td>
      <td>2017</td>
      <td>116.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>146143</th>
      <td>tt9916754</td>
      <td>Chico Albuquerque - Revelações</td>
      <td>Chico Albuquerque - Revelações</td>
      <td>2013</td>
      <td>nan</td>
      <td>Documentary</td>
    </tr>
  </tbody>
</table>
<p>146144 rows × 6 columns</p>
</div>



## Data Preparation

Describe and justify the process for preparing the data for analysis.

***
Questions to consider:
* Were there variables you dropped or created?
* How did you address missing values or outliers?
* Why are these choices appropriate given the data and the business problem?
***

### The Numbers
I start by replacing the incorrect `id` column with a column of genuinely unique ID numbers. I also create a `release_year` column, because it will come in handy later when merging tables.


```python
del tn['id']
tn.insert(0, 'tn_id', np.arange(tn.shape[0]) + 1)
tn.insert(2, 'release_year', tn['release_date'].dt.year)
tn.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2009-12-18</td>
      <td>2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-05-20</td>
      <td>2011</td>
      <td>Pirates of the Caribbean: On Stranger Tides</td>
      <td>$410,600,000</td>
      <td>$241,063,875</td>
      <td>$1,045,663,875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2019-06-07</td>
      <td>2019</td>
      <td>Dark Phoenix</td>
      <td>$350,000,000</td>
      <td>$42,762,350</td>
      <td>$149,762,350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2015-05-01</td>
      <td>2015</td>
      <td>Avengers: Age of Ultron</td>
      <td>$330,600,000</td>
      <td>$459,005,868</td>
      <td>$1,403,013,963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2017-12-15</td>
      <td>2017</td>
      <td>Star Wars Ep. VIII: The Last Jedi</td>
      <td>$317,000,000</td>
      <td>$620,181,382</td>
      <td>$1,316,721,747</td>
    </tr>
  </tbody>
</table>
</div>



The columns `production_budget`, `domestic gross`, and `worldwide gross` are in string format, so I remove the extraneous symbols and convert them to `np.float64`.


```python
money_cols = ['production_budget', 'domestic_gross', 'worldwide_gross']
tn[money_cols] = (tn.loc[:, money_cols]
                    .apply(cleaning.process_strings)
                    .apply(lambda x: x.astype('float64')))
tn.sort_values('worldwide_gross').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5037</th>
      <td>5038</td>
      <td>2019-04-23</td>
      <td>2019</td>
      <td>Living Dark: The Story of Ted the Caver</td>
      <td>1,750,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3975</th>
      <td>3976</td>
      <td>2015-05-15</td>
      <td>2015</td>
      <td>Pound of Flesh</td>
      <td>7,500,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4627</th>
      <td>4628</td>
      <td>2011-06-28</td>
      <td>2011</td>
      <td>2:13</td>
      <td>3,500,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4628</th>
      <td>4629</td>
      <td>2013-01-29</td>
      <td>2013</td>
      <td>Batman: The Dark Knight Returns, Part 2</td>
      <td>3,500,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3947</th>
      <td>3948</td>
      <td>2019-06-21</td>
      <td>2019</td>
      <td>Burn Your Maps</td>
      <td>8,000,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



These 0 values for `domestic_gross` and `worldwide_gross` look very suspicious. Some of these 0s are for Netflix original productions such as *Bright* and *The Ridiculous 6*. Obviously those should not be counted as massive commercial failures simply because they were not released in theaters. Other 0s are for movies like *PLAYMOBIL*, which other sources report as generating revenue. Still other 0s are for movies which were released only domestically or only abroad.


```python
tn.query('(domestic_gross == 0) & (worldwide_gross == 0)').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>194</th>
      <td>195</td>
      <td>2020-12-31</td>
      <td>2020</td>
      <td>Moonfall</td>
      <td>150,000,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>479</th>
      <td>480</td>
      <td>2017-12-13</td>
      <td>2017</td>
      <td>Bright</td>
      <td>90,000,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>480</th>
      <td>481</td>
      <td>2019-12-31</td>
      <td>2019</td>
      <td>Army of the Dead</td>
      <td>90,000,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>535</th>
      <td>536</td>
      <td>2020-02-21</td>
      <td>2020</td>
      <td>Call of the Wild</td>
      <td>82,000,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>670</th>
      <td>671</td>
      <td>2019-08-30</td>
      <td>2019</td>
      <td>PLAYMOBIL</td>
      <td>75,000,000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



I remove any rows where the domestic or worldwide gross is 0, since nearly every 0 is a null value or error.


```python
tn = tn.loc[tn.query('(domestic_gross > 0) & (worldwide_gross > 0)').index]
tn.sort_values('worldwide_gross').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5770</th>
      <td>5771</td>
      <td>2008-08-14</td>
      <td>2008</td>
      <td>The Rise and Fall of Miss Thang</td>
      <td>10,000.00</td>
      <td>401.00</td>
      <td>401.00</td>
    </tr>
    <tr>
      <th>5518</th>
      <td>5519</td>
      <td>2005-10-13</td>
      <td>2005</td>
      <td>The Dark Hours</td>
      <td>400,000.00</td>
      <td>423.00</td>
      <td>423.00</td>
    </tr>
    <tr>
      <th>5769</th>
      <td>5770</td>
      <td>1996-04-01</td>
      <td>1996</td>
      <td>Bang</td>
      <td>10,000.00</td>
      <td>527.00</td>
      <td>527.00</td>
    </tr>
    <tr>
      <th>5466</th>
      <td>5467</td>
      <td>2018-05-11</td>
      <td>2018</td>
      <td>Higher Power</td>
      <td>500,000.00</td>
      <td>528.00</td>
      <td>528.00</td>
    </tr>
    <tr>
      <th>5027</th>
      <td>5028</td>
      <td>1993-01-01</td>
      <td>1993</td>
      <td>Ed and his Dead Mother</td>
      <td>1,800,000.00</td>
      <td>673.00</td>
      <td>673.00</td>
    </tr>
  </tbody>
</table>
</div>



Looks like the data extends back in time much farther than I want.


```python
ax = sns.histplot(data=tn, x='release_year', bins=20, palette='deep')
ax.set_title('Distribution of `release_year`')
ax.set_xlabel('Year')
```




    Text(0.5, 0, 'Year')




    
![svg](/images/output_20_1.svg)
    


I drop everything earlier than 2009 because I'm only interested in data that's relevant to current box office performance. 2020 was a particularly bad year because of the COVID-19 pandemic, so I leave that out as well.


```python
tn = tn.loc[tn.query('(release_year <= 2019) & (release_year >= 2009)').index]
tn.sort_values('release_date').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2934</th>
      <td>2935</td>
      <td>2009-01-09</td>
      <td>2009</td>
      <td>The Unborn</td>
      <td>16,000,000.00</td>
      <td>42,670,410.00</td>
      <td>78,208,812.00</td>
    </tr>
    <tr>
      <th>4318</th>
      <td>4319</td>
      <td>2009-01-09</td>
      <td>2009</td>
      <td>Not Easily Broken</td>
      <td>5,000,000.00</td>
      <td>10,572,742.00</td>
      <td>10,732,909.00</td>
    </tr>
    <tr>
      <th>1880</th>
      <td>1881</td>
      <td>2009-01-09</td>
      <td>2009</td>
      <td>Bride Wars</td>
      <td>30,000,000.00</td>
      <td>58,715,510.00</td>
      <td>115,150,424.00</td>
    </tr>
    <tr>
      <th>1164</th>
      <td>1165</td>
      <td>2009-01-16</td>
      <td>2009</td>
      <td>Defiance</td>
      <td>50,000,000.00</td>
      <td>28,644,813.00</td>
      <td>52,987,754.00</td>
    </tr>
    <tr>
      <th>2736</th>
      <td>2737</td>
      <td>2009-01-16</td>
      <td>2009</td>
      <td>Notorious</td>
      <td>19,000,000.00</td>
      <td>36,843,682.00</td>
      <td>44,972,183.00</td>
    </tr>
  </tbody>
</table>
</div>



Looks like all of the basic money distributions are very right-skewed, which is not surprising. I expect there to be many more small films than big films, financially-speaking.


```python
plotting.multi_hist(tn, include=money_cols, xlabel='Dollars', palette='deep')
```




    array([<AxesSubplot:title={'center':'Distribution of `production_budget`'}, xlabel='Dollars', ylabel='Count'>,
           <AxesSubplot:title={'center':'Distribution of `domestic_gross`'}, xlabel='Dollars', ylabel='Count'>,
           <AxesSubplot:title={'center':'Distribution of `worldwide_gross`'}, xlabel='Dollars', ylabel='Count'>],
          dtype=object)




    
![svg](/images/output_24_1.svg)
    


These box plots indicate that there are many extreme values in the dataset. The data points beyond the upper whiskers are not truly outliers in this case. *Avatar* really does have a worldwide gross of 2.8 billion dollars. There is not a good scientific reason to altar or remove these values.


```python
fix, ax = plt.subplots(figsize=(10, 8))
ax = sns.boxplot(data=tn[money_cols],
                 ax=ax,
                palette='deep')
ax.set_title('Distributions of Budgets and Grosses')
ax.set_ylabel('Dollars (billions)')
```




    Text(0, 0.5, 'Dollars (billions)')




    
![svg](/images/output_26_1.svg)
    


#### Financial Calculations
I calculate domestic and worldwide profit by subtracting `production_budget` from each respective gross column.


```python
tn['worldwide_profit'] = tn.eval('worldwide_gross - production_budget')
tn['domestic_profit'] = tn.eval('domestic_gross - production_budget')
tn.sort_values('worldwide_profit', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>worldwide_profit</th>
      <th>domestic_profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2009-12-18</td>
      <td>2009</td>
      <td>Avatar</td>
      <td>425,000,000.00</td>
      <td>760,507,625.00</td>
      <td>2,776,345,279.00</td>
      <td>2,351,345,279.00</td>
      <td>335,507,625.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>2018-04-27</td>
      <td>2018</td>
      <td>Avengers: Infinity War</td>
      <td>300,000,000.00</td>
      <td>678,815,482.00</td>
      <td>2,048,134,200.00</td>
      <td>1,748,134,200.00</td>
      <td>378,815,482.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2015-12-18</td>
      <td>2015</td>
      <td>Star Wars Ep. VII: The Force Awakens</td>
      <td>306,000,000.00</td>
      <td>936,662,225.00</td>
      <td>2,053,311,220.00</td>
      <td>1,747,311,220.00</td>
      <td>630,662,225.00</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>2015-06-12</td>
      <td>2015</td>
      <td>Jurassic World</td>
      <td>215,000,000.00</td>
      <td>652,270,625.00</td>
      <td>1,648,854,864.00</td>
      <td>1,433,854,864.00</td>
      <td>437,270,625.00</td>
    </tr>
    <tr>
      <th>66</th>
      <td>67</td>
      <td>2015-04-03</td>
      <td>2015</td>
      <td>Furious 7</td>
      <td>190,000,000.00</td>
      <td>353,007,020.00</td>
      <td>1,518,722,794.00</td>
      <td>1,328,722,794.00</td>
      <td>163,007,020.00</td>
    </tr>
  </tbody>
</table>
</div>



The distribution of `domestic_profit` is almost symmetrical around 0, although it is still right-skewed overall. The distribution of `worldwide_profit` is even more right-skewed. In both distributions the positive skew indicates that there are more winners than losers. This is unsurprising, since production companies strive to generate profit.


```python
ax = plotting.multi_hist(tn,
                         include=['domestic_profit', 'worldwide_profit'],
                         xlabel='Dollars',
                         bins=100)
```


    
![svg](/images/output_30_0.svg)
    


I calculate the percent return on investment (ROI) by dividing profit by budget and multiplying by 100. The sorted result is... ominous...


```python
tn['worldwide_roi'] = tn.eval('(worldwide_profit / production_budget) * 100')
tn['domestic_roi'] = tn.eval('(domestic_profit / production_budget) * 100')
tn.sort_values('worldwide_roi', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>worldwide_profit</th>
      <th>domestic_profit</th>
      <th>worldwide_roi</th>
      <th>domestic_roi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5492</th>
      <td>5493</td>
      <td>2009-09-25</td>
      <td>2009</td>
      <td>Paranormal Activity</td>
      <td>450,000.00</td>
      <td>107,918,810.00</td>
      <td>194,183,034.00</td>
      <td>193,733,034.00</td>
      <td>107,468,810.00</td>
      <td>43,051.79</td>
      <td>23,881.96</td>
    </tr>
    <tr>
      <th>5679</th>
      <td>5680</td>
      <td>2015-07-10</td>
      <td>2015</td>
      <td>The Gallows</td>
      <td>100,000.00</td>
      <td>22,764,410.00</td>
      <td>41,656,474.00</td>
      <td>41,556,474.00</td>
      <td>22,664,410.00</td>
      <td>41,556.47</td>
      <td>22,664.41</td>
    </tr>
    <tr>
      <th>5211</th>
      <td>5212</td>
      <td>2012-01-06</td>
      <td>2012</td>
      <td>The Devil Inside</td>
      <td>1,000,000.00</td>
      <td>53,262,945.00</td>
      <td>101,759,490.00</td>
      <td>100,759,490.00</td>
      <td>52,262,945.00</td>
      <td>10,075.95</td>
      <td>5,226.29</td>
    </tr>
    <tr>
      <th>5459</th>
      <td>5460</td>
      <td>2009-04-23</td>
      <td>2009</td>
      <td>Home</td>
      <td>500,000.00</td>
      <td>15,433.00</td>
      <td>44,793,168.00</td>
      <td>44,293,168.00</td>
      <td>-484,567.00</td>
      <td>8,858.63</td>
      <td>-96.91</td>
    </tr>
    <tr>
      <th>5062</th>
      <td>5063</td>
      <td>2011-04-01</td>
      <td>2011</td>
      <td>Insidious</td>
      <td>1,500,000.00</td>
      <td>54,009,150.00</td>
      <td>99,870,886.00</td>
      <td>98,370,886.00</td>
      <td>52,509,150.00</td>
      <td>6,558.06</td>
      <td>3,500.61</td>
    </tr>
  </tbody>
</table>
</div>



The following is a box plot of `domestic_roi` and `worldwide_roi` plotted on a logarithmic scale. Interestingly, `domestic_roi` is heavily clustered under 100%, whereas the upper quartile of `worldwide_roi` is much higher. This is probably because production companies focus on the worldwide market nowadays.

There are a number of extreme values beyond the upper whiskers, but as you can see in the previous cell, these are just extremely successful horror movies. There is not a good scientific reason to altar or remove these data points.


```python
fix, ax = plt.subplots(figsize=(10, 8))
ax = sns.boxplot(data=tn[['domestic_roi', 'worldwide_roi']],
                 ax=ax,
                 palette='deep')
ax.set_title('Distributions of Return on Investment')
ax.set_ylabel(None)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
```


    
![svg](/images/output_34_0.svg)
    


Looks like there are some duplicate titles under `movie`, but those rows turn out to be acceptable.


```python
cleaning.info(tn)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dup</th>
      <th>dup_%</th>
      <th>nan</th>
      <th>nan_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tn_id</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>release_date</th>
      <td>1329</td>
      <td>66.72</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>release_year</th>
      <td>1981</td>
      <td>99.45</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>movie</th>
      <td>4</td>
      <td>0.20</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>production_budget</th>
      <td>1687</td>
      <td>84.69</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>domestic_gross</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>worldwide_gross</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>worldwide_profit</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>domestic_profit</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>worldwide_roi</th>
      <td>1</td>
      <td>0.05</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>domestic_roi</th>
      <td>1</td>
      <td>0.05</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
tn[tn[['movie']].duplicated(keep=False)].sort_values('movie')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>worldwide_profit</th>
      <th>domestic_profit</th>
      <th>worldwide_roi</th>
      <th>domestic_roi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2140</th>
      <td>2141</td>
      <td>2009-12-04</td>
      <td>2009</td>
      <td>Brothers</td>
      <td>26,000,000.00</td>
      <td>28,544,157.00</td>
      <td>45,043,870.00</td>
      <td>19,043,870.00</td>
      <td>2,544,157.00</td>
      <td>73.25</td>
      <td>9.79</td>
    </tr>
    <tr>
      <th>3307</th>
      <td>3308</td>
      <td>2015-08-14</td>
      <td>2015</td>
      <td>Brothers</td>
      <td>13,000,000.00</td>
      <td>656,688.00</td>
      <td>17,856,688.00</td>
      <td>4,856,688.00</td>
      <td>-12,343,312.00</td>
      <td>37.36</td>
      <td>-94.95</td>
    </tr>
    <tr>
      <th>243</th>
      <td>244</td>
      <td>2015-03-27</td>
      <td>2015</td>
      <td>Home</td>
      <td>130,000,000.00</td>
      <td>177,397,510.00</td>
      <td>385,997,896.00</td>
      <td>255,997,896.00</td>
      <td>47,397,510.00</td>
      <td>196.92</td>
      <td>36.46</td>
    </tr>
    <tr>
      <th>5459</th>
      <td>5460</td>
      <td>2009-04-23</td>
      <td>2009</td>
      <td>Home</td>
      <td>500,000.00</td>
      <td>15,433.00</td>
      <td>44,793,168.00</td>
      <td>44,293,168.00</td>
      <td>-484,567.00</td>
      <td>8,858.63</td>
      <td>-96.91</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39</td>
      <td>2010-05-14</td>
      <td>2010</td>
      <td>Robin Hood</td>
      <td>210,000,000.00</td>
      <td>105,487,148.00</td>
      <td>322,459,006.00</td>
      <td>112,459,006.00</td>
      <td>-104,512,852.00</td>
      <td>53.55</td>
      <td>-49.77</td>
    </tr>
    <tr>
      <th>408</th>
      <td>409</td>
      <td>2018-11-21</td>
      <td>2018</td>
      <td>Robin Hood</td>
      <td>99,000,000.00</td>
      <td>30,824,628.00</td>
      <td>84,747,441.00</td>
      <td>-14,252,559.00</td>
      <td>-68,175,372.00</td>
      <td>-14.40</td>
      <td>-68.86</td>
    </tr>
    <tr>
      <th>5009</th>
      <td>5010</td>
      <td>2010-04-09</td>
      <td>2010</td>
      <td>The Square</td>
      <td>1,900,000.00</td>
      <td>406,216.00</td>
      <td>740,932.00</td>
      <td>-1,159,068.00</td>
      <td>-1,493,784.00</td>
      <td>-61.00</td>
      <td>-78.62</td>
    </tr>
    <tr>
      <th>5099</th>
      <td>5100</td>
      <td>2013-10-25</td>
      <td>2013</td>
      <td>The Square</td>
      <td>1,500,000.00</td>
      <td>124,244.00</td>
      <td>176,262.00</td>
      <td>-1,323,738.00</td>
      <td>-1,375,756.00</td>
      <td>-88.25</td>
      <td>-91.72</td>
    </tr>
  </tbody>
</table>
</div>



Time to save the data and move on.


```python
tn.to_json(os.path.join('cleanData', 'tn.profit.json'))
```

### Internet Movie Database
After taking a look at my cleaning report, I can see that there are a number of duplicates under `primary_title` and many null values under `runtime_minutes`. I deal with the duplicates first, and later drop the `runtime_minutes` column altogether.


```python
cleaning.info(imdb)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dup</th>
      <th>dup_%</th>
      <th>nan</th>
      <th>nan_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>runtime_minutes</th>
      <td>145776</td>
      <td>99.75</td>
      <td>31739</td>
      <td>21.72</td>
    </tr>
    <tr>
      <th>genres</th>
      <td>145058</td>
      <td>99.26</td>
      <td>5408</td>
      <td>3.70</td>
    </tr>
    <tr>
      <th>original_title</th>
      <td>8370</td>
      <td>5.73</td>
      <td>21</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>tconst</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>primary_title</th>
      <td>10073</td>
      <td>6.89</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>start_year</th>
      <td>146125</td>
      <td>99.99</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



These duplicates are indeed going to be a problem.


```python
imdb[imdb[['primary_title', 'original_title', 'start_year']
          ].duplicated(keep=False)].sort_values('primary_title')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>103890</th>
      <td>tt6085916</td>
      <td>(aguirre)</td>
      <td>(aguirre)</td>
      <td>2016</td>
      <td>97.00</td>
      <td>Biography,Documentary</td>
    </tr>
    <tr>
      <th>106201</th>
      <td>tt6214664</td>
      <td>(aguirre)</td>
      <td>(aguirre)</td>
      <td>2016</td>
      <td>98.00</td>
      <td>Biography,Comedy,Documentary</td>
    </tr>
    <tr>
      <th>129962</th>
      <td>tt8032828</td>
      <td>100 Milioni di bracciate</td>
      <td>100 Milioni di bracciate</td>
      <td>2017</td>
      <td>nan</td>
      <td>Biography</td>
    </tr>
    <tr>
      <th>129979</th>
      <td>tt8034014</td>
      <td>100 Milioni di bracciate</td>
      <td>100 Milioni di bracciate</td>
      <td>2017</td>
      <td>nan</td>
      <td>Biography</td>
    </tr>
    <tr>
      <th>20394</th>
      <td>tt1855110</td>
      <td>180</td>
      <td>180</td>
      <td>2011</td>
      <td>121.00</td>
      <td>Drama,Romance</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>66990</th>
      <td>tt3815124</td>
      <td>Ângelo de Sousa - Tudo o Que Sou Capaz</td>
      <td>Ângelo de Sousa - Tudo o Que Sou Capaz</td>
      <td>2010</td>
      <td>60.00</td>
      <td>Biography,Documentary</td>
    </tr>
    <tr>
      <th>66992</th>
      <td>tt3815128</td>
      <td>Ângelo de Sousa - Tudo o Que Sou Capaz</td>
      <td>Ângelo de Sousa - Tudo o Que Sou Capaz</td>
      <td>2010</td>
      <td>60.00</td>
      <td>Biography,Documentary</td>
    </tr>
    <tr>
      <th>66995</th>
      <td>tt3815134</td>
      <td>Ângelo de Sousa - Tudo o Que Sou Capaz</td>
      <td>Ângelo de Sousa - Tudo o Que Sou Capaz</td>
      <td>2010</td>
      <td>60.00</td>
      <td>Biography,Documentary</td>
    </tr>
    <tr>
      <th>92592</th>
      <td>tt5352034</td>
      <td>Çagrilan</td>
      <td>Çagrilan</td>
      <td>2016</td>
      <td>85.00</td>
      <td>Horror</td>
    </tr>
    <tr>
      <th>109103</th>
      <td>tt6412726</td>
      <td>Çagrilan</td>
      <td>Çagrilan</td>
      <td>2016</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3031 rows × 6 columns</p>
</div>



I drop rows with duplicates across `primary_title`, `original_title`, and `start_year`.


```python
imdb.drop_duplicates(
    subset=['primary_title', 'original_title', 'start_year'], inplace=True)
```

Next I preprocess the titles of both `imdb` and `tn` in preparation for the merge. Since these tables do not share a unique identifier, I have to merge them using the year and title fields.

My string processing function makes all characters lowercase, removes punctuation, and translates Unicode characters to ASCII.


```python
imdb['clean_title'] = cleaning.process_strings(imdb.loc[:, 'primary_title'])
tn = tn.assign(clean_title=cleaning.process_strings(tn['movie']))
```

I merge the tables crudely along the year and title fields. While this merge is sufficient for my analysis, it is inefficient. Some movies are lost in translation because their titles do not match character-for-character between tables.


```python
imdb = pd.merge(imdb,
                tn,
                how='inner',
                left_on=['start_year', 'clean_title'],
                right_on=['release_year', 'clean_title'])
display(imdb.shape)
imdb.head()
```


    (1387, 18)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>clean_title</th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>worldwide_profit</th>
      <th>domestic_profit</th>
      <th>worldwide_roi</th>
      <th>domestic_roi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>2013</td>
      <td>114.00</td>
      <td>Adventure,Comedy,Drama</td>
      <td>the secret life of walter mitty</td>
      <td>437</td>
      <td>2013-12-25</td>
      <td>2013</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>91,000,000.00</td>
      <td>58,236,838.00</td>
      <td>187,861,183.00</td>
      <td>96,861,183.00</td>
      <td>-32,763,162.00</td>
      <td>106.44</td>
      <td>-36.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0365907</td>
      <td>A Walk Among the Tombstones</td>
      <td>A Walk Among the Tombstones</td>
      <td>2014</td>
      <td>114.00</td>
      <td>Action,Crime,Drama</td>
      <td>a walk among the tombstones</td>
      <td>2067</td>
      <td>2014-09-19</td>
      <td>2014</td>
      <td>A Walk Among the Tombstones</td>
      <td>28,000,000.00</td>
      <td>26,017,685.00</td>
      <td>62,108,587.00</td>
      <td>34,108,587.00</td>
      <td>-1,982,315.00</td>
      <td>121.82</td>
      <td>-7.08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0369610</td>
      <td>Jurassic World</td>
      <td>Jurassic World</td>
      <td>2015</td>
      <td>124.00</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>jurassic world</td>
      <td>34</td>
      <td>2015-06-12</td>
      <td>2015</td>
      <td>Jurassic World</td>
      <td>215,000,000.00</td>
      <td>652,270,625.00</td>
      <td>1,648,854,864.00</td>
      <td>1,433,854,864.00</td>
      <td>437,270,625.00</td>
      <td>666.91</td>
      <td>203.38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0376136</td>
      <td>The Rum Diary</td>
      <td>The Rum Diary</td>
      <td>2011</td>
      <td>119.00</td>
      <td>Comedy,Drama</td>
      <td>the rum diary</td>
      <td>1316</td>
      <td>2011-10-28</td>
      <td>2011</td>
      <td>The Rum Diary</td>
      <td>45,000,000.00</td>
      <td>13,109,815.00</td>
      <td>21,544,732.00</td>
      <td>-23,455,268.00</td>
      <td>-31,890,185.00</td>
      <td>-52.12</td>
      <td>-70.87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0383010</td>
      <td>The Three Stooges</td>
      <td>The Three Stooges</td>
      <td>2012</td>
      <td>92.00</td>
      <td>Comedy,Family</td>
      <td>the three stooges</td>
      <td>1904</td>
      <td>2012-04-13</td>
      <td>2012</td>
      <td>The Three Stooges</td>
      <td>30,000,000.00</td>
      <td>44,338,224.00</td>
      <td>54,052,249.00</td>
      <td>24,052,249.00</td>
      <td>14,338,224.00</td>
      <td>80.17</td>
      <td>47.79</td>
    </tr>
  </tbody>
</table>
</div>



Looks like the `start_year` range is appropriate. There is a large spike around 2010, which is not ideal. Unfortunately, I am working with a pretty small dataset at this point (~1400 observations), so I am reluctant to discard these early years.


```python
ax = sns.histplot(imdb, x='start_year', bins=8, palette='deep')
ax.set_title('Distribution of `start_year`')
ax.set_xlabel('Year')
```




    Text(0.5, 0, 'Year')




    
![svg](/images/output_51_1.svg)
    


Next I drop all irrelevant or extraneous columns and check again for nulls and duplicates.


```python
imdb.drop(columns=['start_year', 'release_year', 'clean_title',
          'movie', 'original_title', 'runtime_minutes'], inplace=True)
```


```python
cleaning.info(imdb)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dup</th>
      <th>dup_%</th>
      <th>nan</th>
      <th>nan_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tconst</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>primary_title</th>
      <td>14</td>
      <td>1.01</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>genres</th>
      <td>1171</td>
      <td>84.43</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>tn_id</th>
      <td>14</td>
      <td>1.01</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>release_date</th>
      <td>820</td>
      <td>59.12</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>production_budget</th>
      <td>1146</td>
      <td>82.62</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>domestic_gross</th>
      <td>14</td>
      <td>1.01</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>worldwide_gross</th>
      <td>14</td>
      <td>1.01</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>worldwide_profit</th>
      <td>14</td>
      <td>1.01</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>domestic_profit</th>
      <td>14</td>
      <td>1.01</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>worldwide_roi</th>
      <td>15</td>
      <td>1.08</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>domestic_roi</th>
      <td>15</td>
      <td>1.08</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



Everything looks to be in order, but I need to convert the `genres` column from `string` to `list` in order to pull apart the individual genre labels.


```python
imdb['genres'] = imdb.loc[:, 'genres'].str.split(',')
imdb[['genres']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Adventure, Comedy, Drama]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Action, Crime, Drama]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Action, Adventure, Sci-Fi]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Comedy, Drama]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Comedy, Family]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1382</th>
      <td>[Horror, Thriller]</td>
    </tr>
    <tr>
      <th>1383</th>
      <td>[Crime, Drama, Thriller]</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>[Drama, Horror, Mystery]</td>
    </tr>
    <tr>
      <th>1385</th>
      <td>[Documentary]</td>
    </tr>
    <tr>
      <th>1386</th>
      <td>[Biography, Drama]</td>
    </tr>
  </tbody>
</table>
<p>1387 rows × 1 columns</p>
</div>



Time to inspect the distribution of genres.


```python
imdb.explode('genres')['genres'].value_counts()
```




    Drama          671
    Comedy         488
    Action         421
    Adventure      345
    Thriller       235
    Crime          214
    Romance        183
    Horror         150
    Biography      129
    Sci-Fi         129
    Fantasy        120
    Mystery        115
    Animation      100
    Family          87
    Music           50
    History         39
    Documentary     35
    Sport           32
    War             17
    Western         10
    Musical          9
    Name: genres, dtype: int64



Inspecting the movies in the "Music" genre reveals that they are in fact musicals. I collapse these two labels into "Musical".


```python
imdb.explode('genres').query('genres == "Music"').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primary_title</th>
      <th>genres</th>
      <th>tn_id</th>
      <th>release_date</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>worldwide_profit</th>
      <th>domestic_profit</th>
      <th>worldwide_roi</th>
      <th>domestic_roi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>tt0475290</td>
      <td>Hail, Caesar!</td>
      <td>Music</td>
      <td>2422</td>
      <td>2016-02-05</td>
      <td>22,000,000.00</td>
      <td>30,080,225.00</td>
      <td>64,160,680.00</td>
      <td>42,160,680.00</td>
      <td>8,080,225.00</td>
      <td>191.64</td>
      <td>36.73</td>
    </tr>
    <tr>
      <th>128</th>
      <td>tt1017451</td>
      <td>The Runaways</td>
      <td>Music</td>
      <td>3757</td>
      <td>2010-03-19</td>
      <td>9,500,000.00</td>
      <td>3,573,673.00</td>
      <td>5,278,632.00</td>
      <td>-4,221,368.00</td>
      <td>-5,926,327.00</td>
      <td>-44.44</td>
      <td>-62.38</td>
    </tr>
    <tr>
      <th>152</th>
      <td>tt1068242</td>
      <td>Footloose</td>
      <td>Music</td>
      <td>2339</td>
      <td>2011-10-14</td>
      <td>24,000,000.00</td>
      <td>51,802,742.00</td>
      <td>62,989,834.00</td>
      <td>38,989,834.00</td>
      <td>27,802,742.00</td>
      <td>162.46</td>
      <td>115.84</td>
    </tr>
    <tr>
      <th>170</th>
      <td>tt1126591</td>
      <td>Burlesque</td>
      <td>Music</td>
      <td>1024</td>
      <td>2010-11-24</td>
      <td>55,000,000.00</td>
      <td>39,440,655.00</td>
      <td>90,552,675.00</td>
      <td>35,552,675.00</td>
      <td>-15,559,345.00</td>
      <td>64.64</td>
      <td>-28.29</td>
    </tr>
    <tr>
      <th>195</th>
      <td>tt1193631</td>
      <td>Step Up 3D</td>
      <td>Music</td>
      <td>1909</td>
      <td>2010-08-06</td>
      <td>30,000,000.00</td>
      <td>42,400,223.00</td>
      <td>165,889,117.00</td>
      <td>135,889,117.00</td>
      <td>12,400,223.00</td>
      <td>452.96</td>
      <td>41.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
imdb['genres'] = utils.map_list_likes(
    imdb['genres'], lambda x: 'Musical' if x == 'Music' else x)
imdb.explode('genres')['genres'].value_counts()
```




    Drama          671
    Comedy         488
    Action         421
    Adventure      345
    Thriller       235
    Crime          214
    Romance        183
    Horror         150
    Sci-Fi         129
    Biography      129
    Fantasy        120
    Mystery        115
    Animation      100
    Family          87
    Musical         59
    History         39
    Documentary     35
    Sport           32
    War             17
    Western         10
    Name: genres, dtype: int64



Here is the final genre distribution chart. I choose to keep low-frequency genres like "War" and "Western" unless they prove disruptive.


```python
genre_counts = imdb.explode(
    'genres')['genres'].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(8, 8))
ax = sns.barplot(x=genre_counts.values,
                 y=genre_counts.index, ax=ax, palette='deep')
ax.set_title('Distribution of Genres')
ax.set_xlabel('Percentage of Population')
ax.xaxis.set_major_formatter(ticker.PercentFormatter())
```


    
![svg](/images/output_63_0.svg)
    


Time to save the data and move on.


```python
imdb.to_json(os.path.join('cleanData', 'imdb.tn.basics.json'))
```

 
## Data Modeling
### Cross-Tabulation
I begin by creating a movie-per-movie genre frequency table using cross-tabulation. Since no movie can have more than one of each genre (or less than zero), the frequencies can be interpreted as binary truth values. Now I can compute correlations between genres and financial outcomes.


```python
combos = pd.crosstab(imdb.explode('genres')[
                     'tconst'], imdb.explode('genres')['genres'])
combos = combos.astype(np.bool_)
combos = combos.sort_index(axis=1).sort_index(axis=0)
combos.to_json(os.path.join('precomputed', 'genre_combos.json'))
combos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>genres</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>History</th>
      <th>Horror</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
    <tr>
      <th>tconst</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tt0359950</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>tt0365907</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>tt0369610</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>tt0376136</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>tt0383010</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



I set the index of `imdb` to `tconst` for the upcoming computations. I need to use these unique IDs to relate the rows of `imdb` to the rows of `combos`.


```python
imdb.set_index('tconst', inplace=True)
```

### Calculating Correlation
The **Pearson correlation coefficient** is a measure of the degree to which the relationship between two variables resembles a linear relationship. But it's hard to understand intuitively how genre could have anything approaching a linear relationship with, say, profit. What does that even mean?

It all makes good sense if you consider the following violin plots. The blobs indicate the location and density of the points in the distribution. Notice that genres which are positively correlated with profit have a fat violin on `False` and a narrow violin on `True`. Notice that genres which are negatively correlated with profit have a fat violin on `True` and a narrow violin on `False`. And finally, notice that genres with no correlation with profit have two fat violins.


```python
axes = plotting.boolean_violinplots(
        combos,
        imdb['worldwide_profit'],
        suptitle='Worldwide Profit by Genre',
        include=['Adventure', 'Drama', 'Animation', 'Comedy'],
        ylabel='Dollars (billions)',
        size=3,
        figsize=(10, 10),
        palette='deep')
plt.savefig(os.path.join('images', 'violinplots.jpg'),
        dpi=300,
        format='JPG')
```


    
![svg](/images/output_71_0.svg)
    


### Correlation with Profit
Here are the correlations between each genre and worldwide profit. Notice that the frontrunners are "Adventure", "Animation", "Sci-Fi", and "Action". Also notice that "Drama" has the strongest negative correlation. This is an interesting result.


```python
ax = plotting.cat_correlation(combos, imdb['worldwide_profit'])
ax.set_title('Genre Correlation with Worldwide Profit')
ax.set_ylabel(None)
plt.savefig(os.path.join('images', 'corr_world_profit.jpg'),
            dpi=300,
            format='JPG')
```


    
![svg](/images/output_73_0.svg)
    


As a sanity check, I plot the movies with the highest worldwide profit. Many adventure, sci-fi, and action titles show up: *The Avengers*,  *Jurassic World*, *Black Panther*, *The Dark Knight Rises*. There are also several animated films: *Frozen*, *Beauty and the Beast*, *Incredibles 2*. Looks like the correlation numbers make sense.


```python
reds = sns.color_palette('Reds_r', 40, desat=0.6)
ax = plotting.topn_ranking(imdb, 
                            'primary_title', 
                            'worldwide_profit',
                            20,
                            figsize=(8, 10),
                            palette=reds)
ax.set_title('Highest Profit Movies Worldwide')
ax.set_ylabel(None)
ax.set_xlabel('Dollars (billions)')
plt.savefig(os.path.join('images', 'top_world_profit.jpg'),
            dpi=300, 
            format='JPG')
```


    
![svg](/images/output_75_0.svg)
    


### Correlation with ROI
The correlations with worldwide ROI are strikingly different from those with profit. Horror? Mystery? Thriller? These all had a weak negative correlation with worldwide profit. Why are they suddenly the only positive values?

Here's my conjecture: it's because ROI places heavy weight on budget, and top-earning horror films are often very low-budget. A low-budget film can generate revenue which is exponentially higher than its budget. A high-budget film will have a hard time doing that.

Horror movies have a reputation for being low-budget. *Paranormal Activity*, for example, is well-known for its low budget. *The Blair Witch Project* is another obvious example, since it's just a shaky-cam movie with a bunch of kids in the woods. Nonetheless, both of these movies were highly successful at the box office.


```python
ax = plotting.cat_correlation(combos, imdb['worldwide_roi'])
ax.set_title('Genre Correlation with Worldwide ROI')
ax.set_ylabel(None)
plt.savefig(os.path.join('images', 'corr_world_roi.jpg'),
            dpi=300,
            format='JPG')
```


    
![svg](/images/output_77_0.svg)
    


Here's another sanity check: the highest ROI movies worldwide. Nearly all of them are horror titles.


```python
reds = sns.color_palette('Reds_r', 30, desat=0.6)
ax = plotting.topn_ranking(imdb,
                           'primary_title',
                           'worldwide_roi',
                           15,
                           figsize=(5, 8),
                           palette=reds)
ax.xaxis.set_major_formatter(ticker.PercentFormatter())
ax.set_title('Highest ROI Movies Worldwide')
ax.set_xlabel(None)
ax.set_ylabel(None)
plt.savefig(os.path.join('images', 'top_world_roi.jpg'),
            dpi=300,
            format='JPG')
```


    
![svg](/images/output_79_0.svg)
    


### Effects of Budget
Next, I partition the movies by budget quartile. "Low Budget" refers to the lower quartile (25th percentile) and below. "High Budget" refers to the upper quartile (75th percentile) and above. I want to plot the genre-ROI-correlations for low-budget films alongside those for high-budget films.


```python
quartile_labels = ['Low Budget', 'Mid-Low Budget',
                   'Mid-High Budget', 'High Budget']
imdb['budget_quartile'] = pd.qcut(
    imdb['production_budget'], 4, quartile_labels)
quartile_intervals = pd.qcut(imdb['production_budget'], 4).dtype.categories
world_roi_by_budget = combos.groupby(
    imdb['budget_quartile']).corrwith(imdb['worldwide_roi'])
world_roi_by_budget
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>genres</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>History</th>
      <th>Horror</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
    <tr>
      <th>budget_quartile</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Low Budget</th>
      <td>-0.05</td>
      <td>-0.04</td>
      <td>-0.02</td>
      <td>-0.03</td>
      <td>-0.08</td>
      <td>-0.06</td>
      <td>-0.05</td>
      <td>-0.12</td>
      <td>-0.04</td>
      <td>0.00</td>
      <td>-0.03</td>
      <td>0.25</td>
      <td>-0.04</td>
      <td>0.23</td>
      <td>-0.04</td>
      <td>-0.01</td>
      <td>-0.03</td>
      <td>0.18</td>
      <td>-0.03</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>Mid-Low Budget</th>
      <td>-0.11</td>
      <td>-0.04</td>
      <td>-0.01</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>-0.13</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>0.10</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>-0.05</td>
      <td>0.05</td>
      <td>-0.08</td>
      <td>-0.04</td>
    </tr>
    <tr>
      <th>Mid-High Budget</th>
      <td>-0.02</td>
      <td>-0.04</td>
      <td>-0.02</td>
      <td>-0.02</td>
      <td>-0.03</td>
      <td>-0.12</td>
      <td>-0.05</td>
      <td>-0.04</td>
      <td>-0.04</td>
      <td>-0.06</td>
      <td>-0.01</td>
      <td>0.02</td>
      <td>0.19</td>
      <td>0.03</td>
      <td>0.11</td>
      <td>0.04</td>
      <td>-0.01</td>
      <td>0.10</td>
      <td>-0.06</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>High Budget</th>
      <td>-0.15</td>
      <td>0.15</td>
      <td>0.25</td>
      <td>0.01</td>
      <td>0.14</td>
      <td>-0.07</td>
      <td>-0.02</td>
      <td>-0.12</td>
      <td>-0.03</td>
      <td>-0.07</td>
      <td>-0.02</td>
      <td>-0.10</td>
      <td>0.09</td>
      <td>-0.02</td>
      <td>-0.11</td>
      <td>0.09</td>
      <td>-0.05</td>
      <td>-0.05</td>
      <td>-0.03</td>
      <td>-0.03</td>
    </tr>
  </tbody>
</table>
</div>



Here's a plot of worldwide ROI computed separately for low-budget films and high-budget films. Looks like evidence supporting my conjecture that top-earning horror films are often very low-budget, and that low-budget movies are capable of achieving very high ROI.

Interestingly animation, and not adventure, is the frontrunner for high-budget films. Adventure, which led in correlation with worldwide profit, is now in second place.


```python
plotting.cat_corr_by_bins(world_roi_by_budget,
                          'Low Budget',
                          'High Budget',
                          quartile_intervals[0],
                          quartile_intervals[3],
                          'Genre Correlation with World ROI by Budget')
plt.savefig(os.path.join('images', 'corr_world_roi_by_budget.jpg'),
                            dpi=300,
                            format='JPG')
```


    
![svg](/images/output_83_0.svg)
    


Next is the analogous plot for midrange budgets. The correlation scores here are lower, but you can see that Horror and Thriller are still at the top of the mix for mid-low budget films. It's notable that Musical and Romance movies lead the way for mid-high budget films. These genres definitely go together.


```python
plotting.cat_corr_by_bins(world_roi_by_budget,
                          'Mid-Low Budget',
                          'Mid-High Budget',
                          quartile_intervals[1],
                          quartile_intervals[2],
                          'Genre Correlation with World ROI by Budget')
plt.savefig(os.path.join('images', 'corr_world_roi_by_budget_mid.jpg'),
            dpi=300, 
            format='JPG')
```


    
![svg](/images/output_85_0.svg)
    


I perform the same calculations for domestic ROI. The results are similar, with some small differences. Notably, Comedy has risen up in ranking considerably for everything but low-budget films.


```python
domestic_roi_by_budget = combos.groupby(
    imdb['budget_quartile']).corrwith(imdb['domestic_roi'])
domestic_roi_by_budget
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>genres</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>History</th>
      <th>Horror</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
    <tr>
      <th>budget_quartile</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Low Budget</th>
      <td>-0.06</td>
      <td>-0.04</td>
      <td>-0.02</td>
      <td>-0.04</td>
      <td>-0.07</td>
      <td>-0.06</td>
      <td>-0.04</td>
      <td>-0.12</td>
      <td>-0.03</td>
      <td>0.01</td>
      <td>-0.04</td>
      <td>0.24</td>
      <td>-0.05</td>
      <td>0.21</td>
      <td>-0.03</td>
      <td>-0.01</td>
      <td>-0.03</td>
      <td>0.16</td>
      <td>-0.03</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>Mid-Low Budget</th>
      <td>-0.12</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.09</td>
      <td>-0.12</td>
      <td>0.03</td>
      <td>-0.04</td>
      <td>0.08</td>
      <td>-0.03</td>
      <td>0.01</td>
      <td>0.08</td>
      <td>0.03</td>
      <td>-0.01</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>-0.03</td>
      <td>0.00</td>
      <td>-0.10</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>Mid-High Budget</th>
      <td>-0.09</td>
      <td>-0.07</td>
      <td>-0.03</td>
      <td>0.02</td>
      <td>0.08</td>
      <td>-0.07</td>
      <td>-0.06</td>
      <td>-0.05</td>
      <td>0.00</td>
      <td>-0.08</td>
      <td>-0.05</td>
      <td>-0.02</td>
      <td>0.16</td>
      <td>-0.03</td>
      <td>0.10</td>
      <td>-0.03</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>-0.06</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>High Budget</th>
      <td>-0.18</td>
      <td>0.07</td>
      <td>0.22</td>
      <td>0.05</td>
      <td>0.19</td>
      <td>-0.08</td>
      <td>-0.05</td>
      <td>-0.08</td>
      <td>-0.00</td>
      <td>-0.10</td>
      <td>0.03</td>
      <td>-0.11</td>
      <td>0.11</td>
      <td>-0.00</td>
      <td>-0.08</td>
      <td>0.08</td>
      <td>-0.01</td>
      <td>-0.10</td>
      <td>-0.01</td>
      <td>-0.02</td>
    </tr>
  </tbody>
</table>
</div>




```python
plotting.cat_corr_by_bins(domestic_roi_by_budget,
                          'Low Budget',
                          'High Budget',
                          quartile_intervals[0],
                          quartile_intervals[3],
                          'Genre Correlation with Domestic ROI by Budget')
```




    array([<AxesSubplot:title={'center':'Low Budget\n\\$27,000 to \\$10,000,000'}, xlabel='Correlation'>,
           <AxesSubplot:title={'center':'High Budget\n\\$61,500,000 to \\$410,600,000'}, xlabel='Correlation'>],
          dtype=object)




    
![svg](/images/output_88_1.svg)
    



```python
plotting.cat_corr_by_bins(domestic_roi_by_budget,
                          'Mid-Low Budget',
                          'Mid-High Budget',
                          quartile_intervals[1],
                          quartile_intervals[2],
                          'Genre Correlation with Domestic ROI by Budget')
```




    array([<AxesSubplot:title={'center':'Mid-Low Budget\n\\$10,000,000 to \\$28,000,000'}, xlabel='Correlation'>,
           <AxesSubplot:title={'center':'Mid-High Budget\n\\$28,000,000 to \\$61,500,000'}, xlabel='Correlation'>],
          dtype=object)




    
![svg](/images/output_89_1.svg)
    


### Budget Independently?
Could it be that low-budget movies simply have higher ROI in general than high-budget movies? No. Having a low-budget makes it possible to achieve an extremely high ROI percentage, but is not generally conducive to having high ROI. The following bivariate histogram shows that the highest concentration of movies is located between \\$10M and \\$100M with an ROI in the 100s.


```python
fig, ax = plt.subplots(figsize=(12, 8))
cmap = sns.color_palette('vlag', as_cmap=True)
pos_world_rois = tn.query('worldwide_roi > 0')
ax = sns.histplot(data=pos_world_rois,
                  x='production_budget',
                  y='worldwide_roi',
                  ax=ax,
                  bins='auto',
                  stat='count',
                  cbar=True,
                  log_scale=True,
                  cmap=cmap,
                  cbar_kws={'label': 'Count'})
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax.set_xlabel('Budget')
ax.set_ylabel('ROI')
ax.set_title('Distribution of Movies: Budget vs. World ROI')
plt.savefig(os.path.join('images', 'budget_vs_world_roi.jpg'),
            dpi=300, 
            format='JPG')
```


    
![svg](/images/output_91_0.svg)
    


Production budget has almost no correlation with world ROI, 
though it is weakly negative.


```python
tn[['production_budget']].corrwith(tn['worldwide_roi'])
```




    production_budget   -0.04
    dtype: float64



## Conclusions

#### For high-budget productions, go with animation.
Animation has by far the strongest correlation (nearly 0.25) with ROI for high-budget films. The next best score is adventure, which is nearly 0.1 lower.

#### For low-budget productions, go with horror.
Nothing beats horror movies in terms of ROI, both overall and for low-budget films. The only other options are mystery and thriller, which both go along with horror anyway.

#### Stay away from drama, action, and crime.
Drama, action, and crime consistently show up in the negative on correlation with ROI. This means that movies achieve higher ROI when they are not drama, action, or crime. While it's possible to have success with these genres, they are the worst choices from an investment standpoint.

## Evaluation
My analysis provides some useful insights for Microsoft, but there is much more work to be done. For one, making a successful movie is much more complicated than choosing a genre. There are numerous other factors to consider, such as cast and crew.

Furthermore, I conducted my analysis with a very limited dataset of around 1,400 observations. Many movies were lost in the merge between `imdb` and `tn` because these tables had no unique identifiers in common. The merge could be improved by using fuzzy string matching or another sophisticated process for dirty merging. The ideal situation would be to find a source of data which provides both genre labels and finances.

Nonetheless, I am very confident in the finding that horror movies have the highest ROI. That was a very robust and striking pattern in the data. I am fairly confident in my other findings relating to the business recommendations, but I would like to conduct further research.
