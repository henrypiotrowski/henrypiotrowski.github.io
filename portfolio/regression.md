## Part 1: EDA

_Insert cells as needed below to write a short EDA/data section that summarizes the data for someone who has never opened it before._ 
- Answer essential questions about the dataset (observation units, time period, sample size, many of the questions above) 
- Note any issues you have with the data (variable X has problem Y that needs to get addressed before using it in regressions or a prediction model because Z)
- Present any visual results you think are interesting or important

### Dataset Overview
- Observation Unit: Each row represents a single residential property sale
- Sample Size: 1,941 property records
- Time Period: Sales occurred from 2006 to 2008
-Target Variable: v_SalePrice, a continuous variable representing the sale price of the property
- Columns: 81 features including physical attributes of the property, neighborhood, condition, and sale details.

### SalesPrice
- v_SalePrice ranges from 13,100 to 755,000 with a median of around 161,900

### Missing Values
- 27 features contain missing values.
- Major missing data issues:
- v_Pool_QC (1,928 missing)
- v_Misc_Feature, v_Alley, v_Fence: mostly missing

### Variable Types
- Numerical (Continuous/Discrete): v_SalePrice, v_Lot_Area, v_Gr_Liv_Area, v_Total_Bsmt_SF, v_Bedroom_AbvGr, etc.
- Categorical (Nominal): v_Neighborhood, v_House_Style, v_Sale_Condition, etc.
- Categorical (Ordinal): v_Overall_Qual, v_Exter_Qual, v_Kitchen_Qual, v_Heating_QC, etc.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

housing_train = df = pd.read_csv('input_data2/housing_train.csv')

print("Summary Statistics:", housing_train.describe())
print(housing_train.info())

print("Dataset Shape:", housing_train.shape)
print("Data Types:", housing_train.dtypes.value_counts())
print("First Rows:", housing_train.head())

missing = housing_train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nMissing Values:\n", missing)


plt.figure(figsize=(10, 6))
sns.histplot(housing_train['v_SalePrice'], kde=True)
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price')
plt.show()
```

    Summary Statistics:        v_MS_SubClass  v_Lot_Frontage     v_Lot_Area  v_Overall_Qual  \
    count    1941.000000     1620.000000    1941.000000     1941.000000   
    mean       58.088614       69.301235   10284.770222        6.113344   
    std        42.946015       23.978101    7832.295527        1.401594   
    min        20.000000       21.000000    1470.000000        1.000000   
    25%        20.000000       58.000000    7420.000000        5.000000   
    50%        50.000000       68.000000    9450.000000        6.000000   
    75%        70.000000       80.000000   11631.000000        7.000000   
    max       190.000000      313.000000  164660.000000       10.000000   
    
           v_Overall_Cond  v_Year_Built  v_Year_Remod/Add  v_Mas_Vnr_Area  \
    count     1941.000000   1941.000000       1941.000000     1923.000000   
    mean         5.568264   1971.321999       1984.073158      104.846074   
    std          1.087465     30.209933         20.837338      184.982611   
    min          1.000000   1872.000000       1950.000000        0.000000   
    25%          5.000000   1953.000000       1965.000000        0.000000   
    50%          5.000000   1973.000000       1993.000000        0.000000   
    75%          6.000000   2001.000000       2004.000000      168.000000   
    max          9.000000   2008.000000       2009.000000     1600.000000   
    
           v_BsmtFin_SF_1  v_BsmtFin_SF_2  ...  v_Wood_Deck_SF  v_Open_Porch_SF  \
    count     1940.000000     1940.000000  ...     1941.000000      1941.000000   
    mean       436.986598       49.247938  ...       92.458011        49.157135   
    std        457.815715      169.555232  ...      127.020523        70.296277   
    min          0.000000        0.000000  ...        0.000000         0.000000   
    25%          0.000000        0.000000  ...        0.000000         0.000000   
    50%        361.500000        0.000000  ...        0.000000        28.000000   
    75%        735.250000        0.000000  ...      168.000000        72.000000   
    max       5644.000000     1474.000000  ...     1424.000000       742.000000   
    
           v_Enclosed_Porch  v_3Ssn_Porch  v_Screen_Porch  v_Pool_Area  \
    count       1941.000000   1941.000000     1941.000000  1941.000000   
    mean          22.947965      2.249871       16.249871     3.386399   
    std           65.249307     22.416832       56.748086    43.695267   
    min            0.000000      0.000000        0.000000     0.000000   
    25%            0.000000      0.000000        0.000000     0.000000   
    50%            0.000000      0.000000        0.000000     0.000000   
    75%            0.000000      0.000000        0.000000     0.000000   
    max         1012.000000    407.000000      576.000000   800.000000   
    
             v_Misc_Val    v_Mo_Sold    v_Yr_Sold    v_SalePrice  
    count   1941.000000  1941.000000  1941.000000    1941.000000  
    mean      52.553838     6.431221  2006.998454  182033.238022  
    std      616.064459     2.745199     0.801736   80407.100395  
    min        0.000000     1.000000  2006.000000   13100.000000  
    25%        0.000000     5.000000  2006.000000  130000.000000  
    50%        0.000000     6.000000  2007.000000  161900.000000  
    75%        0.000000     8.000000  2008.000000  215000.000000  
    max    17000.000000    12.000000  2008.000000  755000.000000  
    
    [8 rows x 37 columns]
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1941 entries, 0 to 1940
    Data columns (total 81 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   parcel             1941 non-null   object 
     1   v_MS_SubClass      1941 non-null   int64  
     2   v_MS_Zoning        1941 non-null   object 
     3   v_Lot_Frontage     1620 non-null   float64
     4   v_Lot_Area         1941 non-null   int64  
     5   v_Street           1941 non-null   object 
     6   v_Alley            136 non-null    object 
     7   v_Lot_Shape        1941 non-null   object 
     8   v_Land_Contour     1941 non-null   object 
     9   v_Utilities        1941 non-null   object 
     10  v_Lot_Config       1941 non-null   object 
     11  v_Land_Slope       1941 non-null   object 
     12  v_Neighborhood     1941 non-null   object 
     13  v_Condition_1      1941 non-null   object 
     14  v_Condition_2      1941 non-null   object 
     15  v_Bldg_Type        1941 non-null   object 
     16  v_House_Style      1941 non-null   object 
     17  v_Overall_Qual     1941 non-null   int64  
     18  v_Overall_Cond     1941 non-null   int64  
     19  v_Year_Built       1941 non-null   int64  
     20  v_Year_Remod/Add   1941 non-null   int64  
     21  v_Roof_Style       1941 non-null   object 
     22  v_Roof_Matl        1941 non-null   object 
     23  v_Exterior_1st     1941 non-null   object 
     24  v_Exterior_2nd     1941 non-null   object 
     25  v_Mas_Vnr_Type     769 non-null    object 
     26  v_Mas_Vnr_Area     1923 non-null   float64
     27  v_Exter_Qual       1941 non-null   object 
     28  v_Exter_Cond       1941 non-null   object 
     29  v_Foundation       1941 non-null   object 
     30  v_Bsmt_Qual        1891 non-null   object 
     31  v_Bsmt_Cond        1891 non-null   object 
     32  v_Bsmt_Exposure    1889 non-null   object 
     33  v_BsmtFin_Type_1   1891 non-null   object 
     34  v_BsmtFin_SF_1     1940 non-null   float64
     35  v_BsmtFin_Type_2   1891 non-null   object 
     36  v_BsmtFin_SF_2     1940 non-null   float64
     37  v_Bsmt_Unf_SF      1940 non-null   float64
     38  v_Total_Bsmt_SF    1940 non-null   float64
     39  v_Heating          1941 non-null   object 
     40  v_Heating_QC       1941 non-null   object 
     41  v_Central_Air      1941 non-null   object 
     42  v_Electrical       1940 non-null   object 
     43  v_1st_Flr_SF       1941 non-null   int64  
     44  v_2nd_Flr_SF       1941 non-null   int64  
     45  v_Low_Qual_Fin_SF  1941 non-null   int64  
     46  v_Gr_Liv_Area      1941 non-null   int64  
     47  v_Bsmt_Full_Bath   1939 non-null   float64
     48  v_Bsmt_Half_Bath   1939 non-null   float64
     49  v_Full_Bath        1941 non-null   int64  
     50  v_Half_Bath        1941 non-null   int64  
     51  v_Bedroom_AbvGr    1941 non-null   int64  
     52  v_Kitchen_AbvGr    1941 non-null   int64  
     53  v_Kitchen_Qual     1941 non-null   object 
     54  v_TotRms_AbvGrd    1941 non-null   int64  
     55  v_Functional       1941 non-null   object 
     56  v_Fireplaces       1941 non-null   int64  
     57  v_Fireplace_Qu     1001 non-null   object 
     58  v_Garage_Type      1836 non-null   object 
     59  v_Garage_Yr_Blt    1834 non-null   float64
     60  v_Garage_Finish    1834 non-null   object 
     61  v_Garage_Cars      1940 non-null   float64
     62  v_Garage_Area      1940 non-null   float64
     63  v_Garage_Qual      1834 non-null   object 
     64  v_Garage_Cond      1834 non-null   object 
     65  v_Paved_Drive      1941 non-null   object 
     66  v_Wood_Deck_SF     1941 non-null   int64  
     67  v_Open_Porch_SF    1941 non-null   int64  
     68  v_Enclosed_Porch   1941 non-null   int64  
     69  v_3Ssn_Porch       1941 non-null   int64  
     70  v_Screen_Porch     1941 non-null   int64  
     71  v_Pool_Area        1941 non-null   int64  
     72  v_Pool_QC          13 non-null     object 
     73  v_Fence            365 non-null    object 
     74  v_Misc_Feature     63 non-null     object 
     75  v_Misc_Val         1941 non-null   int64  
     76  v_Mo_Sold          1941 non-null   int64  
     77  v_Yr_Sold          1941 non-null   int64  
     78  v_Sale_Type        1941 non-null   object 
     79  v_Sale_Condition   1941 non-null   object 
     80  v_SalePrice        1941 non-null   int64  
    dtypes: float64(11), int64(26), object(44)
    memory usage: 1.2+ MB
    None
    Dataset Shape: (1941, 81)
    Data Types: object     44
    int64      26
    float64    11
    Name: count, dtype: int64
    First Rows:            parcel  v_MS_SubClass v_MS_Zoning  v_Lot_Frontage  v_Lot_Area  \
    0  1056_528110080             20          RL           107.0       13891   
    1  1055_528108150             20          RL            98.0       12704   
    2  1053_528104050             20          RL           114.0       14803   
    3  2213_909275160             20          RL           126.0       13108   
    4  1051_528102030             20          RL            96.0       12444   
    
      v_Street v_Alley v_Lot_Shape v_Land_Contour v_Utilities  ... v_Pool_Area  \
    0     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    1     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    2     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    3     Pave     NaN         IR2            HLS      AllPub  ...           0   
    4     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    
      v_Pool_QC v_Fence v_Misc_Feature v_Misc_Val v_Mo_Sold v_Yr_Sold  \
    0       NaN     NaN            NaN          0         1      2008   
    1       NaN     NaN            NaN          0         1      2008   
    2       NaN     NaN            NaN          0         6      2008   
    3       NaN     NaN            NaN          0         6      2007   
    4       NaN     NaN            NaN          0        11      2008   
    
       v_Sale_Type  v_Sale_Condition  v_SalePrice  
    0          New           Partial       372402  
    1          New           Partial       317500  
    2          New           Partial       385000  
    3          WD             Normal       153500  
    4          New           Partial       394617  
    
    [5 rows x 81 columns]
    
    Missing Values:
     v_Pool_QC           1928
    v_Misc_Feature      1878
    v_Alley             1805
    v_Fence             1576
    v_Mas_Vnr_Type      1172
    v_Fireplace_Qu       940
    v_Lot_Frontage       321
    v_Garage_Cond        107
    v_Garage_Qual        107
    v_Garage_Finish      107
    v_Garage_Yr_Blt      107
    v_Garage_Type        105
    v_Bsmt_Exposure       52
    v_Bsmt_Cond           50
    v_Bsmt_Qual           50
    v_BsmtFin_Type_1      50
    v_BsmtFin_Type_2      50
    v_Mas_Vnr_Area        18
    v_Bsmt_Half_Bath       2
    v_Bsmt_Full_Bath       2
    v_BsmtFin_SF_1         1
    v_Total_Bsmt_SF        1
    v_Garage_Cars          1
    v_Garage_Area          1
    v_Bsmt_Unf_SF          1
    v_BsmtFin_SF_2         1
    v_Electrical           1
    dtype: int64



    
![png](output_2_1.png)
    


## Part 2: Running Regressions

**Run these regressions on the RAW data, even if you found data issues that you think should be addressed.**

_Insert cells as needed below to run these regressions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

_Note: If you are using VS Code, these might not display correctly. Add a "\\" in front of the underscores in the variable names, so `\text{v_Lot_Area}` becomes `\text{v\_Lot\_Area}`._

1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Yr_Sold}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * (\text{v_Yr_Sold==2007})+ \beta_2 * (\text{v_Yr_Sold==2008})$
1. Choose your own adventure: Pick any five variables from the dataset that you think will generate good R2. Use them in a regression of $log(\text{Sale Price}_{i,t})$ 
    - Tip: You can transform/create these five variables however you want, even if it creates extra variables. For example: I'd count Model 6 above as only using one variable: `v_Yr_Sold`.
    - I got an R2 of 0.877 with just "5" variables. How close can you get? One student in five years has beat that. 
    

**Bonus formatting trick:** Instead of reporting all regressions separately, report all seven regressions in a _single_ table using `summary_col`.



```python
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols as sm_ols
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
```


```python
#1.
model = smf.ols('v_SalePrice ~ v_Lot_Area', data=housing_train)

results = model.fit()

print(results.summary())

```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            v_SalePrice   R-squared:                       0.067
    Model:                            OLS   Adj. R-squared:                  0.066
    Method:                 Least Squares   F-statistic:                     138.3
    Date:                Sun, 30 Mar 2025   Prob (F-statistic):           6.82e-31
    Time:                        18:41:09   Log-Likelihood:                -24610.
    No. Observations:                1941   AIC:                         4.922e+04
    Df Residuals:                    1939   BIC:                         4.924e+04
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept   1.548e+05   2911.591     53.163      0.000    1.49e+05     1.6e+05
    v_Lot_Area     2.6489      0.225     11.760      0.000       2.207       3.091
    ==============================================================================
    Omnibus:                      668.513   Durbin-Watson:                   1.064
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3001.894
    Skew:                           1.595   Prob(JB):                         0.00
    Kurtosis:                       8.191   Cond. No.                     2.13e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.13e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
#2. 
housing_train['log_Lot_Area'] = np.log(housing_train['v_Lot_Area'])

model = smf.ols('v_SalePrice ~ log_Lot_Area', data=housing_train)
results = model.fit()

print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            v_SalePrice   R-squared:                       0.128
    Model:                            OLS   Adj. R-squared:                  0.128
    Method:                 Least Squares   F-statistic:                     285.6
    Date:                Sun, 30 Mar 2025   Prob (F-statistic):           6.95e-60
    Time:                        18:41:09   Log-Likelihood:                -24544.
    No. Observations:                1941   AIC:                         4.909e+04
    Df Residuals:                    1939   BIC:                         4.910e+04
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept    -3.279e+05   3.02e+04    -10.850      0.000   -3.87e+05   -2.69e+05
    log_Lot_Area  5.603e+04   3315.139     16.901      0.000    4.95e+04    6.25e+04
    ==============================================================================
    Omnibus:                      650.067   Durbin-Watson:                   1.042
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2623.687
    Skew:                           1.587   Prob(JB):                         0.00
    Kurtosis:                       7.729   Cond. No.                         164.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
#3. 

housing_train['log_SalePrice'] = np.log(housing_train['v_SalePrice'])

model = smf.ols('log_SalePrice ~ v_Lot_Area', data=housing_train)
results = model.fit()

print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          log_SalePrice   R-squared:                       0.065
    Model:                            OLS   Adj. R-squared:                  0.064
    Method:                 Least Squares   F-statistic:                     133.9
    Date:                Sun, 30 Mar 2025   Prob (F-statistic):           5.46e-30
    Time:                        18:41:09   Log-Likelihood:                -927.19
    No. Observations:                1941   AIC:                             1858.
    Df Residuals:                    1939   BIC:                             1870.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     11.8941      0.015    813.211      0.000      11.865      11.923
    v_Lot_Area  1.309e-05   1.13e-06     11.571      0.000    1.09e-05    1.53e-05
    ==============================================================================
    Omnibus:                       75.460   Durbin-Watson:                   0.980
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              218.556
    Skew:                          -0.066   Prob(JB):                     3.48e-48
    Kurtosis:                       4.639   Cond. No.                     2.13e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.13e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
#4. 

housing_train['log_SalePrice'] = np.log(housing_train['v_SalePrice'])
housing_train['log_Lot_Area'] = np.log(housing_train['v_Lot_Area'])

model = smf.ols('log_SalePrice ~ log_Lot_Area', data=housing_train)
results = model.fit()

print(results.summary())

```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          log_SalePrice   R-squared:                       0.135
    Model:                            OLS   Adj. R-squared:                  0.135
    Method:                 Least Squares   F-statistic:                     302.5
    Date:                Sun, 30 Mar 2025   Prob (F-statistic):           4.38e-63
    Time:                        18:41:09   Log-Likelihood:                -851.27
    No. Observations:                1941   AIC:                             1707.
    Df Residuals:                    1939   BIC:                             1718.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept        9.4051      0.151     62.253      0.000       9.109       9.701
    log_Lot_Area     0.2883      0.017     17.394      0.000       0.256       0.321
    ==============================================================================
    Omnibus:                       84.067   Durbin-Watson:                   0.955
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              255.283
    Skew:                          -0.100   Prob(JB):                     3.68e-56
    Kurtosis:                       4.765   Cond. No.                         164.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
#5.

housing_train['log_SalePrice'] = np.log(housing_train['v_SalePrice'])

model = smf.ols('log_SalePrice ~ v_Yr_Sold', data=housing_train)
results = model.fit()

print(results.summary())


```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          log_SalePrice   R-squared:                       0.000
    Model:                            OLS   Adj. R-squared:                 -0.000
    Method:                 Least Squares   F-statistic:                    0.2003
    Date:                Sun, 30 Mar 2025   Prob (F-statistic):              0.655
    Time:                        18:41:09   Log-Likelihood:                -991.88
    No. Observations:                1941   AIC:                             1988.
    Df Residuals:                    1939   BIC:                             1999.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     22.2932     22.937      0.972      0.331     -22.690      67.277
    v_Yr_Sold     -0.0051      0.011     -0.448      0.655      -0.028       0.017
    ==============================================================================
    Omnibus:                       55.641   Durbin-Watson:                   0.985
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              131.833
    Skew:                           0.075   Prob(JB):                     2.36e-29
    Kurtosis:                       4.268   Cond. No.                     5.03e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.03e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
#6.

housing_train['log_SalePrice'] = np.log(housing_train['v_SalePrice'])

housing_train['v_Yr_Sold'] = housing_train['v_Yr_Sold'].astype('category')

model = smf.ols('log_SalePrice ~ C(v_Yr_Sold)', data=housing_train)
results = model.fit()

print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          log_SalePrice   R-squared:                       0.001
    Model:                            OLS   Adj. R-squared:                  0.000
    Method:                 Least Squares   F-statistic:                     1.394
    Date:                Sun, 30 Mar 2025   Prob (F-statistic):              0.248
    Time:                        18:41:09   Log-Likelihood:                -990.59
    No. Observations:                1941   AIC:                             1987.
    Df Residuals:                    1938   BIC:                             2004.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    Intercept               12.0229      0.016    745.087      0.000      11.991      12.055
    C(v_Yr_Sold)[T.2007]     0.0256      0.022      1.150      0.250      -0.018       0.069
    C(v_Yr_Sold)[T.2008]    -0.0103      0.023     -0.450      0.653      -0.055       0.035
    ==============================================================================
    Omnibus:                       54.618   Durbin-Watson:                   0.989
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              127.342
    Skew:                           0.080   Prob(JB):                     2.23e-28
    Kurtosis:                       4.245   Cond. No.                         3.79
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
#7.

housing_train['log_SalePrice'] = np.log(housing_train['v_SalePrice'])
housing_train['log_Gr_Liv_Area'] = np.log(housing_train['v_Gr_Liv_Area'])


model = smf.ols('log_SalePrice ~ v_Overall_Qual + log_Gr_Liv_Area + v_Total_Bsmt_SF + v_Garage_Cars + v_Pool_Area', data=housing_train)
results = model.fit()

print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          log_SalePrice   R-squared:                       0.813
    Model:                            OLS   Adj. R-squared:                  0.813
    Method:                 Least Squares   F-statistic:                     1683.
    Date:                Sun, 30 Mar 2025   Prob (F-statistic):               0.00
    Time:                        18:41:09   Log-Likelihood:                 636.45
    No. Observations:                1939   AIC:                            -1261.
    Df Residuals:                    1933   BIC:                            -1227.
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept           8.2115      0.103     79.432      0.000       8.009       8.414
    v_Overall_Qual      0.1294      0.004     31.725      0.000       0.121       0.137
    log_Gr_Liv_Area     0.3727      0.016     23.450      0.000       0.342       0.404
    v_Total_Bsmt_SF     0.0001    1.1e-05     12.479      0.000       0.000       0.000
    v_Garage_Cars       0.0988      0.007     14.779      0.000       0.086       0.112
    v_Pool_Area      -5.53e-05   9.16e-05     -0.604      0.546      -0.000       0.000
    ==============================================================================
    Omnibus:                      951.739   Durbin-Watson:                   1.685
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            16685.965
    Skew:                          -1.876   Prob(JB):                         0.00
    Kurtosis:                      16.873   Cond. No.                     3.01e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 3.01e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.


## Part 3: Regression interpretation

_Insert cells as needed below to answer these questions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. If you didn't use the `summary_col` trick, list $\beta_1$ for Models 1-6 to make it easier on your graders.
1. Interpret $\beta_1$ in Model 2. 
1. Interpret $\beta_1$ in Model 3. 
    - HINT: You might need to print out more decimal places. Show at least 2 non-zero digits. 
1. Of models 1-4, which do you think best explains the data and why?
1. Interpret $\beta_1$ In Model 5
1. Interpret $\alpha$ in Model 6
1. Interpret $\beta_1$ in Model 6
1. Why is the R2 of Model 6 higher than the R2 of Model 5?
1. What variables did you include in Model 7?
1. What is the R2 of your Model 7?
1. Speculate (not graded): Could you use the specification of Model 6 in a predictive regression? 
1. Speculate (not graded): Could you use the specification of Model 5 in a predictive regression? 



```python
#1.
reg1= smf.ols('v_SalePrice ~ v_Lot_Area', data=housing_train).fit()
reg2= smf.ols('v_SalePrice ~ log_Lot_Area', data=housing_train).fit()
reg3= smf.ols('log_SalePrice ~ v_Lot_Area', data=housing_train).fit()
reg4= smf.ols('log_SalePrice ~ log_Lot_Area', data=housing_train).fit()
reg5= smf.ols('log_SalePrice ~ v_Yr_Sold', data=housing_train).fit()
reg6= smf.ols('log_SalePrice ~ C(v_Yr_Sold)', data=housing_train).fit()

print(summary_col(results=[reg1, reg2, reg3, reg4, reg5, reg6],
                  stars=True,
                  float_format='%0.4f',
                  model_names=['Question 1', 'Question 2', 'Question 3', 'Question 4', 'Question 5', 'Question 6']))

```

    
    ===============================================================================================
                           Question 1      Question 2   Question 3 Question 4 Question 5 Question 6
    -----------------------------------------------------------------------------------------------
    Intercept            154789.5502*** -327915.8023*** 11.8941*** 9.4051***  12.0229*** 12.0229***
                         (2911.5906)    (30221.3471)    (0.0146)   (0.1511)   (0.0161)   (0.0161)  
    v_Lot_Area           2.6489***                      0.0000***                                  
                         (0.2252)                       (0.0000)                                   
    log_Lot_Area                        56028.1700***              0.2883***                       
                                        (3315.1392)                (0.0166)                        
    v_Yr_Sold[T.2007]                                                         0.0256               
                                                                              (0.0222)             
    v_Yr_Sold[T.2008]                                                         -0.0103              
                                                                              (0.0228)             
    C(v_Yr_Sold)[T.2007]                                                                 0.0256    
                                                                                         (0.0222)  
    C(v_Yr_Sold)[T.2008]                                                                 -0.0103   
                                                                                         (0.0228)  
    R-squared            0.0666         0.1284          0.0646     0.1350     0.0014     0.0014    
    R-squared Adj.       0.0661         0.1279          0.0641     0.1345     0.0004     0.0004    
    ===============================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


2. A 1% increase in Lot_Area, increases SalesPrice by $560.28 (56028.17/100)

3. A 1 unit increase in Lot_Area, increases SalesPrice by 0.0013% (100*(exp(0.00001309)−1))

4. Model 4 (log_SalePrice ~ log_Lot_Area) best explains the data since it has the highest R^2 value of 0.1350

5. If v_Yr_Sold increases by 1 year, SalesPrice decreases by 0.509% (100*(exp(-0.0051)−1))

6. The average SalePrice in 2006 was $166,172 (exp(12.0229)

7. If v_Yr_sold increases by 1 year, SalesPrice increases by 2.59% (100*(exp(0.0256)−1))

8. R^2 of model 6 is higher than R^2 of model 5 because model 6 treats year as a categorical variable, allowing for nonlinear differences in sale price across years while Model 5 assumes a linear trend. This gives Model 6 more flexibility to fit the data.

9. The variables I included in model 7 were
   - v_Overall_Qual
   - v_Gr_Liv_Area (log)
   - v_Total_Bsmt_SF
   - v_Garage_Cars
   - v_Pool_Area

10. R^2 = 0.813

11. Probaly not since the model treats year as a dummy variable and not continous, won't work well for future years

12. Yes model 5 could be used for predictive regressions
