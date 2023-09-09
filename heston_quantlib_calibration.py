import QuantLib as ql
import math
import numpy as np
import helpers
import pandas as pd
import matplotlib as mpl
from datetime import date, datetime
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *




dates, percent = helpers.init()

df1 = helpers.load_month(2023,3,'SPX')
df1,len = helpers.split_days(df1, num_groups = 1)
df = df1[len-1]
#only use traded quotes, this is already done in the preprocessing and is theoretically not necessary anymore
#df = df[df[' [C_VOLUME]'] > 0]
df = df[df[' [C_VOLUME]'] != " "]
df = df[df[' [C_VOLUME]'] != ""]

daysaddasint = np.array(helpers.get_daystomaturity(df))
strikes = np.array(helpers.get_strike(df))
implied_vols = np.array(helpers.get_IV_C(df))
prices = np.array(helpers.get_callprice(df))
year, month, day = helpers.get_pricingdate(df)
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(ql.UnitedStates.Settlement)
year, month, day = helpers.get_pricingdate(df)   

calculation_date = ql.Date(day, month, year)
spot = helpers.get_underlyinglast(df)
    


dividend_rate = 0.00
risk_free_rate = helpers.get_interestrate(year, month, day, dates, percent)#, daysaddasint/365.0)
dividend_yield = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))

ql.Settings.instance().evaluationDate = calculation_date



flat_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, dividend_rate, day_count))


#gut: 0.9-1.1 and 30-120 days no div yield

mean_vola = np.mean(implied_vols)

down = 0.835
up = 1.169


filter1 = strikes > down*spot
filter2 = strikes < up*spot
filter3 = daysaddasint >30
filter4 = daysaddasint <= 60


filter = filter1 & filter2 & filter3 & filter4 


implied_vols = implied_vols[filter].flatten()
strikes = strikes[filter].flatten()/spot
daysaddasint = daysaddasint[filter].astype(int).flatten()/365.0


#add some buckets to the grid which we will use later
adddays = np.linspace(0.01,1,6)
short_end = np.array([0.01, 0.02, 0.03, 0.04]).flatten()    #they should be contained in the grid
daysaddasint = np.concatenate((np.round(np.concatenate((daysaddasint, adddays))/0.05)*0.05,short_end))
#
strikesadd = np.round(np.linspace(0.85,1.15,9),2)
strikes = np.round(np.concatenate((strikes, strikesadd))/0.03)*0.03
atm = np.array(1.0).flatten()

strikes = np.concatenate((strikes,atm))#at the money is out of rounding scheme
#16x np.nan
nans = np.repeat(np.nan, 10)
implied_vols = np.concatenate((implied_vols, nans))


consolidated = pd.DataFrame({'Strike': strikes, 'Time to Maturity': daysaddasint, 'Implied Volatility': implied_vols})

#use mean in case of dublicates
consolidated = consolidated.groupby(['Strike', 'Time to Maturity']).mean().reset_index()


# Pivot the dataframe to create the matrix
volmatrix = consolidated.pivot(index='Strike', columns='Time to Maturity', values='Implied Volatility')

#interpolation is used to find outliers, therefore a much denser grid is needed, the interpolated values are not used for calibration
try:
    volmatrix_interpolation = volmatrix.interpolate(method='quadratic')
except:
    print("interpolation failed on ", calculation_date)
    volmatrix_interpolation = volmatrix

#remove negative values after interpolation
volmatrix_interpolation[volmatrix_interpolation < 0] = np.nan

daysaddasint_unique = sorted(np.unique(volmatrix.columns))
# outlier detection, one iteration already gives good results 
# we iterate over smiles and check each smile for outliers
for i, ds in enumerate(np.array(daysaddasint_unique)):
    first_col = volmatrix_interpolation.iloc[:, i]
    y = first_col.values
    # Smooth the data using a Savitzky-Golay filter
    
    # Calculate the absolute difference between y and y_smooth, more than 2% difference is considered an outlier
    diff = np.abs(y - savgol_filter(y, 6, 1, mode='interp'))> 0.02

    #now we remove the outliers from the original matrix
    #without the interpolation, the outlier detection is not as good
    volmatrix.iloc[diff, i] = np.nan
    volmatrix_interpolation.iloc[diff, i] = np.nan

try:
    a = volmatrix.interpolate(method='quadratic', axis=0)
except:
    print("interpolöation failed at axis 0 on ", calculation_date)
    a = volmatrix
try:
    b = a.interpolate(method='quadratic', axis=1)
    b[b < 0.1*mean_vola] = np.nan               #remove some outliers if the interpolation technique fails due to illiquid and isolated quotes (should not be triggered)
except:
    print("interpolöation failed at axis 1 on ", calculation_date)
    b = a
volmatrix = b

#if volmatrix contains col with 0.0, delete
try:
    del volmatrix[0.0]
except:
    pass

#used to make ql dates from unique dates
ql_dates = pd.Series(sorted(volmatrix.columns*365)).astype(int)

#QL date to pandas date 
calculation_date_ts = pd.Timestamp(calculation_date.year(), calculation_date.month(), calculation_date.dayOfMonth())

# Add business days to the Pandas timestamp
py_date = ql_dates.apply(lambda x: calculation_date_ts + pd.tseries.offsets.BusinessDay(x))

# Convert Python date to ql.Date
expiration_dates = [ql.Date(d.day, d.month, d.year) for d in py_date]



strikes_unique = sorted(volmatrix.index*int(spot))


#create empty quantlib matrix
rows, cols = volmatrix.shape
implied_vols = ql.Matrix(rows, cols)

# fill the quantlib matrix with the values from the pivot table
for i in range(rows):
    for j in range(cols):
        value = volmatrix.iloc[i, j]
        if not np.isnan(value):
            implied_vols[i][j] = value
        else: 
            implied_vols[i][j] = np.nan


black_var_surface = ql.BlackVarianceSurface(referenceDate=calculation_date, cal=calendar, dates = expiration_dates, strikes = strikes_unique, blackVols=implied_vols, dayCounter=day_count)
black_var_surface.setInterpolation("bilinear")
black_var_surface.enableExtrapolation()


# Plot the surface

# Create a 2D grid of the index and columns
a, b = np.meshgrid(volmatrix.columns, volmatrix.index)

# Flatten the volmatrix to create an array of implied volatilities
d = volmatrix.values


   
# # Create a 3D surface plot of the implied volatilities
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(a, b, d, cmap =plt.cm.Spectral)# 'coolwarm' plt.cm.Spectral 'bone'
# ax.set_xlabel('Time to Maturity')
# ax.set_ylabel('Strike Price')
# ax.set_zlabel('Implied Volatility')
# ax.set_title('Implied Volatility Surface')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()



# dummy parameters
#v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5;
v0=0.1; kappa=3; theta=0.05; rho=-0.8; sigma=0.3;

spot = 1
process = ql.HestonProcess(flat_ts, dividend_ts, 
                           ql.QuoteHandle(ql.SimpleQuote(spot)), 
                           v0, kappa, theta, sigma, rho)

#as we use moneyness we can disregard the spot and use 1 instead
model = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model) 


heston_helpers = []
strike_helpers = []
maturity_helpers = []


# Heston model calibration
for i, date in enumerate(expiration_dates):
    for j, s in enumerate(volmatrix.index):
        t = (date - calculation_date)
        p = ql.Period(t, ql.Days)
        sigma = volmatrix.iloc[j,i]                                                 #dont use the vola from blackvarsurface, use original data from volmatrix
        if not np.isnan(sigma):
            helper = ql.HestonModelHelper(p, calendar, spot, s,                        #spot is set to 1 as we use moneyness	
                                        ql.QuoteHandle(ql.SimpleQuote(sigma)),
                                        flat_ts, 
                                        dividend_ts)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
            strike_helpers.append(s)
            maturity_helpers.append(date)

lm = ql.LevenbergMarquardt(1e-6, 1e-8, 1e-8)
print("Calibrating Heston parameters...")


model.calibrate(heston_helpers, lm, 
                 ql.EndCriteria(1000, 50, 1.0e-8,1.0e-8, 1.0e-8))
theta, kappa, sigma, rho, v0 = model.params()
lambd = kappa * (1 - rho ** 2) ** 0.5

model.calibrate(heston_helpers, lm, 
                 ql.EndCriteria(1000, 50, 1.0e-8,1.0e-8, 1.0e-8))
theta, kappa, sigma, rho, v0 = model.params()
lambd = kappa * (1 - rho ** 2) ** 0.5


relative_error = []
absolute_error = []
square_error = []



#print("Maturity, Strike, Market Value, Model Value, %% Error")
print("{:<25} {:<9} {:<15} {:<14} {:}".format("Maturity", "Strike", "Market Value", "Model Value", "% Error"))

#attention we look on error in market value, not in implied volatility
for i, opt in enumerate(heston_helpers):
    try:
        err = (opt.modelValue()/opt.marketValue() - 1.0)
    except ZeroDivisionError:
        #options with value 0 are disregarded, this is not a model error => 0
        err = 0
    diff = np.abs(opt.modelValue() - opt.marketValue())
    absolute_error.append(diff)
    
    print("{:} {:<9.1f} {:<15.2f} {:<14.2f} {:<+7.2%}".format(maturity_helpers[i], round(strike_helpers[i], 2), round(opt.marketValue(), 2), round(opt.modelValue(), 2), err))


    #append error to list
    relative_error = np.append(relative_error, abs(err))
    square_error = np.append(square_error, err**2)
    rmse = np.sqrt(np.mean(square_error))



# Print calibrated parameter values
print("Calibrated Parameter Values:")
print("theta = ", theta)
print("kappa = ", kappa)
print("sigma = ", sigma)
print("rho = ", rho)
print("v0 = ", v0)


print("mean relative error: {:.2%}".format(np.mean(relative_error)))
print("median relative error: {:.2%}".format(np.median(relative_error)))
print("mean squared error: {:.2%}".format(np.mean(square_error)))
print("median squared error: {:.2%}".format(np.median(square_error)))
print("mean absolute error option price: {:.2%}".format(np.mean(absolute_error)))
print("median absolute error option price: {:.2%}".format(np.median(absolute_error)))


# Plot the surface

# Create a 2D grid of the index and columns
a, b = np.meshgrid(volmatrix.columns, volmatrix.index)

# Flatten the volmatrix to create an array of implied volatilities
d = volmatrix.values


   
# Create a 3D surface plot of the implied volatilities
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(a, b, d, cmap =plt.cm.Spectral)# 'coolwarm' plt.cm.Spectral 'bone'
ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Strike Price')
ax.set_zlabel('Implied Volatility')
ax.set_title('Implied Volatility Surface')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()



# #all combinations of volmatrix rows and columns
# strikes = np.array(volmatrix.index.values)
# maturities = np.array(volmatrix.columns.values)
# a,b = np.meshgrid(strikes, maturities)
# strikes = a.flatten()
# maturities = b.flatten()
# dataframe = pd.DataFrame({'Strike': strikes, 'Time to Maturity': maturities})
# dataframe['iv'] = dataframe.apply(lambda row: heston_pricer(kappa, theta, sigma, rho, v0, risk_free_rate, dividend_rate, row['Time to Maturity'], row['Strike'], 1)[1], axis=1)
# actual_vola = np.array(volmatrix.values).flatten()


#3d plot of the implied volatilities
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(a, b, dataframe['iv'].values.reshape(a.shape), cmap =plt.cm.Spectral)# 'coolwarm' plt.cm.Spectral 'bone'
#scatter actual vola
#ax.scatter(a, b, actual_vola.reshape(a.shape), color='black')

