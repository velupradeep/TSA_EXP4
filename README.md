# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 13/05/2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv("Customer_Transactions.csv")

data['last_purchase_date'] = pd.to_datetime(data['last_purchase_date'])

data = data.sort_values('last_purchase_date')

X = data['avg_purchase_value'].dropna()

plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(data['last_purchase_date'][:len(X)], X)
plt.title('Original Data (Average Purchase Value)')
plt.xlabel('Date')
plt.ylabel('Average Purchase Value')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=30, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=30, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

N = 1000

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title("ACF - ARMA(1,1)")
plt.show()

plot_pacf(ARMA_1)
plt.title("PACF - ARMA(1,1)")
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']

theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title("ACF - ARMA(2,2)")
plt.show()

plot_pacf(ARMA_2)
plt.title("PACF - ARMA(2,2)")
plt.show()


```

### OUTPUT:
<img width="1161" height="611" alt="image" src="https://github.com/user-attachments/assets/d0170b66-19de-4eb1-8ede-3f116202f74d" />


<img width="1359" height="646" alt="image" src="https://github.com/user-attachments/assets/03d4e87b-d55a-4aef-9a32-76494d87889a" />

<img width="1076" height="561" alt="image" src="https://github.com/user-attachments/assets/e1863431-0ad9-40e4-a5b2-a762e8c88ff7" />


<img width="1111" height="562" alt="image" src="https://github.com/user-attachments/assets/2e096409-3aa9-4859-b165-cc884ed82c4c" />


<img width="1076" height="549" alt="image" src="https://github.com/user-attachments/assets/7f8d1b45-2240-47dd-b0e9-f55403020922" />

<img width="1022" height="502" alt="image" src="https://github.com/user-attachments/assets/acee4023-c946-47f3-b4f0-65a1989d8f8d" />

<img width="1022" height="502" alt="image" src="https://github.com/user-attachments/assets/f1fc0dc1-25b7-413e-a329-b23bd483646d" />


<img width="947" height="486" alt="image" src="https://github.com/user-attachments/assets/279fb2a0-d441-4384-bda3-12776494dcda" />










### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
