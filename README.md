#Paris Saclay Center for Data Science : RAMP on predicting the number of air passengers

The data set was donated to us by an unnamed company handling flight ticket reservations. The data is thin, it contains :
- the date of departure
- the departure airport
- the arrival airport
- the mean and standard deviation of the number of weeks of the reservations made before the departure date
- a field called log_PAX which is related to the number of passengers (the actual number were changed for privacy reasons)

The goal is to predict the log_PAX column. The prediction quality is measured by RMSE.
The data is obviously limited, but since data and location informations are available, it can be joined to external data sets. The challenge in this RAMP is to find good data that can be correlated to flight traffic.

