# Experiment 7: Forecast hourly employee demand for the following month
# Illustrative monthly data. Replace these values with the organization's actual data.
monthly <- data.frame(
  Month = seq(as.Date("2025-01-01"), by = "month", length.out = 18),
  EmployeesNeeded = c(42, 40, 43, 45, 44, 46, 48, 47, 49, 51, 50, 53,
                      52, 54, 56, 55, 58, 60)
)
demand_ts <- ts(monthly$EmployeesNeeded, frequency = 12,
                start = c(2025, 1))

# Use Holt-Winters exponential smoothing. If the seasonal fit fails on a given
# R version, the non-seasonal alternative below is available.
forecast_result <- tryCatch(
  HoltWinters(demand_ts),
  error = function(e) HoltWinters(demand_ts, gamma = FALSE)
)
next_month <- predict(forecast_result, n.ahead = 1)

cat("HR ANALYTICS: HOURLY EMPLOYEE DEMAND FORECAST\\n")
cat("Latest observed demand:", tail(monthly$EmployeesNeeded, 1), "employees\\n")
cat("Forecast demand for next month:", round(as.numeric(next_month[1]), 0),
    "employees (approx.)\\n")
cat("Note: This is an illustrative forecast based on sample data.\\n")
