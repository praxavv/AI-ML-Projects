#Here we are trying to calculate PV (present values) of cash flows.
import numpy as np
import pandas as pd

# Historical data (example)
cash_flows = pd.Series([10000, 12000])

# Forecasting future cash flows (example)
future_cash_flows = pd.Series([18000, 20000])

# Combine historical and projected cash flows using pd.concat
# The append method has been removed in recent Pandas versions,
# so we use pd.concat instead.
all_cash_flows = pd.concat([cash_flows, future_cash_flows], ignore_index=True)

# Discount rate (WACC example)
discount_rate = 0.10

# Function to calculate present value
def calculate_present_value(cash_flows, discount_rate):
    present_value = sum([cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows, start=1)])
    return present_value

# Calculate present value of all cash flows
present_value = calculate_present_value(all_cash_flows, discount_rate)

print(f"Total Present Value of Cash Flows: ${present_value:,.2f}")
