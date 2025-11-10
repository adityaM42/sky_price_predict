import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os

os.makedirs("data", exist_ok=True)

origins = ["Delhi", "Mumbai", "Bangalore"]
destinations = ["Kolkata", "Chennai", "Hyderabad"]
airlines = ["AirIndia", "SpiceJet", "Indigo", "Vistara"]
classes = ["Economy", "Business"]

rows = []
np.random.seed(42)
today = datetime(2025, 1, 1)

for airline in airlines:
    for o in origins:
        for d in destinations:
            if o == d:
                continue
            for c in classes:
                # choose 50 random departure dates within 3 months
                for dep_offset in np.random.randint(10, 90, 50):
                    dep_date = today + timedelta(days=int(dep_offset))
                    base_price = np.random.randint(3000, 8000)
                    # simulate daily prices from 90 days before dep_date
                    for delta in range(90, 0, -1):
                        query_date = dep_date - timedelta(days=delta)
                        days_left = (dep_date - query_date).days
                        # shape: cheap mid-window, expensive last-minute
                        price = (
                            base_price
                            * (1 + 0.4 * np.sin(days_left / 90 * np.pi))
                            + np.random.randint(-300, 300)
                        )
                        price = max(2000, price)
                        rows.append(
                            {
                                "query_date": query_date.date(),
                                "departure_date": dep_date.date(),
                                "origin": o,
                                "destination": d,
                                "airline": airline,
                                "stops": random.choice(["zero", "one"]),
                                "departure_time": random.choice(
                                    ["Morning", "Afternoon", "Evening", "Night"]
                                ),
                                "arrival_time": random.choice(
                                    ["Morning", "Afternoon", "Evening", "Night"]
                                ),
                                "class": c,
                                "price": round(price, 2),
                            }
                        )

df = pd.DataFrame(rows)
df.to_csv("data/historical_prices_timeseries.csv", index=False)
print("✅ Generated data/historical_prices_timeseries.csv with", len(df), "rows")
