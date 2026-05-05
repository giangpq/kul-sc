# Goal

Our goal is to develop a pricing methodology for campsite accommodations that drives strong booking performance. Specifically, we aim to identify prices that maximize booking revenue across different accommodation types, markets, and stay weeks.


# Data

**What is the data about?**
The dataset contains historical booking and pricing records for campsite accommodations across multiple markets and stay weeks. It is a synthetic dataset inspired by real-world yield management data, and is used to develop a pricing methodology that maximises booking revenue.


**How is it recorded?**
The data is structured as a **booking-window panel**: for each unique combination of accommodation, market segment, and stay week (`ReservableOptionMarketGroupId`), there are 53 weekly snapshots tracking how bookings and prices evolve from 52 weeks before arrival down to the arrival week itself. Two years of observations are included: 2024 and 2025.


**What features does it have?**
Features fall into five groups:
- **Identifiers**: the unique ID, booking horizon (`WeekBeforeArrival`), and stay date
- **Segmentation**: market group, brand, campsite, accommodation type, special periods
- **Geography & clustering**: country, region, campsite type, coordinates, pre-computed seasonal and campsite clusters
- **Accommodation features**: bedrooms, bathrooms, sleeps, airco, hot tub, TV, kitchen, decking, etc.
- **Pricing & bookings**: current and last-year prices, incremental and total booked nights, capacity


**How should it be loaded?**
The dataset has 3,130,816 rows and 38 columns. Use `polars` for data loading and manipulation.


**Basic characteristics**
- 19 columns of numeric type; date columns recorded as strings
- 2025 prices and bookings closely mirror 2024 patterns
- Strong with-year seasonality in both prices and bookings
- Last-year lag features (`DiscountedPriceLastYear`, `HistoricalBookedNightsLastYear`, `CapacityLastYear`) are all 0s and should NOT be used.
- Note that `Kitchen`, `DeckingType`, and `DeckingExtras` contain `"None"` as a valid category — make sure these are not treated as missing values.


