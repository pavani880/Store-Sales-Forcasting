
# 📊 Sales Analysis & Sales Forecasting using Python

## 📌 Project Overview

This project performs an end-to-end sales analysis and forecasting using Python. The objective is to analyze historical sales data, identify business trends, uncover valuable insights, and predict future sales using time-series forecasting models.

The project demonstrates the complete data analytics lifecycle including data cleaning, exploratory data analysis (EDA), visualization, statistical testing, and predictive modeling.

# 📂 Dataset

The project uses a retail sales dataset containing transactional information such as:

- Order ID
- Order Date
- Ship Mode
- Customer Details
- Region
- State
- City
- Segment
- Category
- Sub-Category
- Sales
- Quantity
- Discount
- Profit

---

# 🛠 Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Jupyter Notebook

---

# 📋 Project Workflow

## 1. Data Loading

- Imported dataset using Pandas
- Explored dataset structure
- Checked data types

---

## 2. Data Cleaning

The following preprocessing steps were performed:

✔ Removed duplicate records

✔ Checked missing values

✔ Removed unnecessary columns

✔ Converted Order Date into DateTime format

✔ Verified data quality before analysis

---

## 3. Exploratory Data Analysis (EDA)

Multiple KPIs were analyzed to understand business performance.

### KPI 1 — Region-wise Sales

Objective:

Identify the region contributing the highest revenue.

What was done:

- Grouped sales by region
- Compared regional performance
- Created visualization

Business Value:

Helps businesses identify high-performing markets and allocate resources effectively.

---

### KPI 2 — Top 10 States by Sales

Objective:

Find the highest revenue-generating states.

Analysis:

- Aggregated sales state-wise
- Ranked states based on revenue
- Created bar chart

Business Value:

Supports regional marketing and expansion strategies.

---

### KPI 3 — City-wise Sales Analysis

Objective:

Identify cities generating maximum revenue.

Business Value:

Helps businesses understand local market demand.

---

### KPI 4 — Segment-wise Sales

Analyzed customer segments to determine which contributes the highest sales.

Business Value:

Allows businesses to focus marketing efforts on profitable customer groups.

---

### KPI 5 — Sales Percentage by Segment

Created percentage distribution of sales across customer segments.

Business Value:

Provides a clear understanding of customer contribution.

---

### KPI 6 — Sub-Category Performance

Compared sales across different product sub-categories.

Business Value:

Identifies best-selling products.

Supports inventory planning.

---

### KPI 7 — Profit Analysis

Analyzed:

- Highest profit states
- Highest loss-making states

Business Value:

Helps identify profitable and underperforming markets.

---

### KPI 8 — Order Analysis

Studied:

- States with highest orders
- Cities with highest orders

Business Value:

Shows customer concentration and demand.

---

### KPI 9 — Monthly Sales Trend

Analyzed monthly sales trends over time.

Business Value:

Identifies seasonality and peak sales periods.

Useful for demand forecasting.

---

### KPI 10 — Top Customers

Identified:

- Highest revenue customers
- Customers with maximum orders

Business Value:

Supports customer retention strategies.

---

### KPI 11 — Region-wise Sales Contribution

Compared overall contribution of each region.

Business Value:

Measures regional business performance.

---

### KPI 12 — Ship Mode Analysis

Analyzed:

- Most frequently used shipping method
- Sales contribution by shipping mode

Business Value:

Helps optimize logistics and delivery strategy.

---



# 📌 Key Insights/obersvations

Some major business insights obtained from the analysis include:

- West Region generated the highest sales.
- Certain states contributed significantly more revenue than others.
- Customer segments showed different purchasing behavior.
- A few sub-categories dominated overall sales.
- Monthly sales exhibited seasonal trends.
- Customer purchasing patterns varied significantly.
- Shipping mode influenced sales contribution.
- SARIMA produced more accurate forecasts compared to ARIMA.


