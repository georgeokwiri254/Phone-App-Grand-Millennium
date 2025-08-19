# Instructions for Processing Hotel Booking Data## Entered On Report

1.Load the entered on report xlsm. report
2.Look for the ENTERED ON Tab
3.Once located proceed to copy the sheet and convert it to a CSV file.

# split

4.Split the stay period to find the number of days are staying in a specific month. for example if a guest checks in on the 30 August and check out on 2nd Spetember he will spend 2 days in Augaust and one day in september
the guest will check out on the 2nd Sep which will not be counted. Hence total stays will be 3 nights
Data volume & approach — At ~200 bookings/day  a Python ETL that runs in-memory is perfectly fine. It’s simple to test, fast to run, and easy to debug. You don’t need heavy DB-only transforms for this volume.

Source handling (xlsm) — Always read the ENTERED ON sheet from the .xlsm and immediately write an audited raw CSV copy (entered_on_raw.csv) before any transformations.

Keep raw + canonical — Save raw rows into a bookings_raw table or CSV. This lets you re-run transforms or debug if something changes.

Split rule (business) — Use nights = days from check_in (inclusive) to check_out (exclusive). Checkout day is not counted. Example: 30 Aug → 2 Sep → nights = 3, nights in Aug = 2, nights in Sep = 1.

Preferred split location — For your scale: split in Python ETL before loading. Why: easier handling of messy input (missing times, bad formats), immediate local testing, and simple integration with Streamlit buttons. Keep the SQL view option as an alternative for reporting or if you later want DB-native speed.

Idempotence & logging — ETL should be idempotent: use run IDs or overwrite outputs predictably. Log source filename, rows read, rows written, and any rejected rows to etl_log.csv.

Output models — Produce:

booking_nights.csv (one row per night per booking)

booking_month_summary.csv (booking × month × nights)

entered_on_raw.csv (audit)
Index by RESV ID, check in,check out, month_num for fast queries after loading to DB.

Validation rules — Validate check_in < check_out, parseable dates, non-null booking IDs. Flag or store invalid rows separately instead of crashing.
5 After computation load the data into SQL

# EDA ANALYSIS#

# columns provided are as per the converted csv file loaded in SQL

6.Add the AMOUNT column total have it as one of the KPI cards, Room nights in nights column also one of the KPIS
7. I need to see EDAs of pivoted a table showing top companies by AMOUNT column SHOULD BE MULTIPLIED BY 1.1 CHANGE THIS IN THE CONVERTER before loading to SQL booked and a bar graph
8.Booking made by INSERT_USER by count in a table and bar graph
9. Chart showing the Stay dates of the entered on booking in a bar chart.
10. Also room types booked by count. room column G
12. Seasonal EDA for summer and winter COLUMN T
13 Bookings during events Events Dates COLUMN AE
14.Booking Lead Times as histogram COLUMN AD
15. Decriptive statistics of the ADR using hitogram and Box plot analysis of summer and witer ADR. Gracefully error handle cases or mode or kurtosis where it may be only values which mode cannot be determined show as N/A. column AC
Mean
Standard Error
Median
Mode
Standard Deviation
Sample Variance
Kurtosis
Skewness
Range
Minimum
Maximum
Sum
Count
Largest(1)
Smallest(1)
Confidence Level(95.0%)
16. Show which months are being booked and by which company have a slicer for company that can select the company and which months are booked as per the split data.
17.which months as per split data are most booked as per split dates.
18. Booking is more than 10 NIGHTS  flag it

19. A Calendar heatmap showing the top 10 companies by booked nights and the dates booked. And show the number in the chart similar to the one in Block analysis tab. This is just an example but the charts are not related. use red as heatmap color

# Arrival Report

For Arrival report
1.Load the arrival report xlsm. report.
2.Look for the ARRIVAL CHECK Tab
3.Once located located proceed to copy the sheet and convert it to a CSV file.
4.Highest arrirval count by company as per COMPANY_NAME and ARRIVAL count with horrizontal bar chart.
5.Company name and deposit paid pivot.DEPOSIT_PAID and COMPANY_NAME in this case. if company is T- Assos Tourism LLC
 or T-CR7 Wonders Touris, T- Neon Reisen GmbH, T- Kurban Tours, "T- Kalanit Tours Dubai flag it AND NO DEPOSIT OR IN THE COLUMN FLAG IT, IF NO PAYMENTI
6.Booking is more than 10 days flag it
7.show the check check out date by of companies. in a trend cruve by count. show Table and graph next toit 

8. have KPIS FOR ARRIVAL DATES.
