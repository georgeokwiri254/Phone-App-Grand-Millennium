# Grand Millennium Revenue Analytics - Streamlit Dashboard Project

## Project Overview
Single Excel upload → run two existing converters → produce canonical CSVs → ingest into SQLite → Streamlit dashboard with EDA + Forecasting capabilities.

## 10-Point Project Summary

### 1. **Project Goal & Flow**
Single Excel upload → run two existing converters (`corrected_final_mhr_processor.py` for segment analysis and `mhr_occupancy_converter.py` for occupancy) → produce two canonical CSVs (`data/processed/segment.csv`, `data/processed/occupancy.csv`) → ingest into SQLite `db/revenue.db` → Streamlit dashboard reads DB (one Loading tab only) → EDA + Daily Occupancy + Segment Analysis + ADR Analysis + Forecasting.

### 2. **Exact Tools (recommended versions)**
- Python 3.10+
- Pandas (>=1.5), NumPy (>=1.23) — data handling
- Streamlit (>=1.20) — UI
- Plotly (plotly.express) — interactive charts
- statsmodels (>=0.14) — ExponentialSmoothing / time series models
- scikit-learn (>=1.1) — metrics/utilities
- sqlite3 (Python stdlib) — local DB
- PyInstaller (>=5.9) — packaging
- pywebview (optional, for native window) — optional to embed the Streamlit UI
- joblib — model persistence (optional)
- pytest — unit tests
- black / flake8 / mypy — linting & type checks
- logging + RotatingFileHandler (stdlib) — runtime logs

### 3. **Files & Naming Conventions**
- Raw upload: Any xlsm file with DPR sheet or via UI upload
- Segment converter (existing): `converters/corrected_final_mhr_processor.py` — must return `(df, csv_path)` or write `data/processed/segment.csv`
- Occupancy converter: `converters/mhr_occupancy_converter.py` — must return `(df, csv_path)` or write `data/processed/occupancy.csv`
- Canonical outputs (always these filenames after successful load):
  - `data/processed/segment.csv`
  - `data/processed/occupancy.csv`
- DB: `db/revenue.db` (contains tables `segment_analysis`, `occupancy_analysis`, `block_analysis`)
- Logs: `logs/conversion.log` and `logs/app.log`

### 4. **SQLite: schema, ingestion & indexing**
- Ingest CSVs via `pandas.DataFrame.to_sql(table_name, conn, if_exists='replace', index=False)`
- Recommended schema:
  - `segment_analysis`: `Month (DATE)`, `CustomerSegments (TEXT)`, `MergedSegment (TEXT)`, `BusinessOnTheBooksRevenue (REAL)`, `RoomsSold (INTEGER)`, `ADR (REAL)`, ...
  - `occupancy_analysis`: `Date (DATE)`, `DOW (TEXT)`, `Rooms (INTEGER)`, `DailyRevenue (REAL)`, `OccPct (REAL)`, `RoomsSold (INTEGER)`, `ADR (REAL)`, `RevPar (REAL)`
  - `block_analysis`: `BlockSize (INTEGER)`, `AllotmentDate (DATE)`, `SrepCode (TEXT)`, `BookingStatus (TEXT)`, `CompanyName (TEXT)`, `BeginDate (DATE)`, `Year (INTEGER)`, `Month (INTEGER)`, `Quarter (INTEGER)`
- Create indexes after ingest:
  - `CREATE INDEX IF NOT EXISTS idx_segment_month ON segment_analysis(Month);`
  - `CREATE INDEX IF NOT EXISTS idx_occ_date ON occupancy_analysis(Date);`
  - `CREATE INDEX IF NOT EXISTS idx_block_allotment_date ON block_analysis(AllotmentDate);`
  - `CREATE INDEX IF NOT EXISTS idx_block_booking_status ON block_analysis(BookingStatus);`
  - `CREATE INDEX IF NOT EXISTS idx_block_company ON block_analysis(CompanyName);`

### 5. **Segment Mapping (Critical)**
Group segments and pivot as follows:
- Unmanaged/ Leisure - Pre → Retail
- Unmanaged/ Leisure - Dis → Retail
- Package → Retail
- Third Party Intermediary → Retail
- Managed Corporate - Global → Corporate
- Managed Corporate - Local → Corporate
- Government → Corporate
- Wholesale Fixed Value → Leisure
- Corporate group → Groups
- Convention → Groups
- Association → Groups
- AD Hoc Group → Groups
- Tour Group → Groups
- Contract → Contract
- Complimentary → Complimentary

### 6. **Streamlit app behavior & single Loading tab rule**
- **Only** the **Loading** tab calls the converters and writes CSVs + ingests DB
- Loading tab UI: file upload control, `Run Converters` button, progress bar, textual log area, preview (top 5 rows for both CSVs)
- Use `st.cache_data` / `st.session_state` to cache DataFrames after Loading
- If Loading hasn't run or DB/CSVs missing, other tabs show prominent message and button linking back to Loading

### 7. **Data Processing**
- Process raw numeric data without currency conversion
- All monetary values are treated as raw numbers
- Display values with AED labels for presentation only

### 8. **Packaging with PyInstaller + optional PyWebView**
- **Launcher approach**: create `launcher.py` that runs Streamlit headless server via subprocess then opens pywebview window
- **PyInstaller build command (Linux):**
```bash
pyinstaller --onefile launcher.py \
  --add-data "app:app" \
  --add-data "db:db" \
  --add-data "data:./data" \
  --add-data "logs:logs"
```

### 9. **Reliability & bug reduction practices**
- **Atomic writes:** write CSVs to `*.tmp` then `os.replace(tmp, final)`
- **Validation:** validate required columns, date ranges, non-negative numeric fields, occupancy ≤ 100
- **Caching:** use Streamlit caching for DataFrames
- **Logging:** `logs/conversion.log` + `logs/app.log` with `RotatingFileHandler`
- **Unit tests:** pytest for converters with fixture files
- Include tests for segment merge mapping

### 10. **Forecasting & EDA tools & method**
- **Segment forecast (BOB Revenue):** monthly series → ExponentialSmoothing → forecast next **3 calendar months from today**
- **Occupancy forecast:** daily occupancy → short horizon (today → end-of-month)
- **EDA visuals:** Plotly for histograms, boxplots (ADR), time series, correlation heatmaps
- Persist forecast results in DB table `forecasts` with `run_ts`, `target`, `horizon`, `values`

## Streamlit Tab Specifications

### Loading Tab (ONLY place that runs converters)
- File uploader for Excel files
- `Run Converters` button
- Progress bar and streaming log area
- Preview of top 5 rows for each CSV
- Success/failure alerts
- Timestamped backups in `data/processed/`
- Atomic write to canonical filenames
- CSV validation with revert on failure
- SQLite ingestion with index creation
- Cache DataFrames using `st.cache_data`

### Daily Occupancy Tab
- Table (paginated): `Date, DOW, Rooms, Rooms Sold, Revenue (AED), ADR (AED), Occ%`
- Line chart Occ% with date range and MergedSegment filters
- Forecast occupancy from today to month-end with confidence ribbon
- Validation metrics (missing days, occupancy>100 warnings)

### Segment Analysis Tab
- Time-series of `BusinessOnTheBooksRevenue (AED)` by `MergedSegment`
- Aggregation selector (daily/weekly/monthly)
- Top 5 segments by revenue
- MoM growth KPI

### ADR Analysis Tab
- Boxplot ADR by `MergedSegment` showing medians, quartiles, outliers
- Histogram ADR distribution with bin slider
- Summary stats (count, mean, median, std, min, 25th, 75th, max)
- Toggle between merged segments and original segments

### Block Analysis Tab
- File upload interface for TXT block data files
- Calendar heatmap showing companies vs dates with block sizes
- EDA visualizations: booking status distribution, monthly trends, top companies
- Advanced filtering by booking status, company, and date range
- Data overview with key metrics and interactive charts
- Weekly pattern analysis and comprehensive data table

### Block Dashboard Tab
- KPI dashboard with total blocks, confirmed blocks, prospect blocks, conversion rate
- Interactive analytics: block booking timeline, sales rep performance
- Business mix analysis by booking status
- Pipeline analysis with future bookings and status funnel
- Real-time KPI updates and performance tracking

### Events Analysis Tab
- Dubai Events Calendar with major events (Aug 2025 - Jan 2026)
- Events timeline visualization with interactive hover details
- Occupancy-events correlation analysis with adjustable threshold
- Block bookings analysis during event periods
- Cross-referencing between ALLOTMENT_DATE and BEGIN_DATE with events
- Daily occupancy charts with event period overlays
- Smart detection of high occupancy periods coinciding with events

### Controls/Logs Tab
- Last 100 lines of `logs/conversion.log` and `logs/app.log`
- Last-run timestamp
- CSV paths
- Button to re-run Loading
- Re-run must re-build cache and refresh all tabs

## Project Structure
```
Revenue Architecture/
├── app/
│   └── streamlit_app.py
├── converters/
│   ├── corrected_final_mhr_processor.py (existing)
│   ├── mhr_occupancy_converter.py (existing)
│   └── block_converter.py (new - block data processing)
├── data/
│   ├── processed/
│   │   ├── segment.csv
│   │   ├── occupancy.csv
│   │   └── block_data_*.csv (timestamped block data outputs)
│   └── raw/
├── db/
│   └── revenue.db
├── logs/
│   ├── conversion.log
│   └── app.log
├── models/
├── tests/
│   └── fixtures/
├── config/
│   └── exchange_rates.json
├── scripts/
│   └── launcher.py
├── requirements.txt
├── claude.md
└── README.md
```

## Implementation Tasks Breakdown

1. **Setup project structure and dependencies**
2. **Analyze existing converters and create wrapper functions**
3. **Implement SQLite database schema and ingestion**
4. **Create Loading tab with file upload and converter execution**
5. **Implement segment mapping**
6. **Build Daily Occupancy tab with charts and forecasting**
7. **Build Segment Analysis tab with time-series and KPIs**
8. **Build ADR Analysis tab with statistical visualizations**
9. **Implement Block Data functionality (converter, analysis, dashboard)**
10. **Build Events Analysis with Dubai events calendar and correlation**
11. **Add calendar heatmap for company-date-block visualization**
12. **Build Controls/Logs tab for monitoring**
13. **Implement forecasting models (ExponentialSmoothing)**
14. **Add logging, validation, and error handling**
15. **Create launcher.py and PyInstaller packaging**
16. **Write unit tests and documentation**
17. **Final testing and deployment preparation**

## Block Data & Events Features

### Block Data Processing
- **File Format**: Tab-separated TXT files with columns: BLOCKSIZE, ALLOTMENT_DATE, SREP_CODE, BOOKING_STATUS, DESCRIPTION
- **Booking Status Codes**: ACT (Actual), DEF (Definite), PSP (Prospect), TEN (Tentative)
- **Converter**: `converters/block_converter.py` handles data cleaning, validation, and standardization
- **Database Integration**: Automatic ingestion into `block_analysis` table with proper indexing

### Dubai Events Integration
- **Events Tracked**: 8 major Dubai events (Aug 2025 - Jan 2026)
  - Dubai International Film Festival, Sleep Expo, GITEX Global, Dubai Shopping Festival, etc.
- **Smart Analytics**: Correlates events with occupancy spikes and block bookings
- **Cross-Reference Analysis**: Tracks both ALLOTMENT_DATE and BEGIN_DATE patterns during events
- **Visual Indicators**: Event periods highlighted on occupancy charts with interactive overlays

### Calendar Heatmap Features
- **Structure**: Companies (rows) × Dates (columns) × Block Sizes (color intensity)
- **Based on**: Group Forecast Excel structure with top 20 companies displayed
- **Interactive**: Hover details showing company, date, and block size information
- **Performance**: Optimized for large datasets with intelligent filtering and aggregation

## Acceptance Criteria
- Running Loading produces canonical CSVs and updates SQLite
- All other tabs render without invoking converters
- Values displayed with AED labels for presentation
- PyInstaller packaging runs on clean machine
- Unit tests pass
- Logs present with recent run entries

## Commands to Remember
```bash
# Install dependencies
pip install pandas streamlit plotly statsmodels scikit-learn pyinstaller pywebview pytest black flake8 mypy

# Run tests
pytest tests/

# Lint code
black . && flake8 . && mypy .

# Package application
pyinstaller --onefile launcher.py --add-data "app:app" --add-data "db:db" --add-data "data:./data" --add-data "logs:logs"

# Run application
python scripts/launcher.py
```