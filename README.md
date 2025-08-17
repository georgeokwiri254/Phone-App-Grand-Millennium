# Grand Millennium Revenue Analytics Dashboard

A comprehensive Streamlit-based dashboard for analyzing MHR (Monthly Hotel Revenue) pickup reports from Grand Millennium hotels. Features data processing, visualization, forecasting, and interactive analytics.

## 🌟 Features

- **📁 Data Loading**: Upload and process Excel files with automated converters
- **📈 Daily Occupancy Analysis**: Interactive charts, forecasting, and validation metrics
- **🎯 Segment Analysis**: Revenue trends, KPIs, and month-over-month growth
- **💰 ADR Analysis**: Statistical distributions, outlier detection, and comparative analysis
- **🏨 Block Data Management**: TXT file processing for group bookings with calendar heatmaps
- **📅 Events Analysis**: Dubai events correlation with occupancy and booking patterns
- **🔥 Calendar Heatmap**: Company-date-block visualization inspired by Group Forecast structure
- **🔮 Forecasting**: ExponentialSmoothing models for occupancy and revenue prediction
- **📊 Interactive Dashboards**: Plotly-powered visualizations with real-time updates
- **🗄️ SQLite Backend**: Efficient data storage and retrieval
- **📝 Comprehensive Logging**: Detailed conversion and application logs
- **📦 Standalone Packaging**: PyInstaller executable for easy deployment

## 🚀 Quick Start

### Option 1: Run from Source

1. **Clone and Setup**

   ```bash
   cd "Revenue Architecture"
   pip install -r requirements.txt
   ```
2. **Launch Application**

   ```bash
   python scripts/launcher.py
   ```
3. **Access Dashboard**

   - PyWebView window opens automatically (if available)
   - Or visit http://localhost:8501 in your browser

### Option 2: Use Standalone Executable

1. **Build Executable**

   ```bash
   python build_executable.py
   ```
2. **Run Executable**

   ```bash
   cd dist
   ./GrandMillenniumAnalytics  # Linux/Mac
   # or
   GrandMillenniumAnalytics.exe  # Windows
   ```

## 📋 Requirements

### Core Dependencies

- Python 3.10+
- pandas (>=1.5), numpy (>=1.23)
- streamlit (>=1.20)
- plotly (>=5.14)
- statsmodels (>=0.14)
- scikit-learn (>=1.1)
- openpyxl (>=3.0)

### Optional Dependencies

- pywebview (>=4.0) - for native window
- pyinstaller (>=5.9) - for executable packaging
- pytest (>=7.0) - for testing

## 🏗️ Project Structure

```
Revenue Architecture/
├── app/                          # Main application
│   ├── streamlit_app_simple.py      # Streamlit dashboard
│   ├── database.py              # SQLite management
│   ├── forecasting.py           # Time series forecasting
│   └── logging_config.py        # Logging setup
├── converters/                   # Data converters
│   ├── corrected_final_mhr_processor.py  # Segment converter
│   ├── mhr_occupancy_converter.py        # Occupancy converter
│   ├── block_converter.py       # Block data converter
│   ├── segment_converter.py     # Segment wrapper
│   └── occupancy_converter.py   # Occupancy wrapper
├── data/                        # Data storage
│   ├── processed/               # Canonical output CSVs
│   └── raw/                     # Raw Excel files
├── db/                          # SQLite database
├── logs/                        # Application logs
├── models/                      # Saved forecast models
├── scripts/                     # Utilities
│   └── launcher.py              # Application launcher
├── tests/                       # Unit tests
├── requirements.txt             # Dependencies
├── build_executable.py          # PyInstaller build script
└── claude.md                    # Project documentation
```

## 📊 Dashboard Tabs

### 1. Loading Tab

- **File Upload**: Drag-and-drop or select Excel files
- **Converter Execution**: Run segment and occupancy processors
- **Progress Tracking**: Real-time progress bars and logging
- **Data Preview**: View processed data samples
- **Validation**: Automated data quality checks

### 2. Daily Occupancy Tab

- **Key Metrics**: Occupancy %, Revenue, ADR, RevPAR
- **Trend Charts**: Interactive time series with forecasting
- **Data Validation**: Missing days, high occupancy warnings
- **Forecasting**: ExponentialSmoothing predictions to month-end
- **Paginated Tables**: Formatted data display

### 3. Segment Analysis Tab

- **Performance KPIs**: Total and average revenue metrics
- **Top Segments**: Revenue rankings and market share
- **Time Series**: Revenue trends by segment
- **MoM Growth**: Month-over-month growth analysis
- **3-Month Forecast**: Revenue predictions by segment

### 4. ADR Analysis Tab

- **Statistical Summary**: Mean, median, quartiles, standard deviation
- **Distribution Charts**: Histograms with adjustable bins
- **Segment Boxplots**: ADR distribution by customer segment
- **Outlier Detection**: IQR-based outlier identification
- **Comparative Analysis**: Cross-dataset ADR comparison

### 5. Block Analysis Tab

- **File Upload**: Process TXT block data files with validation
- **Calendar Heatmap**: Companies vs dates visualization with block intensity
- **EDA Visualizations**: Booking status distribution, monthly trends, top companies
- **Advanced Filtering**: By booking status, company, and date range
- **Weekly Patterns**: Day-of-week analysis and comprehensive data tables

### 6. Block Dashboard Tab

- **KPI Metrics**: Total blocks, confirmed blocks, prospect blocks, conversion rates
- **Interactive Analytics**: Block booking timeline and sales rep performance
- **Business Mix Analysis**: Breakdown by booking status with statistics
- **Pipeline Analysis**: Future bookings forecasting and status funnel visualization

### 7. Events Analysis Tab

- **Dubai Events Calendar**: 8 major events (Aug 2025 - Jan 2026) timeline
- **Occupancy Correlation**: Smart detection of high occupancy periods during events
- **Block Booking Analysis**: Cross-reference ALLOTMENT_DATE and BEGIN_DATE with events
- **Visual Analytics**: Daily occupancy charts with event period overlays
- **Interactive Thresholds**: Adjustable occupancy thresholds for analysis

### 8. Controls & Logs Tab

- **Cache Management**: Clear and reload data
- **Database Statistics**: Row counts and last updated times
- **Log Viewers**: Real-time conversion and application logs
- **System Status**: Current data loading status

## 🔄 Data Processing Flow

### Primary Flow

1. **Excel Upload** → 2. **Segment Converter** → 3. **Occupancy Converter** → 4. **Segment Mapping** → 5. **SQLite Ingestion** → 6. **Dashboard Analytics**

### Block Data Flow

1. **TXT Upload** → 2. **Block Converter** → 3. **Data Cleaning & Validation** → 4. **SQLite Ingestion** → 5. **Block Analytics & Events Correlation**

### Segment Mapping

Original segments are mapped to consolidated categories:

- **Retail**: Unmanaged/Leisure, Package, Third Party
- **Corporate**: Managed Corporate, Government
- **Leisure**: Wholesale Fixed Value
- **Groups**: Corporate Group, Convention, Association, AD Hoc, Tour Group
- **Contract**: Contract bookings
- **Complimentary**: Complimentary stays

## 🔮 Forecasting Models

### Occupancy Forecasting

- **Method**: ExponentialSmoothing with trend
- **Horizon**: Today to end of current month
- **Confidence Intervals**: 95% confidence ribbons
- **Fallback**: Linear trend if insufficient data

### Revenue Forecasting

- **Method**: ExponentialSmoothing by segment
- **Horizon**: 3 months from today
- **Granularity**: Monthly predictions
- **Validation**: Historical accuracy metrics

## 🧪 Testing

Run unit tests to verify functionality:

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_converters.py
python -m pytest tests/test_database.py

# Run with coverage
python -m pytest tests/ --cov=app --cov=converters
```

## 📦 Building Executable

Create standalone executable for deployment:

```bash
# Build executable
python build_executable.py

# Output location
dist/GrandMillenniumAnalytics[.exe]

# Test executable
cd dist
./GrandMillenniumAnalytics --help
```

## 🔧 Configuration

### Launcher Options

```bash
python scripts/launcher.py --help

Options:
  --no-webview     Use browser instead of PyWebView
  --port PORT      Streamlit server port (default: 8501)
  --host HOST      Server host (default: 127.0.0.1)
  --debug          Enable debug logging
```

### Environment Variables

- `STREAMLIT_SERVER_PORT`: Override default port
- `PYTHONPATH`: Include project root for imports

## 📝 Logging

Two separate log files are maintained:

- **conversion.log**: Data converter operations
- **app.log**: Streamlit application events

Logs use rotating file handlers (10MB max, 5 backups) and are viewable in the dashboard.

## 🚨 Troubleshooting

### Common Issues

1. **PyWebView not opening**

   - Install: `pip install pywebview`
   - Use: `python scripts/launcher.py --no-webview`
2. **Conversion failures**

   - Check Excel file has DPR sheet
   - Verify file is not corrupted
   - Review conversion.log for details
3. **Memory issues with large files**

   - Close other applications
   - Process files individually
   - Use 64-bit Python
4. **Database connection errors**

   - Ensure db/ directory is writable
   - Check disk space
   - Restart application

### Performance Tips

- Use SSD storage for database files
- Close unused browser tabs when using web interface
- Process smaller date ranges for large datasets
- Clear cache periodically in Controls tab

## 📈 Data Specifications

### Input Requirements

- **Excel Format**: (.xlsm, .xlsx) with DPR sheet for segment/occupancy data
- **Block Data Format**: Tab-separated TXT files with columns:
  - BLOCKSIZE (integer): Number of rooms booked
  - ALLOTMENT_DATE: Date when blocks were made
  - SREP_CODE: Account owner/sales rep name
  - BOOKING_STATUS: ACT/DEF/PSP/TEN status codes
  - DESCRIPTION: Company name
- **Structure**: Standard MHR pickup report format
- **Columns**: Minimum AZ-BK range for occupancy data
- **Segments**: Must include standard 15 segments

### Output Formats

- **Segment CSV**: 180 rows (15 segments × 12 months), 49+ columns
- **Occupancy CSV**: 365 rows (daily), 9+ columns
- **Block Data CSV**: Variable rows (timestamped), standardized columns
- **Database**: SQLite with indexed tables (segment_analysis, occupancy_analysis, block_analysis)
- **Forecasts**: JSON/CSV export capability

## 🎉 Dubai Events Integration

The Events Analysis tab tracks 8 major Dubai events and their impact on hotel bookings:

### Events Covered (Aug 2025 - Jan 2026)

- **Dubai International Film Festival** (Aug 30, 2025)
- **Sleep Expo Middle East** (Sep 15-17, 2025)
- **Dubai PodFest** (Sep 30, 2025)
- **Gulf Food Manufacturing** (Sep 29 - Oct 1, 2025)
- **GITEX Global** (Oct 13-17, 2025) - Major technology event
- **Asia Pacific Cities Summit** (Oct 27-29, 2025)
- **Dubai Shopping Festival** (Dec 15, 2025 - Jan 29, 2026) - City-wide retail festival
- **World Sports Summit** (Dec 29-30, 2025)

### Analytics Features

- **Correlation Analysis**: Automatically detects bookings coinciding with events
- **Occupancy Impact**: Shows occupancy spikes during event periods
- **Booking Patterns**: Tracks both booking dates (ALLOTMENT_DATE) and stay dates (BEGIN_DATE)
- **Visual Timeline**: Interactive event calendar with booking overlays
- **Smart Thresholds**: Configurable occupancy thresholds for event impact analysis

## 📅 Calendar Heatmap Features

Based on the Group Forecast Excel structure, the calendar heatmap provides:

### Visualization Structure

- **Rows**: Top 20 companies by total block size
- **Columns**: Allotment dates (calendar view)
- **Color Intensity**: Block sizes (darker = more blocks)
- **Interactive**: Hover for company, date, and block details

### Use Cases

- **Capacity Planning**: Visualize booking density across time
- **Company Analysis**: Identify top block booking companies
- **Pattern Recognition**: Spot seasonal or event-driven booking patterns
- **Resource Allocation**: Plan staffing and services based on booking patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

## 📄 License

This project is created for Grand Millennium Revenue Analytics. All rights reserved.

## 👤 Author

**Created with Claude Code** for Grand Millennium Revenue Analytics

---

For technical support or questions, check the application logs or refer to the troubleshooting section above.
