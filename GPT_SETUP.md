# 🤖 GPT Tab Setup Guide

## Overview
The GPT tab enables natural language queries to your hotel revenue database using Google's Gemini AI. Ask questions in plain English and get SQL-powered answers!

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install google-generativeai
```
Or install from requirements:
```bash
pip install -r requirements-gemini.txt
```

### 2. Get Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key

### 3. Update API Key
Edit `app/streamlit_app_simple.py` line 5244:
```python
api_key = "YOUR_GEMINI_API_KEY_HERE"  # Replace with your actual key
```

## 💬 Example Questions

Ask natural language questions about your revenue data:

### Revenue Queries
- "What was the total revenue last month?"
- "Show me revenue trends for the past 6 months"
- "Compare this year's revenue to last year"

### Occupancy Analysis
- "What's the average occupancy rate this year?"
- "Show me occupancy by day of week"
- "Which months have the highest occupancy?"

### Segment Performance
- "What are the top 5 performing segments?"
- "Show me segment revenue breakdown"
- "Which segment has the highest ADR?"

### ADR & RevPAR Analysis
- "What's the average ADR for weekends vs weekdays?"
- "Show me RevPAR trends over time"
- "Compare ADR by month"

## 📊 Available Data Tables

The GPT can query these main tables:

- **`occupancy_analysis`** - Daily occupancy metrics
- **`segment_analysis`** - Market segment performance  
- **`hotel_data_combined`** - Comprehensive daily data
- **`historical_occupancy_YYYY`** - Historical occupancy by year
- **`historical_segment_YYYY`** - Historical segment data
- **`entered_on`** - Reservation entries
- **`arrivals`** - Guest arrival data
- **`block_analysis`** - Group/block business

## 🛡️ Security Features

- **Read-Only Access**: Only SELECT queries allowed
- **SQL Injection Protection**: Dangerous operations blocked
- **Query Validation**: All queries checked before execution
- **Error Handling**: Clear error messages for failed queries

## 🎯 Tips for Better Results

1. **Be Specific**: "revenue last month" vs "total revenue in December 2024"
2. **Use Hotel Terms**: ADR, RevPAR, occupancy, segments
3. **Specify Time Periods**: "last 30 days", "this year", "Q1 2024"
4. **Ask for Comparisons**: "compare to last year", "vs previous month"

## 📋 Features

- ✅ Natural language to SQL conversion
- ✅ Real-time query execution  
- ✅ Conversation history
- ✅ SQL query display (transparent)
- ✅ Formatted results tables
- ✅ Loading indicators
- ✅ Error handling
- ✅ Clear chat option

## 🔧 Troubleshooting

### "Gemini AI backend not available"
```bash
pip install google-generativeai --break-system-packages
```

### "Database query error: no such table"
Check that `db/revenue.db` exists and contains data.

### "API error: 404 model not found"  
Model updated to `gemini-1.5-flash` (fixed in latest version).

### "Failed to connect to Gemini AI"
Verify your API key is correct and you have internet connection.

## 📝 Need Help?

1. Check the conversation history for previous successful queries
2. Try simpler questions first
3. Use the example questions as templates
4. Check the Tips panel in the GPT tab

---

**Enjoy exploring your hotel data with AI! 🏨✨**