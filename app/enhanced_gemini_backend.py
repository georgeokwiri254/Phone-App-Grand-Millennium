"""
Enhanced Gemini AI Backend for Interactive Revenue Analytics
Provides conversational AI with context awareness, insights, and dynamic suggestions
"""

import sqlite3
import pandas as pd
import google.generativeai as genai
import os
import re
import json
from typing import Tuple, Optional, Dict, Any, List
import traceback
from datetime import datetime, timedelta
import numpy as np


class EnhancedGeminiBackend:
    """Enhanced conversational AI backend with context awareness and intelligent insights"""
    
    def __init__(self, api_key: str, db_path: str = "db/revenue.db"):
        """Initialize the Enhanced Gemini backend"""
        self.api_key = api_key
        self.db_path = db_path
        self.model = None
        self.conversation_history = []
        self.context_memory = {}
        self.user_preferences = {}
        
        # Configure Gemini
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.connected = True
        except Exception as e:
            self.connected = False
            self.error_message = f"Failed to initialize Gemini: {str(e)}"
    
    def get_enhanced_database_schema(self) -> str:
        """Get comprehensive database schema with sample data and insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            schema_info = """📊 GRAND MILLENNIUM HOTEL REVENUE DATABASE SCHEMA

🏨 MAIN BUSINESS TABLES:
======================

📈 DAILY PERFORMANCE:
- hotel_data_combined: Complete daily metrics with 60+ advanced features
  Key columns: Date, Rooms_Sold, Revenue, ADR, RevPAR, Occupancy_Pct
  Advanced: Moving averages (7/14/30 day), lag features, scaled/normalized data
  
- occupancy_analysis: Core daily occupancy metrics
  Columns: Date, DOW, Rms, Daily_Revenue, Occ_Pct, Rm_Sold, Revenue, ADR, RevPar

📊 HISTORICAL DATA (2022-2024):
- historical_occupancy_YYYY: Year-specific occupancy and revenue data
- historical_segment_YYYY: Business segment performance by year
  
🎯 SEGMENT ANALYSIS:
- segment_analysis: Current segment performance with forecasting
  Includes: Daily pickup, MTD, BOB (Business on Books), YoY comparisons, budget vs actual
  
🚀 RESERVATIONS & ARRIVALS:
- entered_on: Future reservations and booking patterns
  Key fields: Lead time, seasonal patterns, company analysis, monthly splits
  
- arrivals: Guest arrival details and patterns
  Fields: Company analysis, deposit status, rate codes, seasonal flags

🏢 BLOCK BUSINESS:
- block_analysis: Group bookings and corporate blocks
  
📈 FORECASTING:
- forecasts: AI-generated predictions with confidence intervals
- combined_forecast_data: Integrated forecast results

💡 KEY REVENUE METRICS:
- ADR: Average Daily Rate
- RevPAR: Revenue Per Available Room  
- BOB: Business on the Books (future reservations)
- STL: Same Time Last Year comparisons
- YoY: Year over Year performance

🔍 BUSINESS INTELLIGENCE FEATURES:
- Seasonal analysis (high/low seasons)
- Day-of-week patterns
- Lead time analysis
- Company/segment performance
- Forecasting with multiple horizons
"""
            
            # Get recent data samples for context
            cursor.execute("SELECT COUNT(*) FROM hotel_data_combined")
            total_days = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(Date), MAX(Date) FROM hotel_data_combined")
            date_range = cursor.fetchone()
            
            schema_info += f"\n📅 DATA COVERAGE: {total_days} days from {date_range[0]} to {date_range[1]}\n"
            
            conn.close()
            return schema_info
            
        except Exception as e:
            return f"Error getting enhanced schema: {str(e)}"
    
    def analyze_data_patterns(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Analyze data for patterns, trends, and insights"""
        if df.empty:
            return {"insights": [], "patterns": [], "recommendations": []}
        
        insights = []
        patterns = []
        recommendations = []
        
        try:
            # Revenue analysis
            if 'Revenue' in df.columns or 'Daily_Revenue' in df.columns:
                revenue_col = 'Revenue' if 'Revenue' in df.columns else 'Daily_Revenue'
                total_revenue = df[revenue_col].sum()
                avg_revenue = df[revenue_col].mean()
                
                insights.append(f"💰 Total Revenue: ${total_revenue:,.0f}")
                insights.append(f"📊 Average Daily Revenue: ${avg_revenue:,.0f}")
                
                # Trend analysis
                if len(df) > 1:
                    revenue_trend = "increasing" if df[revenue_col].iloc[-1] > df[revenue_col].iloc[0] else "decreasing"
                    patterns.append(f"📈 Revenue trend is {revenue_trend}")
            
            # Occupancy analysis
            if 'Occupancy_Pct' in df.columns or 'Occ_Pct' in df.columns:
                occ_col = 'Occupancy_Pct' if 'Occupancy_Pct' in df.columns else 'Occ_Pct'
                avg_occ = df[occ_col].mean()
                max_occ = df[occ_col].max()
                min_occ = df[occ_col].min()
                
                insights.append(f"🏨 Average Occupancy: {avg_occ:.1f}%")
                
                if avg_occ > 80:
                    patterns.append("🔥 High occupancy period - strong demand")
                    recommendations.append("Consider dynamic pricing strategies")
                elif avg_occ < 60:
                    patterns.append("📉 Lower occupancy - opportunity for improvement")
                    recommendations.append("Review marketing strategies and rate positioning")
            
            # ADR analysis
            if 'ADR' in df.columns:
                avg_adr = df['ADR'].mean()
                insights.append(f"💵 Average ADR: ${avg_adr:.0f}")
                
                if avg_adr > 200:
                    patterns.append("💎 Premium rate positioning")
                elif avg_adr < 150:
                    patterns.append("🎯 Value-focused pricing")
            
            # Day of week patterns
            if 'DOW' in df.columns:
                dow_performance = df.groupby('DOW').agg({
                    col: 'mean' for col in df.columns if col in ['Revenue', 'Daily_Revenue', 'Occupancy_Pct', 'Occ_Pct', 'ADR']
                })
                if not dow_performance.empty:
                    best_dow = dow_performance.iloc[:, 0].idxmax()
                    patterns.append(f"📅 Best performing day: {best_dow}")
            
            # Segment analysis
            if 'Segment' in df.columns:
                if 'Business_on_the_Books_Revenue' in df.columns:
                    top_segment = df.loc[df['Business_on_the_Books_Revenue'].idxmax(), 'Segment']
                    patterns.append(f"🎯 Top revenue segment: {top_segment}")
                    
                elif 'Business_on_the_Books_Rooms' in df.columns:
                    top_segment = df.loc[df['Business_on_the_Books_Rooms'].idxmax(), 'Segment']
                    patterns.append(f"🏨 Top volume segment: {top_segment}")
            
            # General recommendations
            if not recommendations:
                if 'question' in question.lower() and any(word in question.lower() for word in ['forecast', 'predict', 'future']):
                    recommendations.append("📈 Consider reviewing forecasting models for better accuracy")
                elif any(word in question.lower() for word in ['segment', 'market']):
                    recommendations.append("🎯 Analyze segment mix optimization opportunities")
                elif any(word in question.lower() for word in ['revenue', 'adr', 'rate']):
                    recommendations.append("💰 Review pricing strategies and revenue optimization")
        
        except Exception as e:
            insights.append(f"⚠️ Analysis error: {str(e)}")
        
        return {
            "insights": insights,
            "patterns": patterns,
            "recommendations": recommendations
        }
    
    def generate_follow_up_suggestions(self, question: str, results: pd.DataFrame) -> List[str]:
        """Generate intelligent follow-up questions based on context"""
        suggestions = []
        
        try:
            question_lower = question.lower()
            
            # Revenue-focused follow-ups
            if any(word in question_lower for word in ['revenue', 'sales', 'income']):
                suggestions.extend([
                    "💡 What's driving the revenue trends?",
                    "📊 How does this compare to last year?",
                    "🎯 Which segments contribute most to revenue?",
                    "📈 Show me revenue by day of week patterns"
                ])
            
            # Occupancy-focused follow-ups
            elif any(word in question_lower for word in ['occupancy', 'rooms', 'sold']):
                suggestions.extend([
                    "🏨 What's the average length of stay?",
                    "📅 Show seasonal occupancy patterns",
                    "🎯 Which rate codes have highest occupancy?",
                    "💰 How does occupancy correlate with ADR?"
                ])
            
            # Segment-focused follow-ups
            elif any(word in question_lower for word in ['segment', 'market', 'business']):
                suggestions.extend([
                    "📊 Compare segment performance year-over-year",
                    "🎯 What's the ADR by segment?",
                    "📈 Show segment booking lead times",
                    "💼 Which segments have strongest growth?"
                ])
            
            # Time-based follow-ups
            elif any(word in question_lower for word in ['month', 'year', 'quarter', 'week']):
                suggestions.extend([
                    "📅 Compare with same period last year",
                    "📊 Show monthly trending patterns",
                    "🔍 Analyze weekend vs weekday performance",
                    "📈 What are the seasonal variations?"
                ])
            
            # Forecasting follow-ups
            elif any(word in question_lower for word in ['forecast', 'predict', 'future']):
                suggestions.extend([
                    "🔮 Show confidence intervals for forecasts",
                    "📊 Compare forecast accuracy over time",
                    "🎯 What factors influence forecast accuracy?",
                    "📈 Generate longer-term forecasts"
                ])
            
            # General business intelligence
            else:
                suggestions.extend([
                    "🎯 Analyze top performing segments",
                    "📊 Show revenue optimization opportunities",
                    "📈 Compare current vs budget performance",
                    "💡 Identify booking pattern insights"
                ])
            
            # Add data-driven suggestions based on results
            if not results.empty:
                if 'Segment' in results.columns and len(results['Segment'].unique()) > 1:
                    suggestions.append("🔍 Deep dive into segment performance")
                
                if any(col in results.columns for col in ['ADR', 'RevPAR']):
                    suggestions.append("💰 Analyze pricing optimization opportunities")
                
                if 'Date' in results.columns and len(results) > 7:
                    suggestions.append("📊 Show trend analysis and patterns")
        
        except Exception:
            suggestions = [
                "📊 Show overall performance summary",
                "🎯 Analyze segment breakdown",
                "📈 Display trending patterns",
                "💡 Generate business insights"
            ]
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    def generate_enhanced_response(self, question: str, sql_query: str, results: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive response with insights and suggestions"""
        
        # Analyze data patterns
        analysis = self.analyze_data_patterns(results, question)
        
        # Generate follow-up suggestions
        suggestions = self.generate_follow_up_suggestions(question, results)
        
        # Create natural language summary
        summary = self.create_intelligent_summary(question, results, analysis)
        
        return {
            "sql_query": sql_query,
            "results": results,
            "summary": summary,
            "insights": analysis["insights"],
            "patterns": analysis["patterns"],
            "recommendations": analysis["recommendations"],
            "follow_up_suggestions": suggestions,
            "row_count": len(results)
        }
    
    def create_intelligent_summary(self, question: str, results: pd.DataFrame, analysis: Dict) -> str:
        """Create intelligent natural language summary of results"""
        if results.empty:
            return "🔍 Your query executed successfully but returned no data. This might indicate:\n• The specified criteria don't match any records\n• The time period might not have data\n• Consider adjusting filters or date ranges"
        
        summary_parts = []
        
        # Question context
        summary_parts.append(f"📋 **Analysis Results for**: {question}")
        
        # Data overview
        if len(results) == 1:
            summary_parts.append(f"🎯 Found **1 record** matching your criteria")
        else:
            summary_parts.append(f"📊 Found **{len(results)} records** matching your criteria")
        
        # Key insights
        if analysis["insights"]:
            summary_parts.append("💡 **Key Metrics**:")
            for insight in analysis["insights"][:3]:  # Top 3 insights
                summary_parts.append(f"  • {insight}")
        
        # Important patterns
        if analysis["patterns"]:
            summary_parts.append("🔍 **Key Patterns**:")
            for pattern in analysis["patterns"][:2]:  # Top 2 patterns
                summary_parts.append(f"  • {pattern}")
        
        # Business recommendations
        if analysis["recommendations"]:
            summary_parts.append("💡 **Recommendations**:")
            for rec in analysis["recommendations"][:2]:  # Top 2 recommendations
                summary_parts.append(f"  • {rec}")
        
        return "\n".join(summary_parts)
    
    def generate_contextual_sql(self, question: str) -> Tuple[bool, str, str]:
        """Generate SQL with enhanced context awareness"""
        try:
            if not self.connected:
                return False, "", "Gemini AI not connected"
            
            # Get enhanced schema
            schema = self.get_enhanced_database_schema()
            
            # Build conversation context
            context = ""
            if self.conversation_history:
                context = "\n\n📝 CONVERSATION CONTEXT:\n"
                for i, msg in enumerate(self.conversation_history[-3:]):  # Last 3 messages
                    context += f"{i+1}. User: {msg.get('question', '')}\n"
                    if msg.get('successful_sql'):
                        context += f"   SQL: {msg['successful_sql'][:100]}...\n"
            
            # Create enhanced prompt
            prompt = f"""
You are an expert AI assistant for Grand Millennium Hotel's revenue analytics. You have deep knowledge of hospitality industry KPIs and hotel operations.

{schema}

{context}

CONVERSATION INTELLIGENCE:
- Remember previous queries and build upon them
- Suggest related insights and deeper analysis
- Use hospitality domain expertise
- Focus on actionable business intelligence

ADVANCED SQL GENERATION RULES:
1. ONLY generate SELECT queries (no INSERT, UPDATE, DELETE, DROP)
2. Use appropriate JOINs when connecting related data
3. Apply proper date filtering with SQLite date functions
4. Use aggregations (SUM, COUNT, AVG, MAX, MIN) for business metrics
5. Include ORDER BY for meaningful result ranking
6. Use proper GROUP BY for segment/time-based analysis
7. Apply LIMIT when appropriate to prevent overwhelming results
8. Consider seasonal patterns and business cycles
9. Format dates properly for display
10. Use meaningful column aliases for clarity

HOSPITALITY DOMAIN KNOWLEDGE:
- ADR = Revenue / Rooms Sold
- RevPAR = Revenue / Available Rooms = ADR × Occupancy%
- Occupancy% = Rooms Sold / Available Rooms × 100
- Business on Books (BOB) = Future confirmed revenue
- Lead time = Days between booking and arrival
- High season typically: Oct-Apr for Dubai
- Weekend = Friday-Saturday in Middle East

QUESTION: {question}

Generate a precise SQL query that provides actionable business intelligence:
"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()
            
            # Clean up the response
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Final safety check
            if not self.is_safe_query(sql_query):
                return False, "", "Generated query is not safe"
            
            return True, sql_query, ""
            
        except Exception as e:
            error_msg = f"Enhanced SQL generation error: {str(e)}"
            return False, "", error_msg
    
    def is_safe_query(self, sql_query: str) -> bool:
        """Enhanced safety check for SQL queries"""
        # Remove comments and normalize whitespace
        query_clean = re.sub(r'--.*?$', '', sql_query, flags=re.MULTILINE)
        query_clean = re.sub(r'/\*.*?\*/', '', query_clean, flags=re.DOTALL)
        query_clean = query_clean.strip().upper()
        
        # Check for dangerous keywords
        dangerous_keywords = [
            'DELETE', 'UPDATE', 'INSERT', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'REPLACE', 'EXEC', 'EXECUTE', 'PRAGMA',
            'ATTACH', 'DETACH'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_clean:
                return False
        
        # Must start with SELECT
        if not query_clean.startswith('SELECT'):
            return False
        
        return True
    
    def query_database(self, sql_query: str) -> Tuple[bool, pd.DataFrame, str]:
        """Execute SQL query with enhanced error handling"""
        try:
            if not self.is_safe_query(sql_query):
                return False, pd.DataFrame(), "Query not allowed. Only SELECT statements are permitted."
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            return True, df, ""
            
        except Exception as e:
            error_msg = f"Database query error: {str(e)}"
            return False, pd.DataFrame(), error_msg
    
    def process_enhanced_question(self, question: str) -> Dict[str, Any]:
        """Process question with full conversational intelligence"""
        result = {
            'success': False,
            'sql_query': '',
            'results': pd.DataFrame(),
            'summary': '',
            'insights': [],
            'patterns': [],
            'recommendations': [],
            'follow_up_suggestions': [],
            'error_message': '',
            'row_count': 0,
            'processing_time': datetime.now()
        }
        
        try:
            # Validate input
            if not question or question.strip() == "":
                result['error_message'] = "Please enter a question"
                return result
            
            # Generate contextual SQL
            sql_success, sql_query, sql_error = self.generate_contextual_sql(question)
            
            if not sql_success:
                result['error_message'] = f"SQL generation failed: {sql_error}"
                return result
            
            # Execute query
            query_success, df, query_error = self.query_database(sql_query)
            
            if not query_success:
                result['error_message'] = f"Query execution failed: {query_error}"
                return result
            
            # Generate enhanced response
            enhanced_response = self.generate_enhanced_response(question, sql_query, df)
            
            # Update result with enhanced data
            result.update(enhanced_response)
            result['success'] = True
            
            # Update conversation history
            self.conversation_history.append({
                'question': question,
                'successful_sql': sql_query,
                'result_count': len(df),
                'timestamp': datetime.now()
            })
            
            # Limit conversation history to last 10 interactions
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return result
            
        except Exception as e:
            result['error_message'] = f"Unexpected error: {str(e)}"
            return result
    
    def get_conversation_insights(self) -> Dict[str, Any]:
        """Generate insights about the conversation patterns"""
        if not self.conversation_history:
            return {"total_queries": 0, "common_topics": [], "suggestions": []}
        
        topics = []
        total_queries = len(self.conversation_history)
        
        for conv in self.conversation_history:
            question = conv['question'].lower()
            if any(word in question for word in ['revenue', 'sales']):
                topics.append('Revenue Analysis')
            elif any(word in question for word in ['occupancy', 'rooms']):
                topics.append('Occupancy Analysis')
            elif any(word in question for word in ['segment', 'market']):
                topics.append('Segment Analysis')
            elif any(word in question for word in ['forecast', 'predict']):
                topics.append('Forecasting')
        
        from collections import Counter
        common_topics = [topic for topic, count in Counter(topics).most_common(3)]
        
        return {
            "total_queries": total_queries,
            "common_topics": common_topics,
            "suggestions": [
                "🔍 Try exploring different time periods",
                "📊 Compare performance metrics across segments",
                "📈 Analyze seasonal patterns and trends"
            ]
        }


def create_enhanced_gemini_backend(api_key: str, db_path: str = "db/revenue.db") -> EnhancedGeminiBackend:
    """Factory function to create Enhanced Gemini backend instance"""
    return EnhancedGeminiBackend(api_key, db_path)