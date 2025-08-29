"""
Enhanced Ollama AI Backend for Interactive Revenue Analytics
Provides conversational AI with context awareness, insights, and dynamic suggestions using local Ollama models
"""

import sqlite3
import pandas as pd
import requests
import json
import os
import re
from typing import Tuple, Optional, Dict, Any, List
import traceback
from datetime import datetime, timedelta
import numpy as np


class EnhancedOllamaBackend:
    """Enhanced conversational AI backend with context awareness and intelligent insights using Ollama"""
    
    def __init__(self, model_name: str = "phi3:mini", ollama_url: str = "http://localhost:11434", db_path: str = "db/revenue.db"):
        """Initialize the Enhanced Ollama backend"""
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.db_path = db_path
        self.conversation_history = []
        self.context_memory = {}
        self.user_preferences = {}
        self.connected = False
        self.error_message = ""
        
        # Test Ollama connection
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                if any(self.model_name in name for name in model_names):
                    self.connected = True
                else:
                    self.connected = False
                    self.error_message = f"Model '{self.model_name}' not found. Available models: {model_names}"
            else:
                self.connected = False
                self.error_message = f"Ollama server returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            self.connected = False
            self.error_message = f"Failed to connect to Ollama server: {str(e)}"
    
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
  Columns: Date, DOW, "Rm Sold", Revenue, ADR, RevPar, Year
  
- historical_segment_YYYY: Business segment performance by year
  Columns: Month, Segment, Business_on_the_Books_Rooms, Business_on_the_Books_Revenue, Business_on_the_Books_ADR, MergedSegment, Year
  ⚠️ CRITICAL: Use "Business_on_the_Books_Revenue" NOT "Revenue"
  
🎯 SEGMENT ANALYSIS:
- segment_analysis: Current segment performance with forecasting
  Revenue Columns: Business_on_the_Books_Revenue, Daily_Pick_up_Revenue, Month_to_Date_Revenue
  ADR Columns: Business_on_the_Books_ADR, Daily_Pick_up_ADR, Month_to_Date_ADR
  ⚠️ CRITICAL: NO simple "Revenue" column exists - must use specific revenue types
  
🚀 RESERVATIONS & ARRIVALS:
- entered_on: Future reservations and booking patterns
  Key columns: COMPANY_CLEAN (company name), C_T_S_NAME (travel agent), AMOUNT (revenue), TOTAL, NET, ADR
  ⚠️ CRITICAL: Use "COMPANY_CLEAN" NOT "Company", use "AMOUNT" NOT "Revenue"
  Lead time: BOOKING_LEAD_TIME, Season: SEASON
  
- arrivals: Guest arrival details and patterns
  Key columns: COMPANY_NAME, COMPANY_NAME_CLEAN, CALCULATED_ADR, AMOUNT
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
- Company/segment performance"""

            # Add actual table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            schema_info += "\n\n🔍 ACTUAL TABLES AVAILABLE:\n"
            for table in tables:
                table_name = table[0]
                schema_info += f"\n📋 {table_name}:\n"
                
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    schema_info += f"  • {col_name} ({col_type})\n"
            
            conn.close()
            return schema_info
            
        except Exception as e:
            return f"Error getting schema: {str(e)}"
    
    def generate_enhanced_sql_query(self, question: str, context: str = "") -> Tuple[str, List[str]]:
        """Generate SQL query with enhanced context and insights"""
        try:
            schema = self.get_enhanced_database_schema()
            
            # Build conversation context
            recent_context = ""
            if self.conversation_history:
                recent_context = "\n📝 RECENT CONVERSATION CONTEXT:\n"
                for entry in self.conversation_history[-3:]:  # Last 3 interactions
                    if entry.get('question'):
                        recent_context += f"Q: {entry['question']}\n"
                    if entry.get('insights'):
                        recent_context += f"Insights: {', '.join(entry['insights'][:2])}\n"
            
            prompt = f"""You are an expert SQL analyst for Grand Millennium Hotel's revenue analytics database. Generate a comprehensive SQLite query to answer the user's question with deep insights.

{schema}

{recent_context}

User Question: {question}
Additional Context: {context}

🎯 ENHANCED ANALYSIS GUIDELINES:
1. Generate a main SELECT query for the core answer
2. Consider time periods, trends, and comparisons
3. Include relevant aggregations (SUM, AVG, COUNT, MIN, MAX)
4. Add appropriate date filtering and GROUP BY clauses
5. Use window functions for trends when applicable
6. Consider seasonal patterns and YoY comparisons
7. Include business context (weekday vs weekend, high/low season)

🔒 SAFETY RULES:
- ONLY SELECT statements allowed
- Use proper SQLite syntax
- Return only the SQL query without explanation
- Use 'YYYY-MM-DD' format for dates
- Add LIMIT clause if result might be very large

🧠 INTELLIGENCE FEATURES:
- Think about follow-up questions users might ask
- Consider business implications of the data
- Identify patterns and anomalies
- Suggest actionable insights

SQL Query:"""

            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 512
                    }
                },
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                sql_query = result.get('response', '').strip()
                
                # Clean up the response
                sql_query = self.clean_sql_response(sql_query)
                
                # Generate follow-up suggestions
                suggestions = self.generate_follow_up_suggestions(question, sql_query)
                
                return sql_query, suggestions
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Error generating enhanced SQL: {str(e)}")
    
    def clean_sql_response(self, response: str) -> str:
        """Clean the SQL response from Ollama"""
        response = response.strip()
        
        # Remove markdown code blocks
        response = re.sub(r'```sql\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # Extract SQL query
        lines = response.split('\n')
        sql_lines = []
        found_select = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT') or found_select:
                found_select = True
                sql_lines.append(line)
                if line.endswith(';'):
                    break
        
        if sql_lines:
            sql_query = ' '.join(sql_lines)
        else:
            # Fallback: look for any SELECT statement
            select_match = re.search(r'(SELECT.*?(?:;|$))', response, re.IGNORECASE | re.DOTALL)
            if select_match:
                sql_query = select_match.group(1)
            else:
                sql_query = response
        
        return sql_query.strip()
    
    def generate_follow_up_suggestions(self, question: str, sql_query: str) -> List[str]:
        """Generate intelligent follow-up suggestions"""
        suggestions = []
        
        question_lower = question.lower()
        
        # Revenue-based suggestions
        if 'revenue' in question_lower:
            suggestions.extend([
                "💰 Compare this revenue to the same period last year",
                "📊 Show revenue breakdown by business segment",
                "📈 What are the revenue trends over the last 6 months?",
                "🎯 Which days of the week generate the highest revenue?"
            ])
        
        # Occupancy-based suggestions
        if 'occupancy' in question_lower or 'rooms' in question_lower:
            suggestions.extend([
                "🏨 What's the average ADR for these occupancy levels?",
                "📅 Compare weekday vs weekend occupancy patterns",
                "🎯 Show occupancy trends by business segment",
                "📊 How does this occupancy compare to budget targets?"
            ])
        
        # Segment-based suggestions
        if 'segment' in question_lower:
            suggestions.extend([
                "💼 Which segments have the highest growth rate?",
                "📈 Show segment performance trends over time",
                "🎯 What's the ADR difference between segments?",
                "📊 Which segments book the furthest in advance?"
            ])
        
        # Time-based suggestions
        if any(time_word in question_lower for time_word in ['month', 'year', 'week', 'day']):
            suggestions.extend([
                "📈 Show seasonal patterns for this metric",
                "🔄 Compare to same period last year",
                "📊 What are the monthly trends?",
                "🎯 Identify peak and low performance periods"
            ])
        
        # ADR/RevPAR suggestions
        if 'adr' in question_lower or 'revpar' in question_lower:
            suggestions.extend([
                "💰 How does this ADR compare to market competitors?",
                "📊 Show ADR trends by booking lead time",
                "🎯 Which room types drive the highest ADR?",
                "📈 What's the RevPAR optimization opportunity?"
            ])
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    def generate_business_insights(self, df: pd.DataFrame, question: str, sql_query: str) -> List[str]:
        """Generate intelligent business insights from query results"""
        insights = []
        
        if df.empty:
            return ["📭 No data found for the specified criteria. Consider adjusting the time period or filters."]
        
        try:
            # Revenue insights
            if 'revenue' in df.columns.str.lower().str.join(' '):
                revenue_cols = [col for col in df.columns if 'revenue' in col.lower()]
                if revenue_cols:
                    total_revenue = df[revenue_cols[0]].sum()
                    avg_revenue = df[revenue_cols[0]].mean()
                    insights.append(f"💰 Total revenue: AED {total_revenue:,.0f} | Average: AED {avg_revenue:,.0f}")
            
            # Occupancy insights
            if any('occ' in col.lower() for col in df.columns):
                occ_cols = [col for col in df.columns if 'occ' in col.lower()]
                if occ_cols:
                    avg_occ = df[occ_cols[0]].mean()
                    max_occ = df[occ_cols[0]].max()
                    insights.append(f"🏨 Average occupancy: {avg_occ:.1f}% | Peak: {max_occ:.1f}%")
            
            # ADR insights
            if any('adr' in col.lower() for col in df.columns):
                adr_cols = [col for col in df.columns if 'adr' in col.lower()]
                if adr_cols:
                    avg_adr = df[adr_cols[0]].mean()
                    insights.append(f"💎 Average ADR: AED {avg_adr:,.0f}")
            
            # Trend insights
            if 'date' in df.columns.str.lower().str.join(' ') and len(df) > 1:
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                if date_cols and len(df) >= 2:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        first_val = df[numeric_cols[0]].iloc[0]
                        last_val = df[numeric_cols[0]].iloc[-1]
                        if pd.notna(first_val) and pd.notna(last_val) and first_val != 0:
                            change_pct = ((last_val - first_val) / first_val) * 100
                            trend = "📈 increasing" if change_pct > 0 else "📉 decreasing"
                            insights.append(f"📊 Trend: {trend} by {abs(change_pct):.1f}% over period")
            
            # Segment insights
            if 'segment' in df.columns.str.lower().str.join(' ') and len(df) > 1:
                segment_cols = [col for col in df.columns if 'segment' in col.lower()]
                if segment_cols:
                    top_segment = df.loc[df.iloc[:, -1].idxmax(), segment_cols[0]] if len(df) > 0 else "N/A"
                    insights.append(f"🎯 Top performing segment: {top_segment}")
            
            # Performance insights
            if len(df) > 0:
                insights.append(f"📋 Dataset contains {len(df)} records spanning the analysis period")
            
        except Exception as e:
            insights.append("🔍 Additional insights not available due to data complexity")
        
        return insights[:5]  # Limit to 5 insights
    
    def generate_recommendations(self, df: pd.DataFrame, insights: List[str], question: str) -> List[str]:
        """Generate actionable business recommendations"""
        recommendations = []
        
        try:
            question_lower = question.lower()
            
            # Revenue optimization recommendations
            if 'revenue' in question_lower and not df.empty:
                recommendations.append("💡 Consider implementing dynamic pricing strategies during low-revenue periods")
                recommendations.append("🎯 Focus marketing efforts on high-performing segments identified")
            
            # Occupancy optimization
            if 'occupancy' in question_lower and not df.empty:
                recommendations.append("🏨 Optimize room inventory allocation based on demand patterns")
                recommendations.append("📈 Consider promotional campaigns for low-occupancy periods")
            
            # Segment optimization
            if 'segment' in question_lower and not df.empty:
                recommendations.append("💼 Strengthen relationships with top-performing segment partners")
                recommendations.append("🎯 Develop targeted packages for underperforming segments")
            
            # ADR optimization
            if 'adr' in question_lower and not df.empty:
                recommendations.append("💎 Review rate positioning against market competitors")
                recommendations.append("🎯 Implement value-added services to justify rate premiums")
            
            # General recommendations
            recommendations.append("📊 Monitor these metrics regularly for trend identification")
            
        except Exception:
            recommendations.append("📈 Regular monitoring of these metrics is recommended for optimization")
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    def process_enhanced_question(self, question: str) -> Dict[str, Any]:
        """Process question with enhanced analytics and insights"""
        try:
            # Generate enhanced SQL query and suggestions
            sql_query, follow_up_suggestions = self.generate_enhanced_sql_query(question)
            
            # Execute query
            success, results_df, error_msg = self.execute_sql_query(sql_query)
            
            if success:
                # Generate insights and recommendations
                insights = self.generate_business_insights(results_df, question, sql_query)
                recommendations = self.generate_recommendations(results_df, insights, question)
                
                # Create enhanced summary
                summary = self.create_enhanced_summary(results_df, question, insights)
                
                # Store in conversation history
                self.add_to_conversation_history(question, sql_query, insights, recommendations)
                
                return {
                    "success": True,
                    "sql_query": sql_query,
                    "results": results_df,
                    "row_count": len(results_df),
                    "question": question,
                    "summary": summary,
                    "insights": insights,
                    "recommendations": recommendations,
                    "follow_up_suggestions": follow_up_suggestions
                }
            else:
                return {
                    "success": False,
                    "sql_query": sql_query,
                    "error_message": error_msg,
                    "question": question,
                    "follow_up_suggestions": follow_up_suggestions
                }
                
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Enhanced processing error: {str(e)}",
                "question": question
            }
    
    def create_enhanced_summary(self, df: pd.DataFrame, question: str, insights: List[str]) -> str:
        """Create an intelligent summary of the analysis"""
        if df.empty:
            return "📭 **No Data Found**: The query returned no results. Consider adjusting your criteria or time period."
        
        summary_parts = []
        summary_parts.append(f"🎯 **Analysis Summary for:** {question}")
        summary_parts.append(f"📊 **Data Points:** {len(df)} records analyzed")
        
        # Add key metric if available
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            primary_metric = numeric_cols[0]
            total_value = df[primary_metric].sum()
            avg_value = df[primary_metric].mean()
            
            if 'revenue' in primary_metric.lower():
                summary_parts.append(f"💰 **Total Revenue:** AED {total_value:,.0f}")
                summary_parts.append(f"📈 **Average:** AED {avg_value:,.0f}")
            elif 'occ' in primary_metric.lower():
                summary_parts.append(f"🏨 **Average Occupancy:** {avg_value:.1f}%")
            elif 'adr' in primary_metric.lower():
                summary_parts.append(f"💎 **Average ADR:** AED {avg_value:,.0f}")
        
        # Add top insight
        if insights:
            summary_parts.append(f"🔍 **Key Insight:** {insights[0]}")
        
        return "\n".join(summary_parts)
    
    def add_to_conversation_history(self, question: str, sql_query: str, insights: List[str], recommendations: List[str]):
        """Add interaction to conversation history"""
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "question": question,
            "sql_query": sql_query,
            "insights": insights,
            "recommendations": recommendations
        })
        
        # Keep only last 10 interactions
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_insights(self) -> Dict[str, Any]:
        """Get insights about the conversation patterns"""
        if not self.conversation_history:
            return {}
        
        total_queries = len(self.conversation_history)
        
        # Extract common topics
        all_questions = [entry['question'].lower() for entry in self.conversation_history]
        common_topics = []
        
        topics = ['revenue', 'occupancy', 'adr', 'segment', 'forecast']
        for topic in topics:
            count = sum(1 for q in all_questions if topic in q)
            if count > 0:
                common_topics.append(f"{topic.title()} ({count})")
        
        return {
            "total_queries": total_queries,
            "common_topics": common_topics[:5],
            "session_start": self.conversation_history[0]["timestamp"] if self.conversation_history else None
        }
    
    def execute_sql_query(self, sql_query: str) -> Tuple[bool, Any, str]:
        """Execute SQL query with enhanced error handling"""
        try:
            # Validate query
            is_valid, error_msg = self.validate_sql_query(sql_query)
            if not is_valid:
                return False, None, error_msg
            
            # Execute query
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            return True, df, ""
            
        except Exception as e:
            return False, None, f"Database error: {str(e)}"
    
    def validate_sql_query(self, sql_query: str) -> Tuple[bool, str]:
        """Validate SQL query for safety"""
        upper_query = sql_query.upper().strip()
        
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 
            'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in upper_query:
                return False, f"Dangerous operation detected: {keyword}"
        
        if not upper_query.startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        return True, ""


def create_enhanced_ollama_backend(model_name: str = "phi3:mini", ollama_url: str = "http://localhost:11434", db_path: str = "db/revenue.db") -> EnhancedOllamaBackend:
    """Factory function to create Enhanced Ollama backend"""
    return EnhancedOllamaBackend(model_name, ollama_url, db_path)