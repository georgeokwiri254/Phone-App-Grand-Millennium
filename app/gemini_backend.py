"""
Gemini AI Backend Module for Revenue Analytics App
Connects to SQLite database and generates SQL queries using Gemini API
"""

import sqlite3
import pandas as pd
import google.generativeai as genai
import os
import re
from typing import Tuple, Optional, Dict, Any
import traceback


class GeminiSQLBackend:
    """Backend class for Gemini AI SQL query generation"""
    
    def __init__(self, api_key: str, db_path: str = "db/revenue.db"):
        """
        Initialize the Gemini backend
        
        Args:
            api_key (str): Gemini API key
            db_path (str): Path to SQLite database
        """
        self.api_key = api_key
        self.db_path = db_path
        self.model = None
        
        # Configure Gemini
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
            self.connected = True
        except Exception as e:
            self.connected = False
            self.error_message = f"Failed to initialize Gemini: {str(e)}"
    
    def get_database_schema(self) -> str:
        """
        Get the database schema for context
        
        Returns:
            str: Database schema as string
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = "Database Schema:\n\n"
            
            for table in tables:
                table_name = table[0]
                schema_info += f"Table: {table_name}\n"
                
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    schema_info += f"  - {col_name} ({col_type})\n"
                
                schema_info += "\n"
            
            conn.close()
            return schema_info
            
        except Exception as e:
            return f"Error getting schema: {str(e)}"
    
    def is_safe_query(self, sql_query: str) -> bool:
        """
        Check if SQL query is safe (only SELECT statements)
        
        Args:
            sql_query (str): SQL query to check
            
        Returns:
            bool: True if safe, False otherwise
        """
        # Remove comments and normalize whitespace
        query_clean = re.sub(r'--.*?$', '', sql_query, flags=re.MULTILINE)
        query_clean = re.sub(r'/\*.*?\*/', '', query_clean, flags=re.DOTALL)
        query_clean = query_clean.strip().upper()
        
        # Must start with SELECT
        if not query_clean.startswith('SELECT'):
            return False
        
        # Check for dangerous keywords (excluding UNION which is safe in SELECT context)
        dangerous_keywords = [
            'DELETE', 'UPDATE', 'INSERT', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'REPLACE', 'EXEC', 'EXECUTE'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_clean:
                return False
        
        # Additional check: if UNION is present, ensure it's only in SELECT context
        if 'UNION' in query_clean:
            union_parts = query_clean.split('UNION')
            for i, part in enumerate(union_parts):
                part = part.strip()
                if part.startswith('ALL'):
                    part = part[3:].strip()
                if i == 0:
                    if not part.startswith('SELECT'):
                        return False
                else:
                    if not part.startswith('SELECT'):
                        return False
        
        return True
    
    def query_database(self, sql_query: str) -> Tuple[bool, pd.DataFrame, str]:
        """
        Execute SQL query on the database
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            Tuple[bool, pd.DataFrame, str]: (success, dataframe, error_message)
        """
        try:
            # Safety check
            if not self.is_safe_query(sql_query):
                return False, pd.DataFrame(), "Query not allowed. Only SELECT statements are permitted."
            
            # Execute query
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            return True, df, ""
            
        except Exception as e:
            error_msg = f"Database query error: {str(e)}"
            return False, pd.DataFrame(), error_msg
    
    def generate_sql_query(self, question: str) -> Tuple[bool, str, str]:
        """
        Generate SQL query using Gemini AI
        
        Args:
            question (str): Natural language question
            
        Returns:
            Tuple[bool, str, str]: (success, sql_query, error_message)
        """
        try:
            if not self.connected:
                return False, "", "Gemini AI not connected"
            
            # Get database schema
            schema = self.get_database_schema()
            
            # Create prompt
            prompt = f"""
You are a SQL expert for a hotel revenue analytics database. Given the following database schema and a natural language question, 
generate a SQL SELECT query that answers the question.

{schema}

IMPORTANT CONTEXT:
- This is hotel revenue data with occupancy, room sales, ADR (Average Daily Rate), RevPAR (Revenue Per Available Room)
- Main tables for daily data: occupancy_analysis, hotel_data_combined, historical_occupancy_YYYY
- Segment data: segment_analysis, historical_segment_YYYY 
- Reservations: entered_on (reservations entered), arrivals (guest arrivals)
- Block business: block_analysis
- Dates are in various formats: 'YYYY-MM-DD', timestamps, or text

⚠️ CRITICAL COLUMN RULES:
- segment_analysis does NOT have "Revenue" column - use "Business_on_the_Books_Revenue"
- historical_segment_YYYY tables use "Business_on_the_Books_Revenue" NOT "Revenue"
- entered_on table: Use "COMPANY_CLEAN" NOT "Company_Name", use "AMOUNT" NOT "Revenue"
- ❌ NEVER use: Company_Name, Revenue (these columns DO NOT EXIST in entered_on table)
- ✅ ALWAYS use: COMPANY_CLEAN for company, AMOUNT for revenue in entered_on table
- arrivals table: Use "COMPANY_NAME_CLEAN" NOT "Company", use "AMOUNT" for revenue
- Some columns have spaces: "Rm Sold" (must be quoted in SQL)
- Room columns: Rm_Sold, Rooms_Sold, "Rm Sold", Business_on_the_Books_Rooms
- ADR columns: ADR, Business_on_the_Books_ADR
- Replace YYYY with actual years: historical_segment_2022, historical_segment_2023, historical_segment_2024
- For segments use: SELECT MergedSegment, SUM(Business_on_the_Books_Revenue) FROM historical_segment_2024
- For top companies from entered_on: SELECT COMPANY_CLEAN AS Company_Name, SUM(AMOUNT) AS Total_Revenue FROM entered_on GROUP BY COMPANY_CLEAN ORDER BY SUM(AMOUNT) DESC LIMIT 5

Rules:
1. Only generate SELECT queries
2. Do not use DELETE, UPDATE, INSERT, DROP, or other modifying statements
3. Use proper SQL syntax for SQLite
4. Return only the SQL query, no explanations
5. Use appropriate table and column names from the schema above
6. For date queries, use appropriate date functions like strftime() for SQLite
7. Use aggregations (SUM, COUNT, AVG) when appropriate
8. Use proper WHERE clauses for filtering
9. When asking about "last month" or recent periods, use recent dates from the data
10. For revenue totals, use SUM() on revenue columns
11. For occupancy, use occupancy_analysis or hotel_data_combined tables
12. For segment analysis, use segment_analysis or historical_segment_YYYY tables
13. 💰 CURRENCY: All revenue amounts are in AED (United Arab Emirates Dirham)
14. When mentioning currency, always use "AED" not "$" or "USD"

Question: {question}

CRITICAL: Return ONLY a single SQL SELECT statement with NO comments or explanations.

SQL Query:
"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()
            
            # Clean up the response
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Handle multiple statements - take first SELECT
            if '--' in sql_query:
                sql_query = sql_query.split('--')[0].strip()
            
            if ';' in sql_query:
                statements = sql_query.split(';')
                for stmt in statements:
                    stmt = stmt.strip()
                    if stmt.upper().startswith('SELECT'):
                        sql_query = stmt
                        break
            
            # Final safety check
            if not self.is_safe_query(sql_query):
                return False, "", "Generated query is not safe"
            
            return True, sql_query, ""
            
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            return False, "", error_msg
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question and return SQL + results
        
        Args:
            question (str): Natural language question
            
        Returns:
            Dict containing success status, SQL query, results, and error messages
        """
        result = {
            'success': False,
            'sql_query': '',
            'results': pd.DataFrame(),
            'error_message': '',
            'row_count': 0
        }
        
        try:
            # Validate input
            if not question or question.strip() == "":
                result['error_message'] = "Please enter a question"
                return result
            
            # Generate SQL query
            sql_success, sql_query, sql_error = self.generate_sql_query(question)
            
            if not sql_success:
                result['error_message'] = f"SQL generation failed: {sql_error}"
                return result
            
            result['sql_query'] = sql_query
            
            # Execute query
            query_success, df, query_error = self.query_database(sql_query)
            
            if not query_success:
                result['error_message'] = f"Query execution failed: {query_error}"
                return result
            
            result['success'] = True
            result['results'] = df
            result['row_count'] = len(df)
            
            return result
            
        except Exception as e:
            result['error_message'] = f"Unexpected error: {str(e)}"
            return result


def create_gemini_backend(api_key: str, db_path: str = "db/revenue.db") -> GeminiSQLBackend:
    """
    Factory function to create Gemini backend instance
    
    Args:
        api_key (str): Gemini API key
        db_path (str): Path to SQLite database
        
    Returns:
        GeminiSQLBackend: Configured backend instance
    """
    return GeminiSQLBackend(api_key, db_path)


# Usage example:
# backend = create_gemini_backend("your-api-key")
# result = backend.process_question("What is the total revenue for last month?")
# print(result['sql_query'])
# print(result['results'])