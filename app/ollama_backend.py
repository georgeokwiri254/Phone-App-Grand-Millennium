"""
Ollama AI Backend Module for Revenue Analytics App
Connects to SQLite database and generates SQL queries using local Ollama API
"""

import sqlite3
import pandas as pd
import requests
import json
import os
import re
from typing import Tuple, Optional, Dict, Any
import traceback


class OllamaSQLBackend:
    """Backend class for Ollama AI SQL query generation"""
    
    def __init__(self, model_name: str = "phi3:mini", ollama_url: str = "http://localhost:11434", db_path: str = "db/revenue.db"):
        """
        Initialize the Ollama backend
        
        Args:
            model_name (str): Ollama model name (default: phi3:mini)
            ollama_url (str): Ollama server URL
            db_path (str): Path to SQLite database
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.db_path = db_path
        self.connected = False
        self.error_message = ""
        
        # Test Ollama connection
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                # Check if the specified model is available
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
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            schema_info = "Database Schema:\n\n"
            
            for table in tables:
                table_name = table[0]
                schema_info += f"Table: {table_name}\n"
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
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
    
    def generate_sql_query(self, question: str) -> str:
        """
        Generate SQL query using Ollama AI
        
        Args:
            question (str): Natural language question
            
        Returns:
            str: Generated SQL query
        """
        try:
            schema = self.get_database_schema()
            
            prompt = f"""You are a SQL expert for a hotel revenue analytics database. Generate a valid SQLite query to answer the user's question.

{schema}

User Question: {question}

Guidelines:
1. Only generate SELECT statements - no INSERT, UPDATE, DELETE, DROP
2. Use proper SQLite syntax
3. Return only the SQL query without explanation
4. Use appropriate WHERE clauses for date filtering
5. Format dates as 'YYYY-MM-DD' for comparisons
6. Use LIMIT clause if the result might be very large
7. Use appropriate aggregations (SUM, AVG, COUNT) when needed

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
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                sql_query = result.get('response', '').strip()
                
                # Clean up the response to extract just the SQL
                sql_query = self.clean_sql_response(sql_query)
                return sql_query
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Error generating SQL: {str(e)}")
    
    def clean_sql_response(self, response: str) -> str:
        """
        Clean the SQL response from Ollama to extract pure SQL
        
        Args:
            response (str): Raw response from Ollama
            
        Returns:
            str: Cleaned SQL query
        """
        # Remove common prefixes/suffixes
        response = response.strip()
        
        # Remove markdown code blocks
        response = re.sub(r'```sql\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # Remove explanatory text before the query
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
    
    def validate_sql_query(self, sql_query: str) -> Tuple[bool, str]:
        """
        Validate SQL query for safety
        
        Args:
            sql_query (str): SQL query to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Convert to uppercase for checking
        upper_query = sql_query.upper().strip()
        
        # Check for dangerous operations
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 
            'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in upper_query:
                return False, f"Dangerous operation detected: {keyword}"
        
        # Must start with SELECT
        if not upper_query.startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        return True, ""
    
    def execute_sql_query(self, sql_query: str) -> Tuple[bool, Any, str]:
        """
        Execute SQL query against the database
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            Tuple[bool, Any, str]: (success, results_df, error_message)
        """
        try:
            # Validate query first
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
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question and return results
        
        Args:
            question (str): Natural language question
            
        Returns:
            Dict[str, Any]: Result dictionary with success status, data, and metadata
        """
        try:
            # Generate SQL query
            sql_query = self.generate_sql_query(question)
            
            # Execute query
            success, results_df, error_msg = self.execute_sql_query(sql_query)
            
            if success:
                return {
                    "success": True,
                    "sql_query": sql_query,
                    "results": results_df,
                    "row_count": len(results_df),
                    "question": question
                }
            else:
                return {
                    "success": False,
                    "sql_query": sql_query,
                    "error_message": error_msg,
                    "question": question
                }
                
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Processing error: {str(e)}",
                "question": question
            }


def create_ollama_backend(model_name: str = "phi3:mini", ollama_url: str = "http://localhost:11434", db_path: str = "db/revenue.db") -> OllamaSQLBackend:
    """
    Factory function to create Ollama backend
    
    Args:
        model_name (str): Ollama model name
        ollama_url (str): Ollama server URL  
        db_path (str): Database path
        
    Returns:
        OllamaSQLBackend: Configured backend instance
    """
    return OllamaSQLBackend(model_name, ollama_url, db_path)


# Test connection function
def test_ollama_connection(model_name: str = "phi3:mini", ollama_url: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Test Ollama connection and model availability
    
    Args:
        model_name (str): Model name to test
        ollama_url (str): Ollama server URL
        
    Returns:
        Dict[str, Any]: Connection test results
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            available = any(model_name in name for name in model_names)
            
            return {
                "connected": True,
                "model_available": available,
                "available_models": model_names,
                "message": "Connected successfully" if available else f"Model {model_name} not found"
            }
        else:
            return {
                "connected": False,
                "model_available": False,
                "message": f"Server returned status {response.status_code}"
            }
    except Exception as e:
        return {
            "connected": False,
            "model_available": False,
            "message": f"Connection failed: {str(e)}"
        }