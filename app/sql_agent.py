"""
Simple SQL Agent using Phi-3-mini for natural language to SQL conversion
Works with Grand Millennium Revenue Analytics database
"""

import sqlite3
import os
from typing import Dict, List, Tuple, Optional
import json
from llama_cpp import Llama

class SQLAgent:
    def __init__(self, db_path: str, model_path: str):
        self.db_path = db_path
        self.model_path = model_path
        self.llm = None
        self.schema_info = {}
        self.initialize_model()
        self.load_schema()
    
    def initialize_model(self):
        """Initialize the Phi-3-mini model"""
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,  # Context window
                n_threads=4,  # Use 4 threads
                verbose=False,
                n_gpu_layers=0  # CPU only
            )
            print("SUCCESS: Phi-3-mini model loaded successfully")
        except Exception as e:
            print(f"FAILED: Failed to load model: {e}")
            raise
    
    def load_schema(self):
        """Load database schema information"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get schema for each table
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            self.schema_info[table] = {
                'columns': [(col[1], col[2]) for col in columns],
                'description': self.get_table_description(table)
            }
        
        conn.close()
        print(f"SUCCESS: Loaded schema for {len(tables)} tables")
    
    def get_table_description(self, table_name: str) -> str:
        """Get human-readable description of table purpose"""
        descriptions = {
            'occupancy_analysis': 'Daily occupancy data with rooms sold, revenue, ADR, RevPar, occupancy percentage',
            'segment_analysis': 'Hotel segment performance data including revenue, ADR, rooms for different customer segments',
            'block_analysis': 'Group block bookings data with company names, booking status, dates, and block sizes',
            'historical_occupancy_2022': '2022 historical daily occupancy data',
            'historical_occupancy_2023': '2023 historical daily occupancy data', 
            'historical_occupancy_2024': '2024 historical daily occupancy data',
            'historical_segment_2022': '2022 historical segment performance data',
            'historical_segment_2023': '2023 historical segment performance data',
            'historical_segment_2024': '2024 historical segment performance data',
            'monthly_forecast_data': 'Monthly aggregated revenue forecasting data',
            'combined_forecast_data': 'Daily forecasted occupancy and revenue data',
            'arrivals': 'Guest arrival information with booking details',
            'entered_on': 'Reservation entry tracking data',
            'str_analysis': 'STR (Smith Travel Research) competitive analysis data'
        }
        return descriptions.get(table_name, f'Data table: {table_name}')
    
    def generate_schema_prompt(self) -> str:
        """Generate schema description for the LLM prompt"""
        schema_text = "Database Schema:\n"
        
        # Focus on main tables
        main_tables = ['occupancy_analysis', 'segment_analysis', 'block_analysis', 
                      'historical_occupancy_2024', 'historical_segment_2024']
        
        for table in main_tables:
            if table in self.schema_info:
                schema_text += f"\n{table}: {self.schema_info[table]['description']}\n"
                for col_name, col_type in self.schema_info[table]['columns'][:8]:  # Limit columns
                    schema_text += f"  - {col_name} ({col_type})\n"
        
        return schema_text
    
    def create_sql_prompt(self, question: str) -> str:
        """Create a prompt for SQL generation"""
        schema = self.generate_schema_prompt()
        
        prompt = f"""<|system|>
You are a SQL expert for a hotel revenue analytics database. Convert natural language questions to SQL queries.

{schema}

Important guidelines:
- Use exact column names from schema
- For revenue/money questions, use columns ending in 'Revenue' or 'Amount'
- For occupancy, use 'Occ_Pct' or 'Occupancy_Pct' columns  
- For dates, use DATE format: 'YYYY-MM-DD'
- Always include LIMIT 100 for safety
- Return only the SQL query, no explanations

<|user|>
Question: {question}

SQL Query:<|assistant|>
"""
        return prompt
    
    def generate_sql(self, question: str) -> str:
        """Generate SQL query from natural language question"""
        prompt = self.create_sql_prompt(question)
        
        try:
            response = self.llm(
                prompt,
                max_tokens=256,
                temperature=0.1,
                stop=["<|user|>", "<|system|>", "\n\n"],
                echo=False
            )
            
            sql = response['choices'][0]['text'].strip()
            
            # Clean up the SQL
            sql = sql.replace('```sql', '').replace('```', '').strip()
            if not sql.upper().startswith('SELECT'):
                # Try to extract SELECT statement
                lines = sql.split('\n')
                for line in lines:
                    if line.strip().upper().startswith('SELECT'):
                        sql = line.strip()
                        break
            
            return sql
            
        except Exception as e:
            return f"Error generating SQL: {e}"
    
    def execute_sql(self, sql: str) -> Tuple[List[Tuple], List[str], Optional[str]]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(sql)
            results = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            conn.close()
            return results, column_names, None
            
        except Exception as e:
            return [], [], str(e)
    
    def query(self, question: str) -> Dict:
        """Main method to process natural language question"""
        # Generate SQL
        sql = self.generate_sql(question)
        
        if sql.startswith("Error"):
            return {
                'question': question,
                'sql': sql,
                'results': [],
                'columns': [],
                'error': 'Failed to generate SQL',
                'success': False
            }
        
        # Execute SQL
        results, columns, error = self.execute_sql(sql)
        
        return {
            'question': question,
            'sql': sql,
            'results': results,
            'columns': columns,
            'error': error,
            'success': error is None,
            'row_count': len(results)
        }
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions users can ask"""
        return [
            "What was the total revenue in August 2025?",
            "Show me occupancy percentage for the last 7 days",
            "Which segment had the highest ADR this month?",
            "How many group blocks are confirmed for September?",
            "What is the average RevPar for weekends in 2025?",
            "Show me the top 5 companies by block size",
            "What was the occupancy rate on 2025-08-15?",
            "Compare revenue between July and August 2025"
        ]

# Test the SQL Agent
if __name__ == "__main__":
    # Test with sample questions
    agent = SQLAgent(
        db_path="../db/revenue.db",
        model_path="../models/Phi-3-mini-4k-instruct-q4.gguf"
    )
    
    sample_questions = agent.get_sample_questions()
    
    for question in sample_questions[:2]:  # Test first 2 questions
        print(f"\nQ: {question}")
        result = agent.query(question)
        print(f"SQL: {result['sql']}")
        if result['success']:
            print(f"Results: {len(result['results'])} rows")
            if result['results']:
                print(f"Sample: {result['results'][0]}")
        else:
            print(f"Error: {result['error']}")