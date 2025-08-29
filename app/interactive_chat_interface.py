"""
Enhanced Interactive Chat Interface for Revenue Analytics
Provides conversational AI with rich visualizations and dynamic interactions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import io
from typing import Dict, Any, List


class InteractiveChatInterface:
    """Enhanced chat interface with rich interactions and visualizations"""
    
    def __init__(self, enhanced_backend):
        """Initialize the interactive chat interface"""
        self.backend = enhanced_backend
        self.session_key_messages = 'enhanced_chat_messages'
        self.session_key_insights = 'conversation_insights'
        
        # Initialize session state
        if self.session_key_messages not in st.session_state:
            st.session_state[self.session_key_messages] = []
        
        if self.session_key_insights not in st.session_state:
            st.session_state[self.session_key_insights] = {}
    
    def render_message_input(self) -> str:
        """Render enhanced message input with suggestions"""
        
        # Quick action buttons
        st.markdown("### 🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        quick_questions = [
            "📊 Show today's performance",
            "💰 Revenue trends this month", 
            "🏨 Occupancy by segment",
            "📈 Year-over-year comparison"
        ]
        
        selected_quick = None
        with col1:
            if st.button(quick_questions[0], use_container_width=True):
                selected_quick = "What is today's hotel performance including revenue, occupancy, and ADR?"
        
        with col2:
            if st.button(quick_questions[1], use_container_width=True):
                selected_quick = "Show me revenue trends for this month compared to last month"
        
        with col3:
            if st.button(quick_questions[2], use_container_width=True):
                selected_quick = "What is the occupancy percentage breakdown by business segment?"
        
        with col4:
            if st.button(quick_questions[3], use_container_width=True):
                selected_quick = "Compare this year's performance to last year's same period"
        
        # Main input area
        st.markdown("### 💬 Ask Your Question")
        
        # Show conversation insights if available
        if st.session_state[self.session_key_insights]:
            insights = st.session_state[self.session_key_insights]
            if insights.get('total_queries', 0) > 0:
                with st.expander("🧠 Conversation Insights", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Questions", insights['total_queries'])
                    with col2:
                        if insights.get('common_topics'):
                            st.write("**Common Topics:**")
                            for topic in insights['common_topics']:
                                st.write(f"• {topic}")
        
        # Text input
        question = st.text_area(
            "Type your question about hotel revenue, occupancy, forecasts, or any business metric:",
            value=selected_quick or "",
            height=100,
            placeholder="Example: What was our RevPAR performance last week?\nExample: Show me the top 5 segments by revenue this quarter\nExample: How does weekend occupancy compare to weekdays?"
        )
        
        return question
    
    def render_follow_up_suggestions(self, suggestions: List[str]):
        """Render interactive follow-up suggestions"""
        if not suggestions:
            return None
        
        st.markdown("### 💡 Suggested Follow-ups")
        
        # Create columns for suggestions
        cols = st.columns(min(len(suggestions), 3))
        
        selected_suggestion = None
        for i, suggestion in enumerate(suggestions[:6]):  # Limit to 6 suggestions
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Create clean button text
                button_text = suggestion.replace("💡", "").replace("📊", "").replace("🎯", "").replace("📈", "").strip()
                if st.button(f"{suggestion}", key=f"suggestion_{i}", use_container_width=True):
                    selected_suggestion = button_text
        
        return selected_suggestion
    
    def render_enhanced_results(self, result: Dict[str, Any]):
        """Render results with enhanced visualizations and insights"""
        
        # Display summary with rich formatting
        if result.get('summary'):
            st.markdown("### 📋 Analysis Summary")
            st.markdown(result['summary'])
        
        # Display insights in an attractive format
        if result.get('insights') or result.get('patterns') or result.get('recommendations'):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result.get('insights'):
                    st.markdown("#### 💡 Key Insights")
                    for insight in result['insights']:
                        st.info(insight)
            
            with col2:
                if result.get('patterns'):
                    st.markdown("#### 🔍 Patterns Found")
                    for pattern in result['patterns']:
                        st.success(pattern)
            
            with col3:
                if result.get('recommendations'):
                    st.markdown("#### 🎯 Recommendations")
                    for rec in result['recommendations']:
                        st.warning(rec)
        
        # Show SQL query in expandable section
        if result.get('sql_query'):
            with st.expander("🔧 Generated SQL Query", expanded=False):
                st.code(result['sql_query'], language='sql')
        
        # Display results with enhanced formatting
        if not result['results'].empty:
            df = result['results']
            
            st.markdown(f"### 📊 Results ({result['row_count']} records)")
            
            # Auto-generate visualizations based on data type
            self.auto_visualize_data(df)
            
            # Display the data table
            st.markdown("#### 📋 Detailed Data")
            
            # Format numeric columns for better display
            display_df = df.copy()
            for col in display_df.columns:
                if display_df[col].dtype in ['float64', 'int64']:
                    if 'Revenue' in col or 'ADR' in col or 'RevPAR' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"AED {x:,.0f}" if pd.notna(x) else "")
                    elif 'Occupancy' in col or 'Occ_Pct' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
                    elif 'Rooms' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=min(400, (len(display_df) + 1) * 35)
            )
            
            # Export options
            self.render_export_options(df)
        
        else:
            st.info("📭 Query executed successfully but returned no results. Try adjusting your criteria or time period.")
    
    def auto_visualize_data(self, df: pd.DataFrame):
        """Automatically generate appropriate visualizations based on data"""
        if df.empty:
            return
        
        # Revenue trend visualization
        if 'Date' in df.columns and any(col in df.columns for col in ['Revenue', 'Daily_Revenue']):
            revenue_col = 'Revenue' if 'Revenue' in df.columns else 'Daily_Revenue'
            
            if len(df) > 1:
                fig = px.line(df, x='Date', y=revenue_col, 
                             title='📈 Revenue Trend',
                             labels={revenue_col: 'Revenue (AED)', 'Date': 'Date'})
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Revenue (AED)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Occupancy visualization
        elif 'Date' in df.columns and any(col in df.columns for col in ['Occupancy_Pct', 'Occ_Pct']):
            occ_col = 'Occupancy_Pct' if 'Occupancy_Pct' in df.columns else 'Occ_Pct'
            
            if len(df) > 1:
                fig = px.line(df, x='Date', y=occ_col,
                             title='🏨 Occupancy Trend',
                             labels={occ_col: 'Occupancy (%)', 'Date': 'Date'})
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Occupancy (%)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Segment analysis visualization
        elif 'Segment' in df.columns:
            if 'Business_on_the_Books_Revenue' in df.columns:
                fig = px.bar(df, x='Segment', y='Business_on_the_Books_Revenue',
                            title='💼 Revenue by Business Segment',
                            labels={'Business_on_the_Books_Revenue': 'Revenue (AED)', 'Segment': 'Business Segment'})
                fig.update_layout(
                    xaxis_title="Business Segment",
                    yaxis_title="Revenue (AED)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif 'Business_on_the_Books_Rooms' in df.columns:
                fig = px.bar(df, x='Segment', y='Business_on_the_Books_Rooms',
                            title='🏨 Room Nights by Business Segment',
                            labels={'Business_on_the_Books_Rooms': 'Room Nights', 'Segment': 'Business Segment'})
                fig.update_layout(
                    xaxis_title="Business Segment", 
                    yaxis_title="Room Nights",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Day of week analysis
        elif 'DOW' in df.columns and len(df) > 1:
            numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            if numeric_cols:
                primary_metric = numeric_cols[0]
                
                fig = px.bar(df, x='DOW', y=primary_metric,
                            title=f'📅 {primary_metric} by Day of Week',
                            labels={primary_metric: primary_metric, 'DOW': 'Day of Week'})
                fig.update_layout(
                    xaxis_title="Day of Week",
                    yaxis_title=primary_metric,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Multi-metric dashboard for comprehensive data
        elif len(df.columns) > 3 and len(df) > 1:
            numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            if len(numeric_cols) >= 2:
                # Create subplot for multiple metrics
                metrics_to_show = numeric_cols[:4]  # Show up to 4 metrics
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=metrics_to_show,
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                for i, metric in enumerate(metrics_to_show):
                    row = (i // 2) + 1
                    col = (i % 2) + 1
                    
                    fig.add_trace(
                        go.Scatter(y=df[metric], mode='lines+markers', name=metric),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    title="📊 Multi-Metric Performance Dashboard",
                    height=600,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_export_options(self, df: pd.DataFrame):
        """Render data export options"""
        st.markdown("#### 📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"revenue_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=f"revenue_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="🔧 Download JSON",
                data=json_data,
                file_name=f"revenue_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def render_chat_message(self, message: Dict[str, Any], is_user: bool = False):
        """Render individual chat message with enhanced formatting"""
        
        with st.container():
            if is_user:
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <div style="font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem;">
                        👤 You asked:
                    </div>
                    <div style="font-size: 1.1rem;">
                        {message['content']}
                    </div>
                    <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
                        ⏰ {message['timestamp'].strftime('%H:%M:%S')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            else:  # AI response
                result = message['content']
                
                st.markdown(f"""
                <div style="background-color: #e8f4fd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #1f77b4;">
                    <div style="font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem;">
                        🤖 AI Analysis:
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if result.get("success", False):
                    self.render_enhanced_results(result)
                    
                    # Render follow-up suggestions
                    if result.get('follow_up_suggestions'):
                        follow_up = self.render_follow_up_suggestions(result['follow_up_suggestions'])
                        if follow_up:
                            # Add the selected suggestion as a new user message
                            self.add_message(follow_up, is_user=True)
                            # Process the follow-up question
                            with st.spinner("🤖 Processing your follow-up question..."):
                                ai_result = self.backend.process_enhanced_question(follow_up)
                                self.add_message(ai_result, is_user=False)
                                st.rerun()
                else:
                    st.error(f"❌ **Error:** {result.get('error_message', 'Unknown error')}")
                
                st.caption(f"⏰ {message['timestamp'].strftime('%H:%M:%S')}")
    
    def add_message(self, content: Any, is_user: bool = False):
        """Add message to conversation history"""
        message = {
            "content": content,
            "timestamp": datetime.now(),
            "is_user": is_user
        }
        st.session_state[self.session_key_messages].append(message)
    
    def clear_conversation(self):
        """Clear conversation history"""
        st.session_state[self.session_key_messages] = []
        st.session_state[self.session_key_insights] = {}
        if hasattr(self.backend, 'conversation_history'):
            self.backend.conversation_history = []
    
    def render_conversation_stats(self):
        """Render conversation statistics"""
        if st.session_state[self.session_key_messages]:
            total_messages = len(st.session_state[self.session_key_messages])
            user_messages = sum(1 for msg in st.session_state[self.session_key_messages] if msg.get('is_user', False))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Messages", total_messages)
            with col2:
                st.metric("Your Questions", user_messages)
            with col3:
                st.metric("AI Responses", total_messages - user_messages)
    
    def render_complete_interface(self):
        """Render the complete interactive chat interface"""
        
        # Header
        st.title("🤖 Enhanced AI Revenue Assistant")
        st.markdown("**Intelligent conversation with advanced analytics, insights, and visual storytelling**")
        
        # Sidebar with conversation management
        with st.sidebar:
            st.markdown("### 🎛️ Conversation Controls")
            
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                self.clear_conversation()
                st.rerun()
            
            if st.button("📊 Refresh Insights", use_container_width=True):
                insights = self.backend.get_conversation_insights()
                st.session_state[self.session_key_insights] = insights
                st.rerun()
            
            st.markdown("---")
            
            # Show conversation stats
            if st.session_state[self.session_key_messages]:
                st.markdown("### 📈 Session Stats")
                self.render_conversation_stats()
        
        # Main chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Message input
            question = self.render_message_input()
            
            # Send button
            col_send, col_clear = st.columns([1, 1])
            
            with col_send:
                send_button = st.button("🚀 Ask AI Assistant", type="primary", use_container_width=True)
            
            with col_clear:
                clear_input = st.button("🧹 Clear Input", use_container_width=True)
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("""
            **Try asking about:**
            • Revenue performance and trends
            • Occupancy patterns and forecasts  
            • Segment analysis and comparisons
            • Seasonal patterns and insights
            • Budget vs actual performance
            • Market segment optimization
            """)
        
        # Process question
        if send_button and question and question.strip():
            # Add user message
            self.add_message(question, is_user=True)
            
            # Show processing indicator
            with st.spinner("🤖 AI is analyzing your question and generating insights..."):
                try:
                    # Process with enhanced backend
                    result = self.backend.process_enhanced_question(question)
                    
                    # Add AI response
                    self.add_message(result, is_user=False)
                    
                    # Update conversation insights
                    insights = self.backend.get_conversation_insights()
                    st.session_state[self.session_key_insights] = insights
                    
                except Exception as e:
                    error_result = {
                        "success": False,
                        "error_message": f"Processing error: {str(e)}"
                    }
                    self.add_message(error_result, is_user=False)
            
            st.rerun()
        
        elif send_button:
            st.warning("⚠️ Please enter a question before clicking Ask.")
        
        # Display conversation history
        if st.session_state[self.session_key_messages]:
            st.markdown("---")
            st.markdown("### 💬 Conversation History")
            
            # Show messages in reverse order (newest first)
            for message in reversed(st.session_state[self.session_key_messages]):
                self.render_chat_message(message, message.get('is_user', False))
                st.markdown("---")
        
        else:
            # Welcome message
            st.markdown("---")
            st.info("""
            👋 **Welcome to your Enhanced AI Revenue Assistant!**
            
            I'm here to help you analyze your hotel's performance with:
            • 🧠 **Intelligent Insights** - Get smart analysis beyond just data
            • 📊 **Auto Visualizations** - See your data come to life
            • 💡 **Smart Suggestions** - Discover follow-up questions you didn't think of
            • 🎯 **Business Recommendations** - Get actionable advice
            • 🔄 **Conversation Memory** - I remember our discussion context
            
            **Ask me anything about your revenue, occupancy, segments, forecasts, or any business metric!**
            """)


# Helper function to integrate with existing app
def render_enhanced_gpt_tab(backend):
    """Render the enhanced GPT tab using the interactive interface"""
    interface = InteractiveChatInterface(backend)
    interface.render_complete_interface()