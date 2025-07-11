"""
Streamlit UI for Academic Research Agent
Provides a user interface for interacting with the research agent.
"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from research_agent import AcademicResearchAgent, ResearchPaper

# Custom CSS for an elegant and modern UI
def load_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 1.5rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
    }
    
    .custom-header {
        background: linear-gradient(135deg, #4a90e2 0%, #9013fe 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.2);
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .custom-header h1 {
        font-size: 2.8rem;
        margin: 0;
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .custom-header p {
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-style: italic;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #4a90e2;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.1);
    }
    
    .metric-card h3 {
        color: #4a90e2;
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .sidebar .sidebar-content {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4a90e2 0%, #9013fe 100%);
        color: #ffffff;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
    }
    
    .paper-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #4a90e2;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .paper-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.1);
    }
    
    .paper-title {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .paper-authors {
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        font-style: italic;
    }
    
    .paper-meta {
        color: #95a5a6;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }
    
    .paper-abstract {
        color: #34495e;
        font-size: 1rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    .status-success {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #2e7d32;
        font-weight: 500;
    }
    
    .status-warning {
        background: #fff3e0;
        color: #e65100;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #e65100;
        font-weight: 500;
    }
    
    .status-error {
        background: #fce4ec;
        color: #c2185b;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #c2185b;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2.5rem;
        background: #ffffff;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.2rem;
        padding: 0 1.8rem;
        background: transparent;
        border-radius: 8px;
        font-weight: 600;
        color: #7f8c8d;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4a90e2 0%, #9013fe 100%);
        color: #ffffff;
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #4a90e2 0%, #9013fe 100%);
        border-radius: 10px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def render_custom_header():
    """Render custom header with gradient background"""
    st.markdown("""
    <div class="custom-header">
        <h1>🔬 Academic Research Agent</h1>
        <p>Elevate Your Literature Review with AI Precision</p>
    </div>
    """, unsafe_allow_html=True)

def render_metric_cards(result):
    """Render metric cards for research results"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📚 Total Papers</h3>
            <p>{result['total_papers']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_citations = sum(p.citation_count or 0 for p in result['papers']) / len(result['papers']) if result['papers'] else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Avg Citations</h3>
            <p>{avg_citations:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        years = [p.publication_date[:4] for p in result['papers'] if p.publication_date]
        latest_year = max(years) if years else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅 Latest Year</h3>
            <p>{latest_year}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        with_pdf = sum(1 for p in result['papers'] if p.pdf_url)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📄 PDFs Available</h3>
            <p>{with_pdf}</p>
        </div>
        """, unsafe_allow_html=True)

def render_research_interface():
    """Render the main research interface"""
    st.markdown("### 🔍 Research Query", unsafe_allow_html=True)
    
    with st.form("research_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "",
                placeholder="Enter your research topic (e.g., AI in Healthcare, Quantum Computing...)",
                label_visibility="collapsed",
                help="Type a topic to start your research journey."
            )
        
        with col2:
            submitted = st.form_submit_button("🚀 Start Research", use_container_width=True)
    
    return query, submitted

def render_sidebar_config():
    """Render sidebar configuration"""
    # st.sidebar.markdown("### ⚙️ Configuration")
    st.sidebar.markdown("**📊 Research Parameters**")
    max_papers = st.sidebar.slider("Maximum Papers", 5, 50, 10, help="Adjust the number of papers to retrieve.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚡ Quick Actions**")
    if st.sidebar.button("📊 Sample Research", use_container_width=True):
        st.session_state.sample_query = "transformer neural networks"
        st.sidebar.success("Sample query loaded!")
    if st.sidebar.button("🔄 Clear Results", use_container_width=True):
        if 'research_result' in st.session_state:
            del st.session_state.research_result
        st.sidebar.success("Results cleared!")
    return max_papers

import re
import html

def clean_text(text):
    """Clean and normalize text by removing HTML tags and entities"""
    if not text:
        return ""
    
    # First decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace multiple whitespaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra newlines and spaces
    text = text.strip()
    
    return text

def render_paper_card(paper: ResearchPaper, index: int):
    """Render individual paper card with clean text display"""
    
    # Clean all text fields
    clean_title = clean_text(paper.title)
    clean_abstract = clean_text(paper.abstract)
    clean_authors = [clean_text(author) for author in paper.authors]
    clean_venue = clean_text(paper.venue) if paper.venue else ""
    
    with st.container():
        # Create the paper card with clean styling
        st.markdown(f"""
                {index}. {clean_title}
            
            
            
                👥 {', '.join(clean_authors)}
            
            
            
                📅 {paper.publication_date} 
                {f"| 📰 {clean_venue}" if clean_venue else ""}
                {f"| 📈 {paper.citation_count} citations" if paper.citation_count else ""}
            
        
        """, unsafe_allow_html=True)
        
        # Display abstract separately with proper formatting
        if clean_abstract:
            st.markdown("**Abstract:**")
            st.markdown(f"""
            
                {clean_abstract}
            
            """, unsafe_allow_html=True)
        
        # Links section
        links_col1, links_col2 = st.columns([1, 1])
        
        with links_col1:
            if paper.pdf_url:
                st.markdown(f"""
                <a href="{paper.pdf_url}" target="_blank" style="
                    
                ">📄 PDF</a>
                """, unsafe_allow_html=True)
        
        with links_col2:
            if paper.doi:
                st.markdown(f"""
                <a href="{paper.doi}" target="_blank" style="
                    
                ">🔗 DOI</a>
                """, unsafe_allow_html=True)
        
        # Summary and key contributions
        if paper.summary:
            st.info(f"🤖 **AI Summary:** {clean_text(paper.summary)}")
        
        if paper.key_contributions:
            st.markdown("**🔑 Key Contributions:**")
            for contrib in paper.key_contributions:
                clean_contrib = clean_text(contrib)
                st.write(f"• {clean_contrib}")
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)

# Alternative simpler version using only Streamlit components
def render_paper_card_simple(paper: ResearchPaper, index: int):
    """Simple version using only Streamlit components"""
    
    # Clean all text fields
    clean_title = clean_text(paper.title)
    clean_abstract = clean_text(paper.abstract)
    clean_authors = [clean_text(author) for author in paper.authors]
    clean_venue = clean_text(paper.venue) if paper.venue else ""
    
    with st.container():
        # Use st.container with border for the card effect
        with st.container():
            st.markdown(f"### {index}. {clean_title}")
            st.markdown(f"**👥 Authors:** {', '.join(clean_authors)}")
            
            # Metadata
            meta_info = f"📅 {paper.publication_date}"
            if clean_venue:
                meta_info += f" | 📰 {clean_venue}"
            if paper.citation_count:
                meta_info += f" | 📈 {paper.citation_count} citations"
            
            st.markdown(meta_info)
            
            # Abstract
            if clean_abstract:
                st.markdown("**Abstract:**")
                st.markdown(f"> {clean_abstract}")
            
            # Links
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                if paper.pdf_url:
                    st.markdown(f"[📄 PDF]({paper.pdf_url})")
            
            with col2:
                if paper.doi:
                    st.markdown(f"[🔗 DOI]({paper.doi})")
            
            # Summary and contributions
            if paper.summary:
                st.info(f"🤖 **AI Summary:** {clean_text(paper.summary)}")
            
            if paper.key_contributions:
                st.markdown("**🔑 Key Contributions:**")
                for contrib in paper.key_contributions:
                    clean_contrib = clean_text(contrib)
                    st.write(f"• {clean_contrib}")
            
            st.divider()  # Add a divider between papers
def render_analytics_charts(result):
    """Render analytics charts for research results"""
    papers = result['papers']
    
    years = [p.publication_date[:4] for p in papers if p.publication_date]
    if years:
        year_counts = pd.Series(years).value_counts().sort_index()
        
        fig_years = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            title="📅 Publications by Year",
            labels={'x': 'Year', 'y': 'Number of Papers'},
            color=year_counts.values,
            color_continuous_scale='Blues'
        )
        fig_years.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_years, use_container_width=True)
    
    citations = [p.citation_count for p in papers if p.citation_count]
    if citations:
        fig_citations = px.histogram(
            x=citations,
            title="📊 Citation Distribution",
            labels={'x': 'Citations', 'y': 'Number of Papers'},
            nbins=10,
            color_discrete_sequence=['#4a90e2']
        )
        fig_citations.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_citations, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Academic Research Agent",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    render_custom_header()
    max_papers = render_sidebar_config()
    
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = AcademicResearchAgent()
        except Exception as e:
            st.markdown(f'<div class="status-error">❌ Failed to initialize agent: {e}</div>', unsafe_allow_html=True)
            return
    
    query, submitted = render_research_interface()
    
    if submitted and query:
        with st.spinner("🔍 Conducting research... Please wait"):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            result = asyncio.run(st.session_state.agent.conduct_research(query, max_papers))
            if "error" in result:
                st.markdown(f'<div class="status-error">❌ Research failed: {result["error"]}</div>', unsafe_allow_html=True)
            else:
                st.session_state.research_result = result
                st.markdown(f'<div class="status-success">✅ Research completed! Found {result["total_papers"]} papers.</div>', unsafe_allow_html=True)
    
    elif submitted and not query:
        st.markdown('<div class="status-warning">⚠️ Please enter a research query</div>', unsafe_allow_html=True)
    
    if 'research_result' in st.session_state:
        result = st.session_state.research_result
        
        render_metric_cards(result)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📚 Papers", "📊 Analytics", "📝 Report", "💾 Export"])
        
        with tab1:
            st.markdown("### 📚 Research Papers", unsafe_allow_html=True)
            search_query = st.text_input("🔍 Search within results:", placeholder="Enter keywords to filter papers...")
            filtered_papers = result['papers']
            if search_query:
                filtered_papers = [p for p in result['papers'] if search_query.lower() in p.title.lower() or search_query.lower() in p.abstract.lower()]
            if filtered_papers:
                for i, paper in enumerate(filtered_papers, 1):
                    render_paper_card(paper, i)
            else:
                st.info("No papers match your search criteria.")
        
        with tab2:
            st.markdown("### 📊 Research Analytics", unsafe_allow_html=True)
            render_analytics_charts(result)
        
        with tab3:
            st.markdown("### 📝 Research Report", unsafe_allow_html=True)
            st.markdown(result['markdown_report'])
        
        with tab4:
            st.markdown("### 💾 Export Options", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📄 Download Markdown Report",
                    data=result['markdown_report'],
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                df_data = []
                for paper in result['papers']:
                    df_data.append({
                        'Title': paper.title,
                        'Authors': ', '.join(paper.authors),
                        'Publication Date': paper.publication_date,
                        'Venue': paper.venue or '',
                        'Citations': paper.citation_count or 0,
                        'Abstract': paper.abstract,
                        'Summary': paper.summary or '',
                        'PDF URL': paper.pdf_url or '',
                        'DOI': paper.doi or ''
                    })
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv,
                    file_name=f"research_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #95a5a6; font-size: 0.9rem;'>"
        "🔬 Academic Research Agent | Powered by AI | Crafted with ❤️ using Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()