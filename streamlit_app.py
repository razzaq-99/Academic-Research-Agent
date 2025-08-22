"""
Streamlit UI for Academic Research Agent
Provides a user interface for interacting with the research agent.
"""
import re
import html
import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
import plotly.express as px
from research_agent import AcademicResearchAgent, ResearchPaper


# ----------------- Custom CSS ---------------- 
def load_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 1.5rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
    }
    
    .custom-header {
        background: linear-gradient(135deg, #EA6666FF 0%, #A00000FF 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
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
        border-left: 5px solid #667eea;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
    }
    
    .metric-card h3 {
        color: #667eea;
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
    
    .paper-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 1.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8ecf0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .paper-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #EA6666FF 0%, #A00000FF 100%);
    }
    
    .paper-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    
    .paper-number {
        position: absolute;
        top: 15px;
        right: 20px;
        background: linear-gradient(135deg, #EA6666FF 0%, #A00000FF 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .paper-title {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        margin-top: 0;
        line-height: 1.4;
        padding-right: 80px;
    }
    
    .paper-authors {
        color: #667eea;
        font-size: 1rem;
        margin-bottom: 0.8rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .paper-meta {
        color: #7f8c8d;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 15px;
        flex-wrap: wrap;
    }
    
    .paper-meta span {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .paper-abstract {
        color: #000000FF;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
        text-align: justify;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .paper-buttons {
        display: flex;
        gap: 12px;
        margin-top: 1.5rem;
    }
    
    .paper-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        border-radius: 25px;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .paper-btn-pdf {
        background: linear-gradient(135deg, #EA6666FF 0%, #A00000FF 100%);
        color: white;
    }
    
    .paper-btn-pdf:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        color: white;
        text-decoration: none;
    }
    
    .paper-btn-doi {
        background: #ffffff;
        color: #667eea;
        border: 2px solid #667eea;
    }
    
    .paper-btn-doi:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-decoration: none;
    }
    
    .paper-btn-disabled {
        background: #e9ecef;
        color: #6c757d;
        cursor: not-allowed;
        opacity: 0.6;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #EA6666FF 0%, #A00000FF 100%);
        color: #ffffff;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
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
        background: linear-gradient(135deg, #EA6666FF 0%, #A00000FF 100%);
        color: #ffffff;
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #EA6666FF 0%, #A00000FF 100%);
        border-radius: 10px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ----------------- HEADER ----------------
def render_custom_header():
    """Render custom header with gradient background"""
    st.markdown("""
    <div class="custom-header">
        <h1> Smart Research Hub</h1>
        <p>Your AI-Powered Research Companion</p>
    </div>
    """, unsafe_allow_html=True)


# ----------------- Metric Cards i.e total papers retrieved , Avg Citations and so on ----------------
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


# ----------------- Main Interface ----------------
def render_research_interface():
    """Render the main research interface"""
    st.markdown("Research Topic", unsafe_allow_html=True)
    
    with st.form("research_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "",
                placeholder="Enter your research topic here",
                label_visibility="collapsed",
                help="Type a topic to start your research journey."
            )
        
        with col2:
            submitted = st.form_submit_button("🚀 Start Research", use_container_width=True)
    
    return query, submitted


# ----------------- Sidebar ----------------
def render_sidebar_config():
    """Render sidebar configuration"""
    st.sidebar.markdown("Research Parameters")
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


def clean_text(text):
    """Clean and normalize text by removing HTML tags and entities"""
    if not text:
        return ""
    
    
    text = html.unescape(text)
    
    
    text = re.sub(r'<[^>]+>', '', text)
    
    
    text = re.sub(r'\s+', ' ', text)
    
    
    text = text.strip()
    
    return text


# ----------------- Paper description cards ----------------
def render_paper_card(paper: ResearchPaper, index: int):
    """Render individual paper card using Streamlit components"""
    
    clean_title = clean_text(paper.title)
    clean_abstract = clean_text(paper.abstract)
    clean_authors = [clean_text(author) for author in paper.authors]
    clean_venue = clean_text(paper.venue) if paper.venue else ""
    
    pub_date = paper.publication_date if paper.publication_date else "Date not available"
    
    with st.container():
        
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(f"### {index}. {clean_title}")
        
        if clean_authors:
            st.markdown(f"**👥 Authors:** {', '.join(clean_authors)}")
        else:
            st.markdown("**👥 Authors:** Not available")
        
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.markdown(f"**📅 Date:** {pub_date}")
        with meta_col2:
            if clean_venue:
                st.markdown(f"**📰 Venue:** {clean_venue}")
        with meta_col3:
            if paper.citation_count:
                st.markdown(f"**📈 Citations:** {paper.citation_count}")
        
        if clean_abstract:
            st.markdown("**📝 Abstract:**")
            st.markdown(f"> {clean_abstract}")
        
        
        button_col1, button_col2, button_col3 = st.columns([1, 1, 8])
        
        with button_col1:
            if paper.pdf_url and paper.pdf_url.strip():
                st.markdown(f'<a href="{paper.pdf_url}" target="_blank" style="display: inline-block; padding: 0.5rem 1rem; background: #667eea; color: white; text-decoration: none; border-radius: 5px; font-weight: 500;">📄 PDF</a>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="display: inline-block; padding: 0.5rem 1rem; background: #e9ecef; color: #6c757d; border-radius: 5px; font-weight: 500;">📄 No PDF</span>', unsafe_allow_html=True)
        
        with button_col2:
            if paper.doi and paper.doi.strip():
                doi_url = paper.doi if paper.doi.startswith('http') else f"https://doi.org/{paper.doi}"
                st.markdown(f'<a href="{doi_url}" target="_blank" style="display: inline-block; padding: 0.5rem 1rem; background: #667eea; color: white; text-decoration: none; border-radius: 5px; font-weight: 500;">🔗 DOI</a>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="display: inline-block; padding: 0.5rem 1rem; background: #e9ecef; color: #6c757d; border-radius: 5px; font-weight: 500;">🔗 No DOI</span>', unsafe_allow_html=True)
        
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if paper.summary:
            clean_summary = clean_text(paper.summary)
            st.info(f"🤖 **AI Summary:** {clean_summary}")
        
        if paper.key_contributions:
            st.markdown("**🔑 Key Contributions:**")
            for contrib in paper.key_contributions:
                clean_contrib = clean_text(contrib)
                st.write(f"• {clean_contrib}")
        
        st.markdown("---")


# ----------------- Graphs and Charts ----------------
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
            color_continuous_scale='magma'
        )
        fig_years.update_layout(
            showlegend=False, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=16,
            title_x=0.5
        )
        st.plotly_chart(fig_years, use_container_width=True)
    
    
    citations = [p.citation_count for p in papers if p.citation_count]
    if citations:
        fig_citations = px.histogram(
            x=citations,
            title="📊 Citation Distribution",
            labels={'x': 'Citations', 'y': 'Number of Papers'},
            nbins=10,
            color_discrete_sequence=['rgba(160, 0, 0, 1.0)']
        )
        fig_citations.update_layout(
            showlegend=False, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=16,
            title_x=0.5
        )
        st.plotly_chart(fig_citations, use_container_width=True)


# ----------------- Main method ----------------
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
    
    
    if 'sample_query' in st.session_state:
        query = st.session_state.sample_query
        submitted = True
        del st.session_state.sample_query
    
    
    if submitted and query:
        with st.spinner("🔍 Conducting research... Please wait"):
            try:
                result = asyncio.run(st.session_state.agent.conduct_research(query, max_papers))
                if "error" in result:
                    st.markdown(f'<div class="status-error">❌ Research failed: {result["error"]}</div>', unsafe_allow_html=True)
                else:
                    st.session_state.research_result = result
                    st.markdown(f'<div class="status-success">✅ Research completed! Found {result["total_papers"]} papers.</div><br>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="status-error">❌ Research failed: {str(e)}</div>', unsafe_allow_html=True)
    
    elif submitted and not query:
        st.markdown('<div class="status-warning">⚠️ Please enter a research query</div>', unsafe_allow_html=True)
    
    
    if 'research_result' in st.session_state:
        result = st.session_state.research_result
        
        
        render_metric_cards(result)
        tab1, tab2, tab3, tab4 = st.tabs(["📚 Papers", "📊 Analytics", "📝 Report", "💾 Export"])
        with tab1:
            st.markdown("### 📚 Research Papers")
            
            
            search_query = st.text_input(
                "🔍 Search within results:", 
                placeholder="Enter keywords to filter papers...",
                help="Search through titles and abstracts"
            )
            
            # ----------------- Filtering papers ----------------
            filtered_papers = result['papers']
            if search_query:
                search_lower = search_query.lower()
                filtered_papers = [
                    p for p in result['papers'] 
                    if search_lower in p.title.lower() or 
                       search_lower in (p.abstract or '').lower() or
                       any(search_lower in author.lower() for author in p.authors)
                ]
            
            # ----------------- Displaying papers ----------------
            if filtered_papers:
                st.markdown(f"**Showing {len(filtered_papers)} of {len(result['papers'])} papers**")
                st.markdown("---")
                
                for i, paper in enumerate(filtered_papers, 1):
                    render_paper_card(paper, i)
            else:
                st.info("No papers match your search criteria.")
        
        with tab2:
            st.markdown("### 📊 Research Analytics")
            render_analytics_charts(result)
        
        with tab3:
            st.markdown("### 📝 Research Report")
            if 'markdown_report' in result:
                st.markdown(result['markdown_report'])
            else:
                st.info("No report available for this research.")
        
        with tab4:
            st.markdown("### 💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'markdown_report' in result:
                    st.download_button(
                        label="📄 Download Markdown Report",
                        data=result['markdown_report'],
                        file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                else:
                    st.info("No report available for download.")
            
            with col2:
                df_data = []
                for paper in result['papers']:
                    df_data.append({
                        'Title': clean_text(paper.title),
                        'Authors': ', '.join([clean_text(author) for author in paper.authors]),
                        'Publication Date': paper.publication_date or '',
                        'Venue': clean_text(paper.venue) if paper.venue else '',
                        'Citations': paper.citation_count or 0,
                        'Abstract': clean_text(paper.abstract) if paper.abstract else '',
                        'Summary': clean_text(paper.summary) if paper.summary else '',
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
    
    # ----------------- Footer ----------------
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem; padding: 1rem;'>"
        "Academic Research Agent | Powered by AI | Made by Abdul Razzaq"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()