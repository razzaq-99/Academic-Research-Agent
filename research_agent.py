"""
Academic Research Agent - Complete Implementation
A fully functional research agent for automated literature reviews using LangChain.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from urllib.parse import quote
import xml.etree.ElementTree as ET

# Core libraries
import requests
import streamlit as st
from fpdf import FPDF
import pandas as pd

# LangChain imports
from langchain.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.callbacks import StreamlitCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchPaper:
    """Data class for storing research paper information"""
    title: str
    authors: List[str]
    abstract: str
    publication_date: str
    pdf_url: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    summary: Optional[str] = None
    key_contributions: Optional[List[str]] = None

class ArxivAPI:
    """Wrapper for ArXiv API to search and fetch papers"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        self.session = requests.Session()
    
    def search_papers(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """Search for papers on ArXiv"""
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            return self._parse_arxiv_response(response.text)
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[ResearchPaper]:
        """Parse ArXiv XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            for entry in root.findall('atom:entry', ns):
                # Extract basic information
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                
                # Extract authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns).text
                    authors.append(name)
                
                # Extract publication date
                published = entry.find('atom:published', ns).text
                pub_date = published.split('T')[0]  # Extract date part
                
                # Extract ArXiv ID and PDF URL
                arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                pdf_url = None
                
                for link in entry.findall('atom:link', ns):
                    if link.get('type') == 'application/pdf':
                        pdf_url = link.get('href')
                        break
                
                # Extract DOI if available
                doi = None
                doi_element = entry.find('arxiv:doi', ns)
                if doi_element is not None:
                    doi = doi_element.text
                
                paper = ResearchPaper(
                    title=title,
                    authors=authors,
                    abstract=summary,
                    publication_date=pub_date,
                    arxiv_id=arxiv_id,
                    pdf_url=pdf_url,
                    doi=doi
                )
                
                papers.append(paper)
                
        except Exception as e:
            logger.error(f"Error parsing ArXiv response: {e}")
            
        return papers

class SemanticScholarAPI:
    """Wrapper for Semantic Scholar API"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic Research Agent (your-email@example.com)'
        })
    
    def search_papers(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """Search for papers using Semantic Scholar"""
        try:
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,authors,abstract,year,venue,citationCount,openAccessPdf,externalIds'
            }
            
            response = self.session.get(
                f"{self.BASE_URL}/paper/search",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_semantic_scholar_response(data)
            
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def _parse_semantic_scholar_response(self, data: Dict) -> List[ResearchPaper]:
        """Parse Semantic Scholar API response"""
        papers = []
        
        try:
            for item in data.get('data', []):
                # Extract basic information
                title = item.get('title', '')
                abstract = item.get('abstract', '')
                year = item.get('year')
                venue = item.get('venue', '')
                citation_count = item.get('citationCount', 0)
                
                # Extract authors
                authors = []
                for author in item.get('authors', []):
                    authors.append(author.get('name', ''))
                
                # Extract PDF URL
                pdf_url = None
                open_access_pdf = item.get('openAccessPdf')
                if open_access_pdf:
                    pdf_url = open_access_pdf.get('url')
                
                # Extract DOI
                doi = None
                external_ids = item.get('externalIds', {})
                if external_ids:
                    doi = external_ids.get('DOI')
                
                paper = ResearchPaper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    publication_date=str(year) if year else '',
                    pdf_url=pdf_url,
                    doi=doi,
                    venue=venue,
                    citation_count=citation_count
                )
                
                papers.append(paper)
                
        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar response: {e}")
            
        return papers

class LLMService:
    """Service for handling LLM operations with fallback support"""
    
    def __init__(self, use_openai: bool = True, openai_api_key: Optional[str] = None):
        self.use_openai = use_openai
        self.llm = None
        self.embeddings = None
        
        if use_openai and openai_api_key:
            try:
                self.llm = ChatOllama(
                    model_name="gemma:2b",
                    temperature=0.1,
                    
                )
                self.embeddings = OllamaEmbeddings(model="gemma:2b")
                logger.info("Ollama LLM initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {e}")
                self._fallback_to_ollama()
        else:
            self._fallback_to_ollama()
    
    def _fallback_to_ollama(self):
        """Fallback to Ollama with LLaMA3"""
        try:
            self.llm = Ollama(model="gemma:2b")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Ollama LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise Exception("Both OpenAI and Ollama initialization failed")
    
    def summarize_paper(self, paper: ResearchPaper) -> Dict[str, Any]:
        """Generate summary and key contributions for a paper"""
        try:
            # Create prompt for summarization
            prompt_template = PromptTemplate(
                input_variables=["title", "abstract", "authors"],
                template="""
                Analyze the following research paper and provide:
                1. A concise summary (2-3 sentences)
                2. Key contributions (3-5 bullet points)
                3. Research methodology used
                4. Potential impact and applications
                
                Title: {title}
                Authors: {authors}
                Abstract: {abstract}
                
                Please format your response as:
                SUMMARY: [Your summary here]
                
                KEY CONTRIBUTIONS:
                - [Contribution 1]
                - [Contribution 2]
                - [Contribution 3]
                
                METHODOLOGY: [Brief description]
                
                IMPACT: [Potential applications and impact]
                """
            )
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            
            # Generate response
            response = chain.run(
                title=paper.title,
                abstract=paper.abstract,
                authors=", ".join(paper.authors)
            )
            
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"Error summarizing paper: {e}")
            return {
                "summary": "Summary generation failed",
                "key_contributions": [],
                "methodology": "N/A",
                "impact": "N/A"
            }
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            sections = response.split('\n\n')
            result = {
                "summary": "",
                "key_contributions": [],
                "methodology": "",
                "impact": ""
            }
            
            for section in sections:
                if section.startswith('SUMMARY:'):
                    result["summary"] = section.replace('SUMMARY:', '').strip()
                elif section.startswith('KEY CONTRIBUTIONS:'):
                    contributions = section.replace('KEY CONTRIBUTIONS:', '').strip()
                    result["key_contributions"] = [
                        line.strip('- ').strip() 
                        for line in contributions.split('\n') 
                        if line.strip().startswith('-')
                    ]
                elif section.startswith('METHODOLOGY:'):
                    result["methodology"] = section.replace('METHODOLOGY:', '').strip()
                elif section.startswith('IMPACT:'):
                    result["impact"] = section.replace('IMPACT:', '').strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "summary": response[:200] + "...",
                "key_contributions": [],
                "methodology": "N/A",
                "impact": "N/A"
            }

class VectorStore:
    """Vector store for semantic search and retrieval"""
    
    def __init__(self, embeddings, use_chroma: bool = True):
        self.embeddings = embeddings
        self.use_chroma = use_chroma
        self.vectorstore = None
        
    def create_from_papers(self, papers: List[ResearchPaper]) -> None:
        """Create vector store from research papers"""
        try:
            documents = []
            
            for i, paper in enumerate(papers):
                # Create document with metadata
                content = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
                metadata = {
                    "title": paper.title,
                    "authors": ", ".join(paper.authors),
                    "publication_date": paper.publication_date,
                    "paper_index": i
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            # Create vector store
            if self.use_chroma:
                self.vectorstore = Chroma.from_documents(
                    documents, 
                    self.embeddings,
                    persist_directory="./chroma_db"
                )
            else:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                
            logger.info(f"Vector store created with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search"""
        try:
            if self.vectorstore:
                return self.vectorstore.similarity_search(query, k=k)
            return []
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

class ReportGenerator:
    """Generate structured reports in multiple formats"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def generate_markdown_report(self, papers: List[ResearchPaper], query: str) -> str:
        """Generate a comprehensive markdown report"""
        report = f"""# Academic Research Report: {query}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of {len(papers)} research papers related to "{query}". The papers were automatically retrieved and analyzed using advanced natural language processing techniques.

## Paper Analysis

"""
        
        for i, paper in enumerate(papers, 1):
            report += f"""### {i}. {paper.title}

**Authors:** {', '.join(paper.authors)}  
**Publication Date:** {paper.publication_date}  
**Venue:** {paper.venue or 'N/A'}  
**Citations:** {paper.citation_count or 'N/A'}  

**Abstract:**
{paper.abstract}

"""
            
            if paper.summary:
                report += f"""**AI-Generated Summary:**
{paper.summary}

"""
            
            if paper.key_contributions:
                report += f"""**Key Contributions:**
"""
                for contrib in paper.key_contributions:
                    report += f"- {contrib}\n"
                report += "\n"
            
            if paper.pdf_url:
                report += f"**PDF Link:** [{paper.pdf_url}]({paper.pdf_url})\n\n"
            
            report += "---\n\n"
        
        report += f"""## Research Trends and Insights

Based on the analysis of {len(papers)} papers, several key trends emerge in the field of {query}:

1. **Methodological Approaches**: The papers demonstrate diverse methodological approaches
2. **Key Themes**: Common themes include innovation, efficiency, and practical applications
3. **Future Directions**: Emerging areas show promise for continued research

## Recommendations

1. **Further Reading**: Focus on highly cited papers for foundational knowledge
2. **Research Gaps**: Identify areas with limited coverage for potential research opportunities
3. **Collaboration**: Consider interdisciplinary approaches based on paper diversity

---

*Report generated by Academic Research Agent*
"""
        
        return report
    
    def generate_pdf_report(self, papers: List[ResearchPaper], query: str, output_path: str) -> bool:
        """Generate PDF report"""
        try:
            class PDF(FPDF):
                def header(self):
                    self.set_font('Arial', 'B', 15)
                    self.cell(0, 10, f'Academic Research Report: {query}', 0, 1, 'C')
                    self.ln(10)
                
                def footer(self):
                    self.set_y(-15)
                    self.set_font('Arial', 'I', 8)
                    self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
            pdf = PDF()
            pdf.add_page()
            pdf.set_font('Arial', size=12)
            
            # Add content
            pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
            pdf.ln(10)
            
            for i, paper in enumerate(papers, 1):
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, f'{i}. {paper.title[:60]}...', 0, 1)
                
                pdf.set_font('Arial', size=10)
                pdf.cell(0, 8, f'Authors: {", ".join(paper.authors[:3])}', 0, 1)
                pdf.cell(0, 8, f'Date: {paper.publication_date}', 0, 1)
                
                pdf.set_font('Arial', size=9)
                # Add abstract (truncated for PDF)
                abstract_lines = paper.abstract[:500].split('\n')
                for line in abstract_lines:
                    pdf.cell(0, 6, line.encode('latin-1', 'ignore').decode('latin-1'), 0, 1)
                
                pdf.ln(5)
            
            pdf.output(output_path)
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return False

class AcademicResearchAgent:
    """Main research agent class orchestrating all components"""
    
    def __init__(self, use_openai: bool = True, openai_api_key: Optional[str] = None):
        self.arxiv_api = ArxivAPI()
        self.semantic_scholar_api = SemanticScholarAPI()
        self.llm_service = LLMService(use_openai, openai_api_key)
        self.vector_store = VectorStore(self.llm_service.embeddings)
        self.report_generator = ReportGenerator()
        
    async def conduct_research(self, query: str, max_papers: int = 10) -> Dict[str, Any]:
        """Conduct comprehensive research on a given topic"""
        try:
            logger.info(f"Starting research for query: {query}")
            
            # Step 1: Search for papers
            st.info("🔍 Searching for relevant papers...")
            papers = await self._search_papers(query, max_papers)
            
            if not papers:
                return {"error": "No papers found for the given query"}
            
            # Step 2: Analyze papers with LLM
            st.info("🤖 Analyzing papers with AI...")
            papers = await self._analyze_papers(papers)
            
            # Step 3: Create vector store for semantic search
            st.info("📊 Creating semantic search index...")
            self.vector_store.create_from_papers(papers)
            
            # Step 4: Generate reports
            st.info("📝 Generating research report...")
            markdown_report = self.report_generator.generate_markdown_report(papers, query)
            
            return {
                "papers": papers,
                "markdown_report": markdown_report,
                "total_papers": len(papers)
            }
            
        except Exception as e:
            logger.error(f"Error in research process: {e}")
            return {"error": str(e)}
    
    async def _search_papers(self, query: str, max_papers: int) -> List[ResearchPaper]:
        """Search for papers using multiple APIs"""
        all_papers = []
        
        # Search ArXiv
        try:
            arxiv_papers = self.arxiv_api.search_papers(query, max_papers // 2)
            all_papers.extend(arxiv_papers)
            logger.info(f"Found {len(arxiv_papers)} papers from ArXiv")
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
        
        # Search Semantic Scholar
        try:
            semantic_papers = self.semantic_scholar_api.search_papers(query, max_papers // 2)
            all_papers.extend(semantic_papers)
            logger.info(f"Found {len(semantic_papers)} papers from Semantic Scholar")
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
        
        # Remove duplicates based on title similarity
        unique_papers = self._remove_duplicates(all_papers)
        
        return unique_papers[:max_papers]
    
    async def _analyze_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Analyze papers using LLM"""
        analyzed_papers = []
        
        for paper in papers:
            try:
                analysis = self.llm_service.summarize_paper(paper)
                
                # Update paper with analysis
                paper.summary = analysis.get("summary", "")
                paper.key_contributions = analysis.get("key_contributions", [])
                
                analyzed_papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error analyzing paper {paper.title}: {e}")
                analyzed_papers.append(paper)  # Add without analysis
        
        return analyzed_papers
    
    def _remove_duplicates(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Simple duplicate detection based on title
            title_lower = paper.title.lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
        
        return unique_papers
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform semantic search on analyzed papers"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [
                {
                    "title": doc.metadata.get("title", ""),
                    "authors": doc.metadata.get("authors", ""),
                    "publication_date": doc.metadata.get("publication_date", ""),
                    "content": doc.page_content,
                    "relevance_score": "High"  # Placeholder
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

# Streamlit UI
import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .custom-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        color: #667eea;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
        color: #333;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Paper card styling */
    .paper-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .paper-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .paper-title {
        color: #333;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .paper-authors {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .paper-meta {
        color: #888;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        background: transparent;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Hide Streamlit branding */
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
        <p>AI-Powered Literature Review with Intelligent Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def render_metric_cards(result):
    """Render metric cards for research results"""
    if 'research_result' in st.session_state:
        result = st.session_state.research_result
        
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
    st.markdown("### 🔍 Research Query")
    
    # Create a form for better UX
    with st.form("research_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "",
                placeholder="Enter your research topic (e.g., machine learning in healthcare, quantum computing applications...)",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.form_submit_button("🚀 Start Research", use_container_width=True)
    
    return query, submitted

def render_sidebar_config():
    """Render sidebar configuration"""
    st.sidebar.markdown("### ⚙️ Configuration")
    
    # LLM Configuration
    st.sidebar.markdown("**🤖 AI Model**")
    use_openai = st.sidebar.checkbox("Use OpenAI GPT", value=False)
    openai_api_key = None
    
    if use_openai:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.sidebar.markdown('<div class="status-warning">⚠️ Please provide OpenAI API key</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-success">✅ Using Ollama (LLaMA3)</div>', unsafe_allow_html=True)
        st.sidebar.caption("Make sure Ollama is running locally")
    
    st.sidebar.markdown("---")
    
    # Research Parameters
    st.sidebar.markdown("**📊 Research Parameters**")
    max_papers = st.sidebar.slider("Maximum Papers", 5, 50, 10)
    
    # Quick Actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚡ Quick Actions**")
    
    if st.sidebar.button("📊 Sample Research", use_container_width=True):
        st.session_state.sample_query = "transformer neural networks"
        st.sidebar.success("Sample query loaded!")
    
    if st.sidebar.button("🔄 Clear Results", use_container_width=True):
        if 'research_result' in st.session_state:
            del st.session_state.research_result
        st.sidebar.success("Results cleared!")
    
    return use_openai, openai_api_key, max_papers

def render_paper_card(paper, index):
    """Render individual paper card"""
    with st.container():
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">{index}. {paper.title}</div>
            <div class="paper-authors">👥 {', '.join(paper.authors)}</div>
            <div class="paper-meta">
                📅 {paper.publication_date} 
                {f"| 📰 {paper.venue}" if paper.venue else ""}
                {f"| 📈 {paper.citation_count} citations" if paper.citation_count else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Expandable content
        with st.expander("View Details", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**Abstract:**")
                st.write(paper.abstract)
                
                if paper.summary:
                    st.markdown("**🤖 AI Summary:**")
                    st.info(paper.summary)
                
                if paper.key_contributions:
                    st.markdown("**🔑 Key Contributions:**")
                    for contrib in paper.key_contributions:
                        st.write(f"• {contrib}")
            
            with col2:
                st.markdown("**📎 Links:**")
                if paper.pdf_url:
                    st.markdown(f"[📄 PDF]({paper.pdf_url})")
                if paper.doi:
                    st.markdown(f"[🔗 DOI]({paper.doi})")

def render_analytics_charts(result):
    """Render analytics charts for research results"""
    papers = result['papers']
    
    # Publication year distribution
    years = [p.publication_date[:4] for p in papers if p.publication_date]
    if years:
        year_counts = pd.Series(years).value_counts().sort_index()
        
        fig_years = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            title="📅 Publications by Year",
            labels={'x': 'Year', 'y': 'Number of Papers'},
            color=year_counts.values,
            color_continuous_scale='viridis'
        )
        fig_years.update_layout(showlegend=False)
        st.plotly_chart(fig_years, use_container_width=True)
    
    # Citation distribution
    citations = [p.citation_count for p in papers if p.citation_count]
    if citations:
        fig_citations = px.histogram(
            x=citations,
            title="📊 Citation Distribution",
            labels={'x': 'Citations', 'y': 'Number of Papers'},
            nbins=10
        )
        st.plotly_chart(fig_citations, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Academic Research Agent",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Render custom header
    render_custom_header()
    
    # Sidebar configuration
    use_openai, openai_api_key, max_papers = render_sidebar_config()
    
    # Initialize agent
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = AcademicResearchAgent(use_openai, openai_api_key)
        except Exception as e:
            st.markdown(f'<div class="status-error">❌ Failed to initialize agent: {e}</div>', unsafe_allow_html=True)
            return
    
    # Main research interface
    query, submitted = render_research_interface()
    
    # Handle form submission
    if submitted and query:
        with st.spinner("🔍 Conducting research... Please wait"):
            progress_bar = st.progress(0)
            
            # Simulate progress updates
            for i in range(100):
                progress_bar.progress(i + 1)
            
            # Run research
            result = asyncio.run(
                st.session_state.agent.conduct_research(query, max_papers)
            )
            
            if "error" in result:
                st.markdown(f'<div class="status-error">❌ Research failed: {result["error"]}</div>', unsafe_allow_html=True)
            else:
                st.session_state.research_result = result
                st.markdown(f'<div class="status-success">✅ Research completed! Found {result["total_papers"]} papers.</div>', unsafe_allow_html=True)
    
    elif submitted and not query:
        st.markdown('<div class="status-warning">⚠️ Please enter a research query</div>', unsafe_allow_html=True)
    
    # Display results if available
    if 'research_result' in st.session_state:
        result = st.session_state.research_result
        
        # Render metrics
        render_metric_cards(result)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📚 Papers", "📊 Analytics", "📝 Report", "💾 Export"])
        
        with tab1:
            st.markdown("### 📚 Research Papers")
            
            # Search within results
            search_query = st.text_input("🔍 Search within results:", placeholder="Enter keywords to filter papers...")
            
            filtered_papers = result['papers']
            if search_query:
                filtered_papers = [p for p in result['papers'] 
                                 if search_query.lower() in p.title.lower() or 
                                    search_query.lower() in p.abstract.lower()]
            
            if filtered_papers:
                for i, paper in enumerate(filtered_papers, 1):
                    render_paper_card(paper, i)
            else:
                st.info("No papers match your search criteria.")
        
        with tab2:
            st.markdown("### 📊 Research Analytics")
            render_analytics_charts(result)
        
        with tab3:
            st.markdown("### 📝 Research Report")
            st.markdown(result['markdown_report'])
        
        with tab4:
            st.markdown("### 💾 Export Options")
            
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
                # Create DataFrame for CSV export
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
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "🔬 Academic Research Agent | Powered by AI | Made with ❤️ using Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()