import os
import streamlit as st
import requests
import tempfile
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

# Load environment variables
load_dotenv()

# Define the Pydantic models
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

class ContentSummary(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    source: str

# Initialize session state if not already done
if 'research_results' not in st.session_state:
    st.session_state.research_results = None
if 'link_summary' not in st.session_state:
    st.session_state.link_summary = None
if 'pdf_summary' not in st.session_state:
    st.session_state.pdf_summary = None
if 'last_saved_file' not in st.session_state:
    st.session_state.last_saved_file = None

# Initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Set up the output parsers
research_parser = PydanticOutputParser(pydantic_object=ResearchResponse)
content_parser = PydanticOutputParser(pydantic_object=ContentSummary)

# Define the research prompt template
research_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=research_parser.get_format_instructions())

# Define the content summary prompt template
content_summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a content summarization assistant.
            Summarize the provided content clearly and concisely.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("human", "Please summarize the following content: {content}"),
    ]
).partial(format_instructions=content_parser.get_format_instructions())

# Define the tools
tools = [search_tool, wiki_tool, save_tool]

# Create the agent and executor
agent = create_tool_calling_agent(
    llm=llm,
    prompt=research_prompt,
    tools=tools
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Function to fetch and summarize URL content with improved header handling
def summarize_url_content(url):
    try:
        # Use browser-like headers to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            content = response.text
            # Truncate content if it's too large
            if len(content) > 50000:
                content = content[:50000] + "... (content truncated)"
            
            # Get summary from LLM
            summary_chain = content_summary_prompt | llm
            result = summary_chain.invoke({"content": content})
            parsed_result = content_parser.parse(result.content)
            return parsed_result
        else:
            return f"Error: Unable to fetch content (Status code: {response.status_code})"
    except Exception as e:
        return f"Error processing URL: {str(e)}"

# Alternative function when automated access fails
def summarize_text_input(text, url=None):
    try:
        # Truncate content if it's too large
        if len(text) > 50000:
            text = text[:50000] + "... (content truncated)"
        
        # Get summary from LLM
        summary_chain = content_summary_prompt | llm
        result = summary_chain.invoke({"content": text})
        parsed_result = content_parser.parse(result.content)
        
        # Set source if URL was provided
        if url:
            parsed_result.source = url
            
        return parsed_result
    except Exception as e:
        return f"Error processing text: {str(e)}"

# Function to extract and summarize PDF content
def summarize_pdf_content(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Truncate content if it's too large
        if len(text) > 50000:
            text = text[:50000] + "... (content truncated)"
        
        # Get summary from LLM
        summary_chain = content_summary_prompt | llm
        result = summary_chain.invoke({"content": text})
        parsed_result = content_parser.parse(result.content)
        return parsed_result
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Streamlit UI
st.title("Enhanced Research Assistant")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Research Topic", "Summarize URL", "Summarize PDF"])

# Tab 1: Research Topic
with tab1:
    st.write("Enter a topic to research, and I'll provide a summary, sources, and tools used.")
    
    # Input field for the research query
    query = st.text_input("What can I help you research?", "", key="research_query")
    
    # Button to trigger research
    if st.button("Start Research"):
        if query:
            with st.spinner("Researching..."):
                try:
                    # Invoke the agent with the query
                    raw_response = agent_executor.invoke({"query": query, "chat_history": []})
                    structured_response = research_parser.parse(raw_response["output"])
                    
                    # Save to session state
                    st.session_state.research_results = structured_response
                    
                    # Save the research results to a file
                    base_filename = f"{structured_response.topic[:20].replace(' ', '_').replace('/', '_')}"
                    txt_filepath = f"{base_filename}.txt"
                    with open(txt_filepath, "w") as f:
                        f.write(f"Research Topic: {structured_response.topic}\n\n")
                        f.write(f"Summary:\n{structured_response.summary}\n\n")
                        f.write("Sources:\n")
                        for source in structured_response.sources:
                            f.write(f"- {source}\n")
                        f.write("\nTools Used:\n")
                        for tool in structured_response.tools_used:
                            f.write(f"- {tool}\n")
                    
                    st.session_state.last_saved_file = txt_filepath
                    
                except Exception as e:
                    st.error(f"Error processing research: {str(e)}")
        else:
            st.warning("Please enter a research topic.")
    
    # Display research results if available
    if st.session_state.research_results:
        structured_response = st.session_state.research_results
        st.subheader("Research Results")
        st.write(f"**Topic**: {structured_response.topic}")
        st.write(f"**Summary**: {structured_response.summary}")
        st.write("**Sources**:")
        for source in structured_response.sources:
            st.write(f"- {source}")
        st.write("**Tools Used**:")
        for tool in structured_response.tools_used:
            st.write(f"- {tool}")
        
        # Download button for text file
        if st.session_state.last_saved_file:
            with open(st.session_state.last_saved_file, "r") as file:
                file_content = file.read()
            st.download_button(
                label="Download Research Results",
                data=file_content,
                file_name=os.path.basename(st.session_state.last_saved_file),
                mime="text/plain"
            )

# Tab 2: Summarize URL
with tab2:
    st.write("Enter a URL to fetch and summarize its content.")
    url = st.text_input("URL to summarize:", key="url_input")
    
    # Add manual text input option for when URL fetching fails
    st.write("If automatic URL fetching fails, you can paste the content here:")
    manual_text = st.text_area("Content from URL (optional):", height=200, key="manual_text_input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Summarize URL"):
            if url:
                with st.spinner("Fetching and summarizing content..."):
                    summary = summarize_url_content(url)
                    st.session_state.link_summary = summary
            else:
                st.warning("Please enter a URL.")
    
    with col2:
        if st.button("Summarize Pasted Content"):
            if manual_text:
                with st.spinner("Summarizing pasted content..."):
                    summary = summarize_text_input(manual_text, url if url else "Manual input")
                    st.session_state.link_summary = summary
            else:
                st.warning("Please paste some content to summarize.")
    
    # Display URL summary if available
    if st.session_state.link_summary:
        summary = st.session_state.link_summary
        if isinstance(summary, str):  # Error message
            st.error(summary)
        else:
            st.subheader("Content Summary")
            st.write(f"**Title**: {summary.title}")
            st.write(f"**Summary**: {summary.summary}")
            st.write("**Key Points**:")
            for point in summary.key_points:
                st.write(f"- {point}")
            st.write(f"**Source**: {summary.source}")
            
            # Generate text for download
            summary_text = f"Title: {summary.title}\n\n"
            summary_text += f"Summary:\n{summary.summary}\n\n"
            summary_text += "Key Points:\n"
            for point in summary.key_points:
                summary_text += f"- {point}\n"
            summary_text += f"\nSource: {summary.source}"
            
            # Download button
            st.download_button(
                label="Download Content Summary",
                data=summary_text,
                file_name=f"content_summary_{summary.title[:20].replace(' ', '_').replace('/', '_')}.txt",
                mime="text/plain"
            )

# Tab 3: Summarize PDF
with tab3:
    st.write("Upload a PDF file to summarize its content.")
    pdf_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_upload")
    
    if pdf_file is not None and st.button("Summarize PDF"):
        with st.spinner("Processing PDF and generating summary..."):
            summary = summarize_pdf_content(pdf_file)
            st.session_state.pdf_summary = summary
    
    # Display PDF summary if available
    if st.session_state.pdf_summary:
        summary = st.session_state.pdf_summary
        if isinstance(summary, str):  # Error message
            st.error(summary)
        else:
            st.subheader("PDF Content Summary")
            st.write(f"**Title**: {summary.title}")
            st.write(f"**Summary**: {summary.summary}")
            st.write("**Key Points**:")
            for point in summary.key_points:
                st.write(f"- {point}")
            st.write(f"**Source**: {summary.source}")
            
            # Generate text for download
            summary_text = f"Title: {summary.title}\n\n"
            summary_text += f"Summary:\n{summary.summary}\n\n"
            summary_text += "Key Points:\n"
            for point in summary.key_points:
                summary_text += f"- {point}\n"
            summary_text += f"\nSource: {summary.source}"
            
            # Download button
            st.download_button(
                label="Download PDF Summary",
                data=summary_text,
                file_name=f"pdf_summary_{summary.title[:20].replace(' ', '_').replace('/', '_')}.txt",
                mime="text/plain"
            )

# Add a button to clear all results
if st.button("Clear All Results"):
    st.session_state.research_results = None
    st.session_state.link_summary = None
    st.session_state.pdf_summary = None
    st.session_state.last_saved_file = None
    st.experimental_rerun()