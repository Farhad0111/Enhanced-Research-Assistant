from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import json

def save_to_file(info):
    """Save information to both JSON and TXT files."""
    try:
        # Extract topic name for filename (first 20 chars)
        if isinstance(info, dict) and "topic" in info:
            base_filename = f"{info['topic'][:20].replace(' ', '_')}"
        else:
            base_filename = "research_results"
        
        json_filename = f"{base_filename}.json"
        txt_filename = f"{base_filename}.txt"
        
        # Save as JSON
        with open(json_filename, "w") as f:
            if isinstance(info, str):
                try:
                    # Try to parse as JSON if it's a string
                    info_dict = json.loads(info)
                    json.dump(info_dict, f, indent=2)
                except:
                    # If not valid JSON, save as is in JSON format
                    json.dump({"content": info}, f, indent=2)
            else:
                json.dump(info, f, indent=2)
        
        # Save as TXT
        with open(txt_filename, "w") as f:
            if isinstance(info, dict):
                # Format dictionary nicely for text file
                f.write(f"Research Topic: {info.get('topic', 'Unknown')}\n\n")
                f.write(f"Summary:\n{info.get('summary', 'No summary provided')}\n\n")
                f.write("Sources:\n")
                for source in info.get('sources', []):
                    f.write(f"- {source}\n")
                f.write("\nTools Used:\n")
                for tool in info.get('tools_used', []):
                    f.write(f"- {tool}\n")
            else:
                # Just write the string content
                f.write(str(info))
        
        return f"Information saved to {json_filename} and {txt_filename}"
    except Exception as e:
        return f"Error saving information: {str(e)}"

save_tool = Tool(
    name="SaveInformation",
    description="Save research information to both JSON and TXT files. Provide a JSON object with at least a 'topic' field.",
    func=save_to_file
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
