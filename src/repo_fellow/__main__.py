import asyncio
from typing import Optional
from .utils.config import configure_environment
from .graph.builder import graph

async def main(repo_url: Optional[str] = None, query: str = "analyze"):
    """Main entry point for the repository analysis"""
    try:
        # Configure environment
        configure_environment()
        
        # Initialize state
        initial_state = {
            "repo_url": repo_url,
            "query": query,
            "messages": [],
            "context": [],
            "files": [],
            "repo_context": {},
            "query_type": "",
            "response": "",
            "analysis_complete": False,
        }
        
        # Execute graph
        result = await graph.ainvoke(initial_state)
        return result
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import sys
    
    repo_url = sys.argv[1] if len(sys.argv) > 1 else None
    query = sys.argv[2] if len(sys.argv) > 2 else "analyze"
    
    result = asyncio.run(main(repo_url, query))
    print(result.get("response", result.get("error", "No response generated"))) 