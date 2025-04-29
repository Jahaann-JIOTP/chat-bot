#!/usr/bin/env python
"""
Safe Web Loader Module

This module provides a safe wrapper around WebBaseLoader that ensures
USER_AGENT is properly set to prevent warnings.
"""

import os

# Set USER_AGENT environment variable before imports
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Nexalyze/1.0"

# Now import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader

class SafeWebLoader(WebBaseLoader):
    """A wrapper around WebBaseLoader that ensures proper User-Agent setting."""
    
    def __init__(self, web_paths, *args, **kwargs):
        """Initialize the loader with proper headers."""
        # Ensure header_template includes User-Agent if not provided
        if "header_template" not in kwargs:
            kwargs["header_template"] = {
                "User-Agent": os.environ.get("USER_AGENT"),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        elif "User-Agent" not in kwargs["header_template"]:
            kwargs["header_template"]["User-Agent"] = os.environ.get("USER_AGENT")
            
        super().__init__(web_paths, *args, **kwargs)


# Test the loader if this file is run directly
if __name__ == "__main__":
    loader = SafeWebLoader(["https://example.com"])
    print(f"SafeWebLoader created with User-Agent: {os.environ.get('USER_AGENT')}")
    print("Loading page...")
    docs = list(loader.lazy_load())
    print(f"Successfully loaded {len(docs)} document(s) without warnings.") 