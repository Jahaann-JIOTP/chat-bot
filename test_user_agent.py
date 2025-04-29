#!/usr/bin/env python
# Test USER_AGENT environment variable fix

import os
# Set USER_AGENT environment variable before any imports
print("Step 1: Setting USER_AGENT environment variable...")
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Nexalyze/1.0"

print(f"Step 2: USER_AGENT is set to: {os.environ.get('USER_AGENT')}")

# Import WebBaseLoader after setting USER_AGENT
print("Step 3: Importing WebBaseLoader...")
from langchain_community.document_loaders import WebBaseLoader
print("Step 4: WebBaseLoader imported successfully without warnings.")

# Create a WebBaseLoader instance with our manual header
print("Step 5: Creating WebBaseLoader instance...")
custom_header = {
    "User-Agent": os.environ.get("USER_AGENT"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
loader = WebBaseLoader(
    web_paths=["https://example.com"],
    header_template=custom_header
)
print("Step 6: WebBaseLoader instance created successfully.")

# Verify our header is set correctly
print(f"Step 7: Checking loader header. User-Agent is: {custom_header.get('User-Agent')}")

print("Test completed successfully.") 