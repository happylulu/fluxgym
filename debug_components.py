#!/usr/bin/env python3
"""Debug script to check component structure"""

import re

# Read the app.py file
with open('app.py', 'r') as f:
    content = f.read()

# Find all output_components.append calls
appends = re.findall(r'output_components\.append\((.*?)\)', content)

print(f"Found {len(appends)} output_components.append calls:")
for i, append in enumerate(appends):
    print(f"{i+1}: {append}")

# Count components by type
print("\nComponent structure:")
print("1 x captioning_area")
print("150 x (captioning_row + image + caption + caption_stats)")
print("1 x start")
print(f"Total expected: {1 + 150*4 + 1} = 602")

# Check for any conditional appends
print("\nChecking for conditional appends...")
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'output_components.append' in line:
        # Check previous lines for conditions
        if i > 0 and ('if' in lines[i-1] or 'for' in lines[i-1]):
            print(f"Line {i+1}: Conditional append found: {lines[i-1].strip()}")
            print(f"         {line.strip()}")