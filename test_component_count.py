#!/usr/bin/env python3
"""Test script to verify component count logic"""

MAX_IMAGES = 150

# Simulate output_components list structure
output_components = []

# 1. captioning_area
output_components.append("captioning_area")

# 2. For each image: row, image, caption, caption_stats
for i in range(1, MAX_IMAGES + 1):
    output_components.append(f"captioning_row_{i}")
    output_components.append(f"image_{i}")
    output_components.append(f"caption_{i}")
    output_components.append(f"caption_stats_{i}")

# 3. start button
output_components.append("start")

print(f"Total output_components: {len(output_components)}")
print(f"Expected: 1 (captioning_area) + {MAX_IMAGES}*4 (images) + 1 (start) = {1 + MAX_IMAGES*4 + 1}")

# Simulate load_captioning updates
updates = []

# Update for captioning_area
updates.append("update_captioning_area")

# Updates for each image
for i in range(1, MAX_IMAGES + 1):
    updates.append(f"update_row_{i}_visibility")
    updates.append(f"update_image_{i}")
    updates.append(f"update_caption_{i}")
    updates.append(f"update_caption_stats_{i}")

# Update for start button
updates.append("update_start_button")

print(f"\nTotal updates returned: {len(updates)}")
print(f"Match: {len(output_components) == len(updates)}")

if len(output_components) != len(updates):
    print(f"MISMATCH: Expected {len(output_components)}, got {len(updates)}")
    print(f"Difference: {len(output_components) - len(updates)}")