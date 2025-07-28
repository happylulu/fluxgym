#!/usr/bin/env python3
"""Test the exact update count issue"""

MAX_IMAGES = 150

def simulate_load_captioning(num_uploaded_images):
    """Simulate the load_captioning function logic"""
    updates = []
    
    # Update for the captioning_area
    updates.append("captioning_area")
    
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        visible = i <= num_uploaded_images
        
        # Update visibility of the captioning row
        updates.append(f"row_{i}_visibility")
        
        # Update for image component
        updates.append(f"image_{i}")
        
        # Update value of captioning area
        updates.append(f"caption_{i}")
        
        # Update for caption_stats component
        updates.append(f"caption_stats_{i}")
    
    # Update for the start button
    updates.append("start_button")
    
    return updates

# Test with 2 images (minimum required)
updates = simulate_load_captioning(2)
print(f"Updates returned for 2 images: {len(updates)}")
print(f"Expected: 1 + (150 * 4) + 1 = 602")

# Calculate what 453 updates would mean
# 453 = 1 (captioning_area) + X + 1 (start)
# 451 = X
# If each image needs 4 updates, 451/4 = 112.75
# If each image needs 3 updates, 451/3 = 150.33
# So it seems like we're getting 3 updates per image instead of 4

print("\nIf we got 453 updates:")
print("453 - 1 (captioning_area) - 1 (start) = 451")
print("451 / 150 images = 3.0067 updates per image")
print("This suggests we're missing 1 update per image!")