# %%
import cv2
import numpy as np

# Function to calculate the length of a line
def line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Load the image
image = cv2.imread('rectangles.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

counter = 0
lines_within_rectangle = []
# Loop over the contours to find rectangles and lines within them
for cnt in contours:
    counter = counter+1
    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # If the polygon has 4 vertices, it could be a rectangle
    if len(approx) == 4:
        rect = cv2.boundingRect(approx)
        x, y, w, h = rect

        # Create a slightly smaller mask for the current rectangle to exclude edges
        mask = np.zeros(gray.shape, np.uint8)
        
        # Offset mask from the edge of the rectangle so that only the lines completely within the rectangle is identified
        offset = 28
        #manual offset didn't work for the biggest rectangle so we need to calculate offset according to each rectangle
        # Calculate dynamic offset based on rectangle size
        if counter == 1:
            offset = int(max(w, h) * 0.199)
        
        cv2.rectangle(mask, (x+offset, y+offset), (x+w-offset, y+h-offset), 255, -1)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Visualize the mask
        # cv2.imshow('Mask', mask)
        # cv2.waitKey(0)

        # Detect lines within the offset rectangle
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=2)

        # Check if any lines were found
        if lines is not None:
            # Sort the lines by length
            sorted_lines = sorted(lines, key=lambda l: line_length(l[0]), reverse=True)

            #Here, I have adjusted the offset of the mask so that the longest line is the line within the rectangle.
            longest_line = sorted_lines[0][0]
            lines_within_rectangle.append(longest_line)
            # Draw the longest line in red
            cv2.line(image, (longest_line[0], longest_line[1]), (longest_line[2], longest_line[3]), (0, 0, 255), 2)
     
sorted_lines_within_rectangle = sorted(lines_within_rectangle, key=line_length)

# Output the ranked lines and draw the rank on the image
for idx, line in enumerate(sorted_lines_within_rectangle):
    # Calculate the position for text (somewhere near the middle of the line)
    x1, y1, x2, y2 = line
    text_position = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
    
    # The rank is idx + 1 since enumerate starts at 0
    rank = str(idx + 1)
    
    # Draw the rank on the image
    cv2.putText(image, rank, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Show the result
cv2.imshow('Image with Ranked Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
