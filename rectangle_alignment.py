# %%
import cv2
import numpy as np

def rotate_bound(image, angle):
    # Grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Grab the rotation matrix, then grab the sine and cosine
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

# Load the image
image = cv2.imread('rectangles.jpg')

# Convert to grayscale and threshold
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Prepare a blank canvas for the new image
new_image = np.full_like(image, 255)

for cnt in contours:
    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:  # Ensure the contour has 4 sides (rectangle)
        # Get the bounding box of the rectangle
        x, y, w, h = cv2.boundingRect(approx)

        # Extract the rectangle region
        rect_region = image[y:y+h, x:x+w]

        # Create a mask for the rectangle
        mask = np.zeros(rect_region.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1, offset=(-x, -y))
        
        # Determine the orientation of the rectangle and adjust rotation angle
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])

        # Check if width is greater than height
        if width > height:
            # If so, adjust angle to rotate rectangle to have longest side downwards
            angle -= 90


        # Rotate the rectangle region
        rotated_rect = rotate_bound(rect_region, angle)

        # Rotate the mask
        rotated_mask = rotate_bound(mask, angle)
        
        # Resize rotated mask to match ROI dimensions if necessary
        rH, rW = rotated_rect.shape[:2]
        roi_new_image = new_image[y:y+rH, x:x+rW]

        # Ensure mask is binary and of type uint8
        _, resized_mask = cv2.threshold(rotated_mask, 0, 255, cv2.THRESH_BINARY)
        resized_mask = cv2.resize(resized_mask, (roi_new_image.shape[1], roi_new_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Resize rotated_rect to match ROI dimensions if necessary
        rotated_rect = cv2.resize(rotated_rect, (roi_new_image.shape[1], roi_new_image.shape[0]))

        # Ensure rotated_rect is of type uint8
        rotated_rect = rotated_rect.astype('uint8')

        # Place the rotated rectangle into the new image using the resized mask
        mask_inv = cv2.bitwise_not(resized_mask)
        bg = cv2.bitwise_and(roi_new_image, roi_new_image, mask=mask_inv)
        fg = cv2.bitwise_and(rotated_rect, rotated_rect, mask=resized_mask)
        new_image[y:y+rH, x:x+rW] = cv2.add(bg, fg)

# Show the result
cv2.imshow('Aligned Rectangles', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%



