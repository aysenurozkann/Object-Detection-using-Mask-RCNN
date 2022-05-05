import cv2
import visualize_image

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)   # Create window
image = cv2.imread("video_Moment.jpg")
image = cv2.resize(image, dsize = (400, 400))


results = visualize_image.model.detect([image], verbose=0)

# Visualize results
r = results[0]

masked_image = visualize_image.display_instances(image,"Image", r['rois'], r['masks'], r['class_ids'], 
                        visualize_image.class_names, r['scores'])


cv2.imwrite("image.jpg", masked_image)    
cv2.imshow("Image", masked_image)
cv2.waitKey(0)
    