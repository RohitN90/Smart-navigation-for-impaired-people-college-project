import cv2
import os
import numpy as np

class ImageLabeler:
    def __init__(self, image_dir, label_dir, class_names):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_names = class_names
        self.current_class = 0
        self.points = []
        self.current_image = None
        self.current_image_path = None
        self.window_name = "Image Labeler"
        
        # Create label directory if it doesn't exist
        os.makedirs(label_dir, exist_ok=True)
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_image_index = 0
        
        # Create window
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            if len(self.points) == 2:
                self.save_annotation()
                self.points = []
                
    def save_annotation(self):
        if len(self.points) != 2:
            return
            
        # Get image dimensions
        height, width = self.current_image.shape[:2]
        
        # Calculate bounding box coordinates
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        
        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Calculate normalized coordinates
        x_center = (x1 + x2) / (2 * width)
        y_center = (y1 + y2) / (2 * height)
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        
        # Create label file path
        label_path = os.path.join(self.label_dir, 
                                os.path.splitext(os.path.basename(self.current_image_path))[0] + '.txt')
        
        # Append annotation to label file
        with open(label_path, 'a') as f:
            f.write(f"{self.current_class} {x_center} {y_center} {box_width} {box_height}\n")
            
        # Draw rectangle on image
        cv2.rectangle(self.current_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(self.current_image, self.class_names[self.current_class], 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    def run(self):
        while self.current_image_index < len(self.image_files):
            # Load current image
            self.current_image_path = os.path.join(self.image_dir, 
                                                 self.image_files[self.current_image_index])
            self.current_image = cv2.imread(self.current_image_path)
            
            # Display instructions
            print("\nInstructions:")
            print("1. Press 'n' for next image")
            print("2. Press 'p' for previous image")
            print("3. Press '0-9' to select class")
            print("4. Click and drag to draw bounding box")
            print("5. Press 'q' to quit")
            print(f"\nCurrent class: {self.class_names[self.current_class]}")
            print(f"Image: {self.image_files[self.current_image_index]}")
            
            while True:
                # Display image
                display_image = self.current_image.copy()
                if self.points:
                    cv2.rectangle(display_image, self.points[0], 
                                (self.points[0][0], self.points[0][1]), (0, 255, 0), 2)
                cv2.imshow(self.window_name, display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    return
                elif key == ord('n'):
                    self.current_image_index += 1
                    self.points = []
                    break
                elif key == ord('p'):
                    self.current_image_index = max(0, self.current_image_index - 1)
                    self.points = []
                    break
                elif ord('0') <= key <= ord('9'):
                    class_index = key - ord('0')
                    if class_index < len(self.class_names):
                        self.current_class = class_index
                        print(f"Selected class: {self.class_names[self.current_class]}")
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    image_dir = "dataset/images"  # Directory containing your images
    label_dir = "dataset/labels"  # Directory where labels will be saved
    class_names = [
        "person", "car", "bicycle", "motorcycle", "bus", "truck",
         "chair", "cat", "bottle", "bird","dog", "sofa", "cow", "bike", "monitor", "train", "sheep", "train", "table", "ship", "pen"
    ]  # Your class names
    
    labeler = ImageLabeler(image_dir, label_dir, class_names)
    labeler.run() 