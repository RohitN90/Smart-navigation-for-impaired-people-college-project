import cv2
import numpy as np
import time
import pyttsx3
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

class ObjectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Navigation - Object Detection")
        self.root.geometry("1280x800")
        self.root.configure(bg='#1E1E1E')  # Dark modern background
        
        # Initialize text-to-speech engine with better voice settings
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        # Try to find a better voice
        for voice in voices:
            if 'en' in voice.id.lower():  # Prefer English voices
                self.engine.setProperty('voice', voice.id)
                break
        else:
            self.engine.setProperty('voice', voices[0].id)  # Use first available voice if no English voice found
            
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Print available voices for debugging
        print("Available voices:")
        for voice in voices:
            print(f"Voice ID: {voice.id}")
            print(f"Voice Name: {voice.name}")
            print(f"Voice Languages: {voice.languages}")
            print("---")
        
        # Load YOLOv3
        self.net = cv2.dnn.readNet('./yolov3.weights', './yolov3.cfg')
        with open('./coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layers
        layer_names = self.net.getLayerNames()
        try:
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Constants
        self.REAL_WIDTH = 0.5
        self.FOCAL_LENGTH = 700
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize variables
        self.detected_objects = set()
        self.start_time = time.time()
        self.clear_time = self.start_time
        self.speech_time = self.start_time
        self.is_speaking = False
        
        # Start video capture
        self.is_running = True
        self.update_frame()
        
    def create_widgets(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme as base
        
        # Configure colors and styles
        style.configure('TFrame', background='#1E1E1E')
        style.configure('TLabel', background='#1E1E1E', foreground='#FFFFFF')
        style.configure('TButton', 
                       padding=10, 
                       background='#007ACC',
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'))
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 24, 'bold'), 
                       foreground='#FFFFFF',
                       background='#1E1E1E')
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 12),
                       foreground='#CCCCCC',
                       background='#1E1E1E')
        style.configure('TLabelframe', 
                       background='#2D2D2D',
                       foreground='#FFFFFF')
        style.configure('TLabelframe.Label', 
                       background='#2D2D2D',
                       foreground='#FFFFFF',
                       font=('Segoe UI', 10, 'bold'))
        
        # Title and subtitle
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, 
                              text="Smart Navigation System",
                              style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame,
                                 text="Real-time Object Detection for Visually Impaired",
                                 style='Subtitle.TLabel')
        subtitle_label.pack(pady=(5, 0))
        
        # Create video frame with border
        video_frame = ttk.LabelFrame(main_frame, 
                                   text="Live Camera Feed",
                                   padding="10")
        video_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_frame = ttk.Label(video_frame)
        self.video_frame.pack(padx=2, pady=2)
        
        # Create detection results frame
        results_frame = ttk.LabelFrame(main_frame, 
                                     text="Detected Objects",
                                     padding="10")
        results_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        # Create text widget for displaying detections with custom styling
        self.detection_text = tk.Text(results_frame,
                                    width=40,
                                    height=30,
                                    bg='#2D2D2D',
                                    fg='#FFFFFF',
                                    font=('Consolas', 10),
                                    relief='flat',
                                    padx=10,
                                    pady=10)
        self.detection_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Create control buttons with better styling
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        self.start_button = ttk.Button(control_frame,
                                     text="Start Detection",
                                     command=self.start_detection,
                                     style='TButton')
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = ttk.Button(control_frame,
                                    text="Stop Detection",
                                    command=self.stop_detection,
                                    style='TButton')
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Add status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame,
                                    text="System Ready",
                                    style='Subtitle.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Configure button hover effects
        def on_enter(e):
            e.widget['style'] = 'Hover.TButton'
            
        def on_leave(e):
            e.widget['style'] = 'TButton'
            
        style.configure('Hover.TButton',
                       background='#0098FF')
        
        for button in [self.start_button, self.stop_button]:
            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)
        
    def speak_detection(self, objects):
        """Speak the detected objects in a separate thread"""
        def speak():
            self.is_speaking = True
            try:
                print(f"Debug - Objects set: {objects}")  # Debug print to see the actual objects
                
                if len(objects) == 1:
                    obj = next(iter(objects))
                    print(f"Debug - Single object details: {obj}")  # Debug print object details
                    if len(obj) >= 4:  # Make sure we have all required elements
                        message = f"I found a {obj[0]}, of distance {obj[2]:.1f} meters at position {obj[3]}"
                    else:
                        print(f"Debug - Invalid object format: {obj}")
                        message = "I found an object"
                else:
                    objects_list = []
                    for obj in objects:
                        print(f"Debug - Object details: {obj}")  # Debug print each object
                        if len(obj) >= 4:
                            objects_list.append(f"a {obj[0]}, of distance {obj[2]:.1f} meters at position {obj[3]}")
                        else:
                            print(f"Debug - Invalid object format: {obj}")
                    message = f"I found {', '.join(objects_list)}" if objects_list else "I found some objects"
                
                print(f"Debug - Full message to speak: {message}")  # Debug print
                
                # Clear any existing speech
                self.engine.stop()
                
                # Speak the complete message at once
                self.engine.say(message)
                self.engine.runAndWait()
                
            except Exception as e:
                print(f"Error in speech: {str(e)}")  # Debug print
                import traceback
                print(f"Full error traceback: {traceback.format_exc()}")  # Print full error traceback
            finally:
                self.is_speaking = False
        
        # Start speaking in a new thread
        threading.Thread(target=speak, daemon=True).start()
        
    def update_frame(self):
        if not self.is_running:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Process frame for object detection
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            distances = []
            positions = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        distance = (self.REAL_WIDTH * self.FOCAL_LENGTH) / w
                        distances.append(distance)
                        
                        if center_x < width / 3:
                            position = 'left'
                        elif center_x > 2 * width / 3:
                            position = 'right'
                        else:
                            position = 'center'
                        positions.append(position)
            
            # Apply non-maximum suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Draw detections
            new_detected_objects = set()
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence = confidences[i]
                    distance = distances[i]
                    position = positions[i]
                    print(f"Debug - Adding object: {label}, {confidence}, {distance}, {position}")  # Debug print
                    new_detected_objects.add((label, confidence, distance, position))
                    
                    # Draw rectangle with better visibility
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Add background to text for better visibility
                    text = f'{label} {confidence:.2f} {distance:.2f}m {position}'
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Update detection text and speak
            current_time = time.time()
            if current_time - self.start_time >= 3:
                if new_detected_objects - self.detected_objects:
                    self.detection_text.delete(1.0, tk.END)
                    for obj in new_detected_objects:
                        detection_text = f"{obj[0]}: {obj[1]:.2f}, Distance: {obj[2]:.2f}m, Position: {obj[3]}\n"
                        self.detection_text.insert(tk.END, detection_text)
                    
                    # Only speak about new objects
                    new_objects = new_detected_objects - self.detected_objects
                    print(f"Debug - New objects to speak about: {new_objects}")  # Debug print
                    if new_objects and not self.is_speaking:
                        self.speak_detection(new_objects)
                    
                    self.detected_objects.update(new_detected_objects)
                self.start_time = current_time
            
            # Convert frame to PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (960, 540))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_frame.configure(image=photo)
            self.video_frame.image = photo
            
        # Schedule next frame update
        self.root.after(10, self.update_frame)
    
    def start_detection(self):
        self.is_running = True
        self.update_frame()
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
    
    def stop_detection(self):
        self.is_running = False
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
    
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop() 