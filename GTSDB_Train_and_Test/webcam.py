import cv2
from ultralytics import YOLO
import os

def main():
    # Load the trained model
    # Note: If your training just started, this file might not exist yet!
    # Wait for at least the first epoch to complete before running this.
    model_path = 'runs/detect/train/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at '{model_path}'.")
        print("Please ensure your training has finished or has completed at least the first epoch!")
        return
        
    print("Loading model...")
    model = YOLO(model_path)

    # Note: '0' is usually the default ID for the built-in webcam.
    # If it fails to open, try changing '0' to '1'.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return
        
    print("Webcam successfully opened! Press 'q' in the video window to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            # Run YOLOv8 inference on the current webcam frame
            # Lowered confidence to 0.25 since the model is still early in training!
            results = model(frame, conf=0.25, verbose=False)
            
            # Plot the bounding boxes and labels on the frame
            annotated_frame = results[0].plot()
            
            # Display the frame to the screen
            cv2.imshow("GTSDB Traffic Sign Recognition", annotated_frame)
            
            # Watch for the 'q' key to be pressed to gracefully exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break
        else:
            print("Warning: Dropped a hardware frame.")
            break
            
    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
