from ultralytics import YOLO
import os

def main():
    model_path = 'runs/detect/train/weights/last.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find checkpoint at {model_path}.")
        print("Training might not have completed a full epoch before stopping.")
        return

    # Load the partially trained model
    model = YOLO(model_path)

    # Resume the training from where it left off
    # YOLO automatically uses the previous configuration (batch size, epochs, etc.)
    print("Resuming YOLO training...")
    results = model.train(resume=True)
    
if __name__ == '__main__':
    main()
