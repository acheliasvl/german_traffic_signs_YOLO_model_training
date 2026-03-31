from ultralytics import YOLO

def main():
    # Load a pre-trained model
    model = YOLO('yolov8n.pt')

    # Train the model
    # Note: imgsz=640 is default, we specify epochs=50.
    results = model.train(data='gtsdb.yaml', epochs=50, imgsz=640)
    
    print("\nTraining complete!")
    
if __name__ == '__main__':
    main()
