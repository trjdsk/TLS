#!/usr/bin/env python3
"""Example script demonstrating TFLite palm detection integration.

This script shows how to use the integrated detector.py with TFLite models
for palm detection and validation.
"""

import logging
import argparse
import cv2
import os
from detector import PalmDetector

def main():
    parser = argparse.ArgumentParser(description="TFLite Palm Detection Example")
    parser.add_argument("--tflite-model", help="Path to .tflite model file")
    parser.add_argument("--source", default=0, help="Video source (0 for webcam, URL for ESP32-CAM)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Palm classification threshold")
    parser.add_argument("--max-hands", type=int, default=1, help="Maximum number of hands to detect")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Convert source to int if it's a number
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Initialize detector
    try:
        detector = PalmDetector(
            max_num_hands=args.max_hands,
            palm_threshold=args.threshold
        )
        
        # Load TFLite model if provided
        if args.tflite_model:
            if os.path.exists(args.tflite_model):
                success = detector.load_tflite_model(args.tflite_model)
                if success:
                    logger.info("Using TFLite model for palm validation")
                else:
                    logger.warning("Failed to load TFLite model, falling back to Edge Impulse")
            else:
                logger.error("TFLite model file not found: %s", args.tflite_model)
                return
        else:
            logger.info("No TFLite model provided, using Edge Impulse")
            
    except Exception as exc:
        logger.error("Failed to initialize detector: %s", exc)
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    
    if isinstance(source, str):
        # ESP32-CAM stream
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("Initialized ESP32-CAM stream: %s", source)
    else:
        # Webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("Initialized webcam: %d", source)
    
    if not cap.isOpened():
        logger.error("Failed to open video source: %s", source)
        return
    
    logger.info("Starting palm detection. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                continue
            
            # Run detection
            annotated, detections = detector.detect(frame)
            
            # Print results
            for i, detection in enumerate(detections):
                model_type = "TFLite" if detector._tflite_interpreter is not None else "Edge Impulse"
                print(f"Hand {i+1}: {'PALM' if detection.is_valid_palm else 'NOT_PALM'} "
                      f"(confidence: {detection.score:.3f}, model: {model_type})")
            
            # Display frame
            cv2.imshow("Integrated Palm Detection", annotated)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        logger.info("Detection stopped")


if __name__ == "__main__":
    main()


