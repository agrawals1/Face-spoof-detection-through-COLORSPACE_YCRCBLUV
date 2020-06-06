import sys
import argparse
import cv2
from time import time
from libfaceid.detector    import FaceDetectorModels, FaceDetector
from libfaceid.encoder     import FaceEncoderModels, FaceEncoder
from libfaceid.liveness    import FaceLivenessModels, FaceLiveness




# Set the window name
WINDOW_NAME = "Facial_Recognition"

# Set the input directories
INPUT_DIR_DATASET               = "datasets"
INPUT_DIR_MODEL_DETECTION       = "models/detection/"
INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
INPUT_DIR_MODEL_TRAINING        = "models/training/"
INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"
INPUT_DIR_MODEL_LIVENESS        = "models/liveness/"

# Set width and height
RESOLUTION_QVGA   = (320, 240)
RESOLUTION_VGA    = (640, 480)
RESOLUTION_HD     = (1280, 720)
RESOLUTION_FULLHD = (1920, 1080)



def cam_init(cam_index, width, height): 
    cap = cv2.VideoCapture(cam_index)
    if sys.version_info < (3, 0):
        cap.set(cv2.cv.CV_CAP_PROP_FPS, 30)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

'''this function draws the frames to the window'''
def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
    if face_id is not None:
        if confidence is not None:
            if confidence > 60.0:   #threshold confidence for unknown class
                text = "{} {:.2f}%".format(face_id, confidence)
            else:
                text = "{}".format("Unknown")                
        else:
            text = "{}".format(face_id)
        cv2.putText(frame, text, (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)





# process_livenessdetection is supposed to run before process_facerecognition
def process_livenessdetection(model_detector, model_recognizer, model_liveness, cam_index, cam_resolution):

    # Initialize the camera
    camera = cam_init(cam_index, cam_resolution[0], cam_resolution[1])

    try:
        # Initialize face detection
        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)

        # Initialize face recognizer
        face_encoder = FaceEncoder(model=model_recognizer, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

        # Initialize face liveness detection
       
        face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)

    except:
        print("Error, check if models and trained dataset models exists!")
        return

    face_id, confidence = (None, 0)

    time_start = time()
    time_elapsed = 0
    frame_count = 0
    identified_unique_faces = {} # dictionary    
    is_fake_count_print = 0

    while (True): 
        # Capture frame from webcam
        ret, frame = camera.read()
        if frame is None:
            print("Error, check if camera is connected!")
            break
        # Detect and identify faces in the frame
        # Indentify face based on trained dataset (note: should run facial_recognition_training.py)
        # if no face detected, then conclude their is no person
        faces = face_detector.detect(frame)
        try:   
            if len(faces) == 0:
                height, width = frame.shape[:2]
                cv2.putText(frame, "No Person Detected", (int(height/2),int(width/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        except Exception as e:
            print(e)
            pass
        
        #authorised_people = ["shubham", "bua"]         
        for (index, face) in enumerate(faces):          

            # Detect if frame is a print attack or replay attack based on colorspace
            is_fake_print  = face_liveness2.is_fake(frame, face)          
          e
            if is_fake_print:
                is_fake_count_print += 1                
                face_id, confidence = ("fake", None)
                cv2.putText(frame, "Alert! Unauthorised Access Detected....", (int(height/10),int(width/10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                face_id, confidence = face_encoder.identify(frame, face)
                try:
                    if face_id.lower() not in authorised_people or ((face_id.lower() in authorised_people) and confidence < 50):
                        height, width = frame.shape[:2]
                        cv2.putText(frame, "Alert! Unauthorised Access Detected....", (int(height/10),int(width/10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)                     
                        
                except Exception as e:
                    print(e)
                    print(frame.shape[1])                                      
                    pass     
                if face_id not in identified_unique_faces:
                    identified_unique_faces[face_id] = 1
                else:
                    identified_unique_faces[face_id] += 1

            label_face(frame, face, face_id, confidence) # Set text and bounding box on face
        # Update frame count
        frame_count += 1
        time_elapsed = time()-time_start

        # Display updated frame
        cv2.imshow(WINDOW_NAME, frame)

        # Check for user actions
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break   

    # Release the camera
    camera.release()
    cv2.destroyAllWindows()



def main(args):
    if sys.version_info < (3, 0):
        print("Error: Python2 is slow. Use Python3 for max performance.")
        return

    cam_index = int(args.webcam)
    resolutions = [ RESOLUTION_QVGA, RESOLUTION_VGA, RESOLUTION_HD, RESOLUTION_FULLHD ]
    try:
        cam_resolution = resolutions[int(args.resolution)]
    except:
        cam_resolution = RESOLUTION_QVGA

    if args.detector and args.encoder and args.liveness:
        try:
            detector = FaceDetectorModels(int(args.detector))
            encoder  = FaceEncoderModels(int(args.encoder))
            liveness = FaceLivenessModels(int(args.liveness))
            print( "Parameters: {} {} {}".format(detector, encoder, liveness) )
            process_livenessdetection(detector, encoder, liveness, cam_index, cam_resolution)
        except:
            print( "Invalid parameter" )
        return
    run(cam_index, cam_resolution)

# I used facenet for face detecction and recogniyion, COLORSPACE_YCRCBLUV for fake/liveliness detection, and a neural net classifier for training
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', required=False, default=0, 
        help='Detector model to use. Options: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET')
    parser.add_argument('--encoder', required=False, default=0, 
        help='Encoder model to use. Options: 0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET')
    parser.add_argument('--liveness', required=False, default=0, 
        help='Liveness detection model to use. Options: 1-COLORSPACE_YCRCBLUV')
    parser.add_argument('--webcam', required=False, default=0, 
        help='Camera index to use. Default is 0. Assume only 1 camera connected.)')
    parser.add_argument('--resolution', required=False, default=0,
        help='Camera resolution to use. Default is 0. Options: 0-QVGA, 1-VGA, 2-HD, 3-FULLHD')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
