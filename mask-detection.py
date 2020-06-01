# coding: utf-8
from tiny_fd import TinyFacesDetector
import sys
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

if __name__ == '__main__':
    
    #maskNet = load_model('model.20-0.99-0.95.hdf5')
    maskNet = load_model('model.20-1.00-0.97.hdf5')
    footage = 'demo.mp4'
    detector = TinyFacesDetector(model_root='./', prob_thresh=0.5, gpu_idx=0)
    cap = cv2.VideoCapture('test/videos/'+footage)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_movie = cv2.VideoWriter(footage+'.avi', fourcc, fps, (width, height))
#    if len(sys.argv) == 2:
#        path = sys.argv[1]
#    else:
#        path = './selfie.jpg'
    while cap.isOpened():
        
        ret, img = cap.read()
        if not ret:
            break
        #img = cv2.imread(path)
        boxes = detector.detect(img)
        print('Faces detected: {}'.format(boxes.shape[0]))

        for r in boxes:
            faces = []
            locs = []
            preds = []
            face = img[r[1]:r[3], r[0]:r[2]]


            #print('FACE',face)
            if face.size and isinstance(face, np.ndarray):
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                #print('face',face)
                face = cv2.resize(face, (224, 224))
                #cv2.imwrite('image1.jpg',face)
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
                faces.append(face)
                locs.append((r[0], r[1], r[2], r[3]))
            if len(faces) > 0:
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                #print('faces len',len(faces))
                preds = maskNet.predict(faces)

            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                #(startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                print('Mask',mask)
                print("Withoutmak",withoutMask)
                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(img, label, (r[0],r[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(img, (r[0],r[1]), (r[2],r[3]), color, 2)



                #cv2.rectangle(img, (r[0],r[1]), (r[2],r[3]), (255,255,0),3)
        output_movie.write(img)
        cv2.namedWindow('Tiny FD', cv2.WINDOW_NORMAL)
        cv2.imshow('Tiny FD', img)
        #cv2.waitKey()
        cv2.resizeWindow('Frame',800,600)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
