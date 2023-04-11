import tensorflow as tf
import numpy as np
import cv2
import pathlib
import time



interpreter = tf.lite.Interpreter(model_path="../models/model-111/output_tflite_graph.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

interpreter.allocate_tensors() # allocate memory

def draw_rect(image, box):
    y_min = int(max(1, (box[0] * image.shape[0])))
    x_min = int(max(1, (box[1] * image.shape[1])))
    y_max = int(min(image.shape[0], (box[2] * image.shape[0])))
    x_max = int(min(image.shape[1], (box[3] * image.shape[1])))
    
    # draw a rectangle on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#c = 0 # counter
webcam = cv2.VideoCapture(0)
fps = "0"

inf_en = True

#for file in pathlib.Path('images').iterdir():
while True:
    start = time.time()
#    c = c + 1
#    if file.suffix != '.jpg' and file.suffix != '.png':
#        continue
    
    check, frame = webcam.read()    
    if check:
        #img = cv2.imread(r"{}".format(file.resolve())) # read image from file
        img = frame
        if inf_en:
            img = cv2.resize(img, (300, 300)) # resize image to 300x300
            interpreter.set_tensor(input_details[0]['index'], [img]) # set input tensor

            interpreter.invoke() # run inference
            rects = interpreter.get_tensor(
                output_details[0]['index'])

            scores = interpreter.get_tensor(
                output_details[2]['index'])
            
            for index, score in enumerate(scores[0]):
            #    print(score)
            # Draw a rectangle if the score is high (>0.8) or it's the highest score and it's above 0.5
                if ((score > 0.8) or (score > 0.5 and index == 0)):
                  try:
                      draw_rect(img,rects[0][index])
                  except:
                      print(rects[0])

            cv2.putText(img, "TF: " + fps + " fps", [0,25], cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0 ,255), 3)
        else:
            cv2.putText(img, "Webcam: " + fps + " fps", [0,25], cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0 ,255), 3)
        cv2.imshow("image", img)        
    end = time.time()    
    fps = '%0.1f' % (1/(end - start))
    key_press = cv2.waitKey(1)
    if key_press == ord('q'):
        break
    elif key_press == ord('i'):
        inf_en = not inf_en
    elif key_press == ord('s'):
        cv2.imwrite("output.jpg",img)
    


webcam.release()
cv2.destroyAllWindows()
