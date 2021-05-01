# opening the camera and its live vision with opencv
import cv2
frame_final =""
i=0
cap= cv2.VideoCapture(0)
while True:

 
 ret, frame = cap.read()
 

 cv2.imshow("tracking", frame)
 print("Press c when you are ready (it has to be lowercase)")	

 #when user presses c in keyboard, then program takes photo of that moment and save as loaded_image.jpeg
 if cv2.waitKey(25) & 0xFF == ord('c'):
    frame_final=frame
    cv2.imwrite("loaded_image.jpeg", frame_final)
    break
print("Your photo has taken and being processed...")

#closing the camera
cap.release()
cv2.destroyAllWindows()
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

# load an image and predict the class
def run_example(filename):
	# load the image
	img = load_image(filename)
	# load model
	model = load_model('final_model.h7')
	# predict the class
	result = model.predict(img)
	print(result[0])
	return(result[0])

returned=run_example("loaded_image.jpeg")
to_str = str(returned)


if "1" in to_str:
	
	img = cv2.imread("approvement_icon.jpeg")
	cv2.imshow("You have a mask",img)
	cv2.waitKey() 
	print("You have a mask, you can enter to the mall")
	
else:
	
	img = cv2.imread("rejected_icon.jpeg")
	cv2.imshow("Mask did not detected",img)
	cv2.waitKey() 
	print("Mask did not detected. Follow the instructions, mask will be drop in few seconds")
