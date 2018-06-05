import cv2 #imports cv to work with

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #creates a cascade element; takes default settings from Haar cascade xml file

img=cv2.imread("s-p-b-bez-fd.jpg") #reads the image where we want to find the face
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converts the image to a grey version

faces=face_cascade.detectMultiScale(gray_img,
scaleFactor=1.1,
minNeighbors=3) #now this line uses the detect tool and defines what image to use, what scale factor and whate are the min minNeighbors - this is all based on the cascade itself
                #this will produce a list with 4 values [[157  84 379 379]] - starting coordinate width (157), height(84) and the height(379) and widht (379) of the rectangle where the face is

for x, y, w, h in faces: #this for loop writes a rectangle on the photo
    img=cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3) #this line writes the rectangle actually, it takes 3 arguments - what image, starting point and conversely the oposite point of the rectangle
                                                            #the last tuple is the color of the rectange, the last integer is a width rectangle.

resized=cv2.resize(img,(int(img.shape[1]/3), int(img.shape[1]/3))) #resizes the enf image

cv2.imwrite("s-p-b-sa-fd.jpg", resized)

cv2.imshow("gray", img)
cv2.waitKey()
cv2.destroyAllWindows
