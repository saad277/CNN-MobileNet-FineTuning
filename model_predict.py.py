from keras.models import load_model
import os 
import numpy as np
import cv2

model=load_model("model_save.h5")

#print(model.summary())


dictionary={"[0]":"mantled_howler",
                   "[1]":"patas_monkey",
                   "[2]":"bald_uakari",
                   "[3]":"japnese_macaque",
                   "[4]":"pygmy_marmoset",
                   "[5]":"white_headed_capuchin",
                   "[6]":"silvery_marmoset",
                   "[7]":"common_squirrel_monkey",
                   "[8]":"black_headed_night_smokey",
                   "[9]":"nilgiri_langur",}



def draw_test(name,pred,img):
    monkey=dictionary[str(pred)]
    black=[0,0,0]
    expanded_image=cv2.copyMakeBorder(img,80,0,0,100,cv2.BORDER_CONSTANT,value=black)
    cv2.putText(expanded_image,monkey,(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow(name,expanded_image)



path="./test"

image_names=os.listdir(path)

random_number=np.random.randint(0,len(image_names))

print(random_number)

original_img=cv2.imread("./test/"+image_names[random_number],1)

img=cv2.resize(original_img,(224,224),interpolation=cv2.INTER_LINEAR)

img=img/255

img=img.reshape(1,224,224,3)

res=np.argmax(model.predict(img,1,verbose=0),axis=1)

draw_test("Prediction ",res,original_img)

cv2.waitKey(0)

cv2.destroyAllWindows();











