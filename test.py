import time
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('Braintumor_model.h5')
total_yes = [0,0]
total_no = [0,0]
start_time = time.time()
for j in range(0,2):
    for i in range(100,301):
        nopath = 'datasets\\no\\no'+str(i)+'.jpg'
        yespath = 'datasets\\yes\y'+str(i)+'.jpg'
        if(j == 0):
            image = cv2.imread(yespath)
        else:
            image = cv2.imread(nopath)
        img = Image.fromarray(image)
        img=img.resize((64,64))
        img=np.array(img)
        input_img = np.expand_dims(img, axis=0)
        result = model.predict(input_img)
        if result[0][0] == 1:
            label = "yes"
            print('\nThe patient suffering from brain tumor')
            total_yes[j] += 1
            # cv2.imshow("Yes",image)
            # cv2.waitKey(0)
        else:
            label = "no"
            print('No tumor...')
            total_no[j] += 1
    # cv2.imshow("No",image)
    # cv2.waitKey(0)
# cv2.putText(image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print("Results:")
print("Total Yes from Yes images:"+str(total_yes[0])+", No from Yes images:"+str(total_no[0]))
print("No Yes from No images:"+str(total_no[1])+", No from Yes images:"+str(total_yes[1]))
end_time = time.time()
total_time = end_time - start_time
print("Total time taken:", total_time)