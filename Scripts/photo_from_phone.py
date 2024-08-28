# Import essential libraries 
import requests, os, cv2, imutils
import numpy as np
from url_phone import url
  
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last. 

  
# While loop to continuously fetching data from the Url 
while True:
    
    img_resp = requests.get(url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
    img = cv2.imdecode(img_arr, -1) 
    img = imutils.resize(img, width=1000, height=1800) 
    cv2.imshow("Android_cam", img) 

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        number = [int(key.replace('.jpg', '').replace('shoot_','')) for key in os.listdir() 
                  if key.endswith('.jpg') and key.startswith('shoot_')]
        cv2.imwrite(f"shoot_{max(number)+1}.jpg", img)
    # Press Esc key to exit 
    if (cv2.waitKey(1) & 0xFF) == ord('w'): 
        break

cv2.destroyAllWindows()
