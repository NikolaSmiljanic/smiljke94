import cv2

from scipy import ndimage
from sympy.physics.secondquant import wicks
from vector import pnt2line2
from skimage import color
from skimage.measure import label
from skimage.measure import regionprops
import numpy as np
import math
from matplotlib.pyplot import cm
from Crypto.PublicKey.pubkey import pubkey
from sklearn.datasets import fetch_mldata
from matplotlib.pyplot import cm
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage.morphology import *
import itertools
from skimage import color
import numpy as np
from skimage import color

import time


from skimage.morphology import *
from skimage import color
import numpy as np

print("poc Prgo poc prggg")

videoName="data/video-1.avi"
cap = cv2.VideoCapture(videoName)
print (videoName)

#x1,y1,x2,y2=findLineParams(videoName)
global x1
global y1
global linesForF
sumaBr = 0

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)
    
def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)
    
def distance(p0,p1):
    return length(vector(p0,p1))
    
def findPoints(lines):    
    dist=0
    Xmin=10000
    Ymin=10000
    Ymax=1
    Xmax=1
    for i in  range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            
            k1=x2-x1
            k2=y1-y2
            dist= math.sqrt(k1*k1 - k2*k2)
            print(dist)
            print("x1={x1}".format(x1=x1))
            print("y1={y1}".format(y1=y1))
            print("-----------------------")
            print("x2={x2}".format(x2=x2))
            print("y2={y2}".format(y2=y2))
            if x1<Xmin :
                Xmin=x1
                Ymin=y1
            if x2>Xmax: #and y2>60:
                Ymax=y2
                Xmax=x2
   
    return Xmin,Ymin,Xmax,Ymax
    

def houghTransformtion(frame,grayImg,minLineLength,maxLineGap):
    edges = cv2.Canny(grayImg,50,150,apertureSize = 3)
    cv2.imwrite('lineDetected13.jpg',frame)# proveri da l treba
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength, maxLineGap)
    
    minx=9999
    miny=9999
    maxy=-2
    maxx=-2
    minx,miny,maxx,maxy=findPoints(lines)
    cv2.line(frame, (minx,miny), (maxx, maxy), (233, 0, 0), 2)
    return minx,miny,maxx,maxy

def printCords(x1,y1,x2,y2):
    print("***********************************")
    print("___________________________________")
    print("Donja leva tacka :")
    print("x1={x1}".format(x1=x1))
    print("y1={y1}".format(y1=y1))
    print("Gornja desna tacka :")
    print("x2={x2}".format(x2=x2))
    print("y2={y2}".format(y2=y2))
    print("___________________________________")
    print("***********************************")
 
    
def myFunct(cap):
    gray="grayFrame"  
    frame1="frame"
    kernel = np.ones((2,2),np.uint8)
    i=0
    if i==0:
        i=i+1
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #dodaj diletac i tu slike prosledi a ne greja
                img0 = cv2.dilate(gray, kernel)
                frame1=frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break
    return frame1,img0
    
def findLineParams(videoName):
    cap = cv2.VideoCapture(videoName)
    maxL=610
    gap=9
    g=""
    f=""
    f,g=myFunct(cap)
    print("Start")
    print(cv2.__version__)
    cap.release()
    cv2.destroyAllWindows()
    return houghTransformtion(f,g,maxL,gap)
    
def update():
   # print("update1")
    global x1
    global y1
    if videoName == "data/video-1.avi" :
        x1 = x1 - 10
        y1 = y1 + 10
    
    if videoName == "data/video-6.avi" :
        x1 = x1 - 10
        y1 = y1 + 10
    
#update() 

def update2(img0):
    kernel = np.ones((3,3),np.uint8)
    if videoName == "data/video-1.avi" :
        img0 = cv2.dilate(img0, kernel)
    if videoName == "data/video-5.avi" :
        img0 = cv2.dilate(img0, kernel)
        img0 = cv2.dilate(img0, kernel)
    if videoName == "data/video-6.avi" :
        img0 = cv2.dilate(img0, kernel)
    return img0

        
        
           
#line = [(x1, y1), (x2, y2)]

dd = -1
def convertIMG(img):
    
    img_BW=img==1.0
    return img_BW
    
def nextId():
    global dd
    dd += 1
    return dd

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        
        if(mdist<r):
            #print "distanca***** " + format(mdist)
            retVal.append(obj)

            
    return retVal

def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    img_gray = 0.5*img_rgb[:, :, 0] + 0.0*img_rgb[:, :, 1] + 0.5*img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')
    return img_gray

    
def findClosest(list,elem):

        
    temp = list[0]
    for obj in list:
        if distance(obj['center'],elem['center']) < distance(temp['center'],elem['center']):
            temp = obj
    

    
    return temp
def putInLeftCorner(img_BW):
    try:
        label_img = label(img_BW)
        regions = regionprops(label_img) # ctrl space
    #    plt.imshow(img_BW,'gray')
    #    plt.show()
        #print("Broj regiona " + format(len(regions)))
        newImg="novaSlika"
        minx=750
        miny=750
        maxx=-2
        maxy=-2
        
        for region in regions:
            bbox = region.bbox
            if bbox[0]<minx:
                minx=bbox[0]
            if bbox[1] <miny:
                miny=bbox[1]
            if bbox[2]>maxx:
                maxx=bbox[2]
            if bbox[3]>maxy:
                maxy=bbox[3]
       
        height = maxx - minx
        width = maxy - miny
    
        newImg = np.zeros((28, 28))
  
        newImg[0:height, 0:width] = newImg[0:height, 0:width] + img_BW[minx:maxx, miny:maxy]
       
        return newImg
    except ValueError:  #raised if `y` is empty.
        pass

#def DrawHoughDots():


def drawPoints(imget,pointArr):
    cv2.circle(imget, (pointArr[0]), 4, (25, 25, 255), 1)
    cv2.circle(imget, (pointArr[1]), 4, (25, 25, 255), 1)
    
def drawAllPoints(imget,linesAll):
    s=20
    for i in  range(len(linesAll)):
        for x1, y1, x2, y2 in linesAll[i]:
            cv2.circle(imget, (line[0]), 4, (2, 2, s), 1)
            cv2.circle(imget, (line[1]), 4, (2, 2, s), 1)
            s=s+20
    
    
def getDigit(img):
    img_BW=color.rgb2gray(img) >= 0.88
 
    img_BW=(img_BW).astype('uint8')

    newImg=putInLeftCorner(img_BW)
    i=0;
    minSum = 9999
    rez = -1
    while i<70000:
         sum=0
         mnist_img=new_mnist_set[i]
         #cv2.imshow('mnistGore', mnist_img)
         #cv2.waitKey()
         #        mnist_img=mnist.data[i].reshape(28,28)
         #  new_mnist_img=putInLeftCorner(mnist_img)
         sum=np.sum(mnist_img!=newImg)
         if sum < minSum:
             minSum = sum
             rez = mnist.target[i]
         i=i+1
         #print("Proslijedjeni broj je " + format(rez))
    
    #print("Proslijedjeni broj je " + format(rez))
    return rez;
# color filter
new_mnist_set=[]
def transformMnist(mnist):

    i=0;
    while i < 70000:
        mnist_img=mnist.data[i].reshape(28,28)
        mnist_img_BW=((color.rgb2gray(mnist_img)/255.0)>0.88).astype('uint8')
        
       
        new_mnist_img=putInLeftCorner(mnist_img_BW)
      
#        new_mnist_set[i]=new_mnist_img
        new_mnist_set.append(new_mnist_img)
        
        i=i+1
def ObjectId(elem,c1,c2,img,t,elements):
    x1=c1-14
    x2=c1+14
    y1=c2-14
    y2=c2+14
    elem['id'] = nextId()
    elem['t'] = t
    elem['pass'] = False
          #          elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
           #         elem['future'] = []
    elem['value'] = getDigit(img[y1:y2,x1:x2])
    elements.append(elem)
    return(x1,x2,y1,y2)

def main():
#    update()
    print("PROGPROGPROG")
    
    boundaries = [([230, 230, 230], [255, 255, 255]) ]
    
    print mnist.target[15000] #r
    transformMnist(mnist) # 
    kernel = np.ones((2,2),np.uint8) 
   
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('images/output-rezB.avi',fourcc, 20.0, (640,480))
   

    elements = []
    t =0
    counter = 0
    times = []

    while (1):
        start_time = time.time()
        ret, img = cap.read()
        if not ret:
            break
        
        (lower, upper) = boundaries[0]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)# 
        
        
        img0 = 1.0 * mask
        img01 = 1.0 * mask
        drawPoints(img,line)
           
        img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
        img0 = cv2.dilate(img0, kernel)
        img0 = update2(img0)
        
        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
      
        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))

            if (dxc > 11 or dyc > 11):
               
                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
               
                lst = inRange(20, elem, elements)
                
                nn = len(lst)
                if nn == 0:
                    x1,x2,y1,y2=ObjectId(elem,xc,yc,img,t,elements)
                    
                else:
                    el = findClosest(lst,elem)
                    el['center'] = elem['center']
                    el['t'] = t
         
        for el in elements:
            tt = t - el['t']
           # x=pnt2line(0,0,0)
            if (tt < 3):
                dist, pnt, r = pnt2line2(el['center'], line[0], line[1])
                if r > 0:
                    #vratiti
                    cv2.line(img, pnt, el['center'], (0, 255, 25), 1)##################
                    
                    if (dist < 10):
                        
                        if el['pass'] == False:
                            el['pass'] = True
                            counter += 1
                            (x,y)=el['center']
                            (sx,sy)=el['size']
                    
                            y1=y-14
                            y2=y+14
                            x1=x-14
                            x2=x+14
                            
                            
                            (p3,p4)=(x2,y2)
                            (p1,p2)=(x1,y1)
                            cv2.rectangle(img,(p1,p2),(p3,p4),(255,255,0),3)
                            print("___________")
                            print "Prepoznao:  " + format(el['value'])
                            global sumaBr
                            sumaBr += el['value']
                            cv2.waitKey()
                         
                            #print ("Pronadjen centar({x},{y}) velicina({sx},{sy})".format(x=x, y=y,sx=sx,sy=sy))
                            
             
          #      cv2.circle(img, el['center'], 2, (25,25,255), 2)
                id = el['id']
                


        elapsed_time = time.time() - start_time
        times.append(elapsed_time * 1000)
        cv2.putText(img, 'Sum: ' + str(sumaBr), (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        # print nr_objects
        t += 1

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        out.write(img)
    print sumaBr

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    et = np.array(times)


 
x1,y1,x2,y2=findLineParams(videoName)
line = [(x1, y1), (x2, y2)]	
printCords(x1,y1,x2,y2)

mnist = fetch_mldata('MNIST original') 
main()