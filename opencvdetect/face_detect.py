import cv2
import sys,os
# Get user supplied values
#imagePath = sys.argv[0]	#python test.py --t help --v 那么sys.argv就是['test.py', '--t', 'help', '--v']
#imagePath = os.getcwd()	#获取当前路径
#print(os.getcwd())
imagePath='Image_Test/heat.jpg'	#当前目录下的图片
cascPath = 'haarcascades/haarcascade_frontalface_default.xml'#其实OpenCV安装包里自带有已经训练好的人脸分类器“haarcascade_frontalface_alt.xml”，位置在“XX\opencv\sources\data\haarcascades”里，这个文件夹下还有其他一些分类器，像左右眼、上身、笑脸检测等等。

# Create the haar cascade，加载级联分类器，相当于人脸特征数据
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath,1)
print(type(image))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image，多尺度检测
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,			#图像的缩放因子
    minNeighbors=5,				#为每一个级联矩形应该保留的邻近个数，可以理解为一个人周边有几个人脸
    minSize=(30, 30),			#检测窗口的大小
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces，获得人脸矩形的坐标和宽高并画矩形
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# cv2.imwrite(imagePath, image)
cv2.imshow("Faces found", image)
if cv2.waitKey(0) & 0xFF == ord('q'):
	exit()