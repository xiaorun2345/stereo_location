# stereo_location
使用双目相机进行足球定位，可自动防守
1、左右相机均进行图像识别，定位出左右相机得图像坐标
2、对双目相机进行标定，得到相机内参、外参
3、针对图像坐标输入，输出三维世界坐标，包括X，Y，Z
4、控制机器人进行拦截球