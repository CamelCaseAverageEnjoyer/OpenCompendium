cd img/11/
 ffmpeg -f image2 -framerate 12  -start_number 1 -i gibbon_%04d.jpg -s 1920x480  -b:v 10000k ../../res_hrd_1.gif
cd ../..
