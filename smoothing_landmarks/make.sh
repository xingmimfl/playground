rm -rf main *.o
g++ -c OneEuroFilter.cpp -o OneEuroFilter.o
g++ -c smoothing.cpp -o smoothing.o --std=c++17 -lstdc++fs -Wall\
  -I .\
  -I xxxx/opencv/include\
  -L xxxx/opencv/lib\
  -lopencv_imgcodecs\
  -lopencv_imgproc\
  -lopencv_core
g++ -o main smoothing.o OneEuroFilter.o --std=c++17 -lstdc++fs -Wall\
  -I .\
  -I xxxx/opencv/include\
  -L xxxx/opencv/lib\
  -lopencv_imgcodecs\
  -lopencv_imgproc\
  -lopencv_core
