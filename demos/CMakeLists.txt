cmake_minimum_required(VERSION 3.2)

project(experiments)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(threshold threshold.cpp util.cpp ../src/instatrack.cpp)
target_link_libraries(threshold ${OpenCV_LIBS})

add_executable(color color.cpp util.cpp)
target_link_libraries(color ${OpenCV_LIBS})

add_executable(corner_harris corner_harris.cpp util.cpp)
target_link_libraries(corner_harris ${OpenCV_LIBS})

add_executable(naive_chessboard naive_chessboard.cpp util.cpp)
target_link_libraries(naive_chessboard ${OpenCV_LIBS})

add_executable(naive_circles naive_circles.cpp util.cpp)
target_link_libraries(naive_circles ${OpenCV_LIBS})

add_executable(my_chessboard my_chessboard.cpp util.cpp ../src/instatrack.cpp)
target_link_libraries(my_chessboard ${OpenCV_LIBS})

add_executable(instatrack instatrack.cpp ../src/instatrack.cpp)
target_link_libraries(instatrack ${OpenCV_LIBS})

if (CMAKE_COMPILER_IS_GNUCC)
	set_property( TARGET instatrack APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )
endif (CMAKE_COMPILER_IS_GNUCC)
