# CMakeLists.txt
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(LibtorchSample)

set(CMAKE_CXX_STANDARD 17)

# libtorchのパスを指定（例: 解凍した場所がプロジェクトと同階層の場合）
set(Torch_DIR "/usr/local/lib/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(main inference.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)