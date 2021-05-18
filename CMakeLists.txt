cmake_minimum_required(VERSION 3.12)

cmake_policy(SET CMP0012 OLD)

if (WIN32)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_DEBUG} /MT")

	set(VCPKG_TARGET_TRIPLET "x64-windows-static-v141")
endif()

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

project(LegendaryWaffle)

set(TESSDATA_DIR "" CACHE STRING "Location of https://github.com/tesseract-ocr/tessdata")

if (NOT EXISTS "${TESSDATA_DIR}")
	message(FATAL_ERROR "Please spcify the location of the tessdata repository (https://github.com/tesseract-ocr/tessdata) with -DTESSDATA_DIR")
endif()

find_package(OpenCV CONFIG REQUIRED)
find_package(Tesseract CONFIG REQUIRED)
find_package(BZip2 REQUIRED)
find_package(LibXml2 REQUIRED)
find_package(lz4 CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(zstd CONFIG REQUIRED)

set(SOURCES
	src/Source.cpp
	src/ResourceMatcher.h
	src/ResourceMatcher.cpp
	src/Utils.hpp
	src/OCR.h
	src/OCR.cpp
)

add_executable(${PROJECT_NAME} "${SOURCES}")

if(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX /wd4505)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBXML2_INCLUDE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE TESSDATA_DIR="${TESSDATA_DIR}")

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Tesseract_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE BZip2::BZip2)
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBXML2_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE lz4::lz4)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${PROJECT_NAME} PRIVATE zstd::libzstd_static)
