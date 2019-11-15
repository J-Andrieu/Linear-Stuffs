PROJECT(LinAlgo)
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

IF(DONT_USE_GPU MATCHES True OR DONT_USE_GPU MATCHES true)
    set(DONT_USE_GPU TRUE)
ELSE()
    set(DONT_USE_GPU FALSE)
ENDIF(DONT_USE_GPU MATCHES True OR DONT_USE_GPU MATCHES true)

find_package(OpenCL)
IF (NOT(OpenCL_FOUND))
    set(DONT_USE_GPU TRUE)
ENDIF(NOT(OpenCL_FOUND))

IF(CMAKE_BUILD_TYPE MATCHES Debug)
  #opencl and address sanitizer hate each other
  #SET(CMAKE_CXX_FLAGS "-fsanitize=address -ggdb ${CMAKE_CXX_FLAGS}")
  SET(CMAKE_XX_FLAGS "-ggdb ${CMAKE_CXX_FLAGS}")
  ADD_DEFINITIONS(-DDEBUG)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

SET(CXX11_FLAGS "-pthread -std=gnu++17")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX11_FLAGS}")

IF(UNIX)
  ADD_DEFINITIONS(-DUNIX)
ENDIF(UNIX)

IF(DONT_USE_GPU)
    ADD_DEFINITIONS(-DDONT_USE_GPU=1)
ENDIF(DONT_USE_GPU)

# Set Includes
SET(INCLUDES ${PROJECT_SOURCE_DIR}/include)
IF(OpenCL_FOUND AND NOT(DONT_USE_GPU))
	INCLUDE_DIRECTORIES(${INCLUDES} ${OpenCL_INCLUDE_DIRS})
	link_directories(${OpenCL_LIBRARY})
ELSE()
    INCLUDE_DIRECTORIES(${INCLUDES})
ENDIF(OpenCL_FOUND AND NOT(DONT_USE_GPU))

# Build LinAlgo static library
FILE(GLOB Lin_SOURCES "${CMAKE_SOURCE_DIR}/src/*")
FILE(GLOB Lin_HEADERS "${CMAKE_SOURCE_DIR}/include/*")
ADD_LIBRARY(linalgo ${Lin_SOURCES} ${Lin_HEADERS})

IF(Build_Tests)
	ADD_CUSTOM_TARGET(run
	    COMMAND "LinAlgo_test"
	    DEPENDS "LinAlgo_test"
	    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
	)

	# Build test suite library
	FILE(GLOB test_SOURCES "${CMAKE_SOURCE_DIR}/test/CMDParser/src/*"
			       "${CMAKE_SOURCE_DIR}/test/Timer/src/*")
	FILE(GLOB test_HEADERS "${CMAKE_SOURCE_DIR}/test/CMDParser/include/*"
			       "${CMAKE_SOURCE_DIR}/test/Timer/include/*"
			       "${CMAKE_SOURCE_DIR}/test/utilities.h")
	ADD_LIBRARY(UTILS ${test_SOURCES} ${test_HEADERS})

	FILE(GLOB test_source "${CMAKE_SOURCE_DIR}/test/test.cpp")
	ADD_EXECUTABLE("LinAlgo_test" ${test_source})
	IF(OpenCL_FOUND AND NOT(DONT_USE_GPU))
		TARGET_LINK_LIBRARIES("LinAlgo_test" linalgo UTILS ${OpenCL_LIBRARY})
	ELSE()
		TARGET_LINK_LIBRARIES("LinAlgo_test" linalgo UTILS)
	ENDIF(OpenCL_FOUND AND NOT(DONT_USE_GPU))
ENDIF(Build_Tests)


IF(Build_Tests)
ADD_CUSTOM_TARGET("${PROJECT_NAME}_SUCCESSFUL" ALL
		  DEPENDS "linalgo"
		  DEPENDS "UTILS"
                  DEPENDS "LinAlgo_test"
                  COMMAND ${CMAKE_COMMAND} -E echo ""
                  COMMAND ${CMAKE_COMMAND} -E echo "====================="
                  COMMAND ${CMAKE_COMMAND} -E echo "  Compile complete!"
                  COMMAND ${CMAKE_COMMAND} -E echo "====================="
                  COMMAND ${CMAKE_COMMAND} -E echo "${CMAKE_CURRENT_BINARY_DIR}"
                 )
ELSE()
ADD_CUSTOM_TARGET("${PROJECT_NAME}_SUCCESSFUL" ALL
                  DEPENDS "linalgo"
                  COMMAND ${CMAKE_COMMAND} -E echo ""
                  COMMAND ${CMAKE_COMMAND} -E echo "====================="
                  COMMAND ${CMAKE_COMMAND} -E echo "  Compile complete!"
                  COMMAND ${CMAKE_COMMAND} -E echo "====================="
                  COMMAND ${CMAKE_COMMAND} -E echo "${CMAKE_CURRENT_BINARY_DIR}"
                 )
ENDIF()
