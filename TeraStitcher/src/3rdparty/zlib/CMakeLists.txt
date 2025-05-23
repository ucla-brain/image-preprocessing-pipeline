# ----------------------------------------------------------------------------
#  CMake file for zlib. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

project(zlib C)
cmake_minimum_required(VERSION 2.7)

# hiding unused predefined variables
set (CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE INTERNAL "")

include(CheckFunctionExists)
include(CheckIncludeFile)
include(CheckCSourceCompiles)
include(CheckTypeSize)

#
# Check for fseeko
#
check_function_exists(fseeko HAVE_FSEEKO)
if(NOT HAVE_FSEEKO)
  add_definitions(-DNO_FSEEKO)
endif()

#
# Check for unistd.h
#
check_include_file(unistd.h Z_HAVE_UNISTD_H)

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()

#
# Check to see if we have large file support
#
check_type_size(off64_t OFF64_T)
if(HAVE_OFF64_T)
  add_definitions(-D_LARGEFILE64_SOURCE=1)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/zconf.h.cmakein"
               "${CMAKE_CURRENT_BINARY_DIR}/zconf.h" @ONLY)
			   
include_directories (${CMAKE_CURRENT_SOURCE_DIR})
include_directories (${CMAKE_CURRENT_BINARY_DIR})
#ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

set(ZLIB_PUBLIC_HDRS
    "${CMAKE_CURRENT_BINARY_DIR}/zconf.h"
    zlib.h
)
set(ZLIB_PRIVATE_HDRS
    crc32.h
    deflate.h
    gzguts.h
    inffast.h
    inffixed.h
    inflate.h
    inftrees.h
    trees.h
    zutil.h
)
set(ZLIB_SRCS
    adler32.c
    compress.c
    crc32.c
    deflate.c
    gzclose.c
    gzlib.c
    gzread.c
    gzwrite.c
    inflate.c
    infback.c
    inftrees.c
    inffast.c
    trees.c
    uncompr.c
    zutil.c
)

add_library(zlib STATIC ${ZLIB_SRCS} ${ZLIB_PUBLIC_HDRS} ${ZLIB_PRIVATE_HDRS})
#set_target_properties(zlib PROPERTIES DEFINE_SYMBOL ZLIB_DLL)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

#disabling annoying warnings
warnings_disable(CMAKE_CXX_FLAGS /wd4290) # vs2012: can't properly handle exceptions
warnings_disable(CMAKE_CXX_FLAGS /wd4996) # vs2012: complains about unsafe standard C++ functions

#ocv_warnings_disable(CMAKE_C_FLAGS -Wattributes -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations)

# set_target_properties(zlib PROPERTIES
        # OUTPUT_NAME zlib		
        # ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
        # #DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        # #ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
    # )

# if(ENABLE_SOLUTION_FOLDERS)
  # set_target_properties(zlib PROPERTIES FOLDER "3rdparty")
# endif()

# if(NOT BUILD_SHARED_LIBS)
  # install(TARGETS zlib ARCHIVE DESTINATION ${3P_LIBRARY_OUTPUT_PATH} COMPONENT main)
# endif()
