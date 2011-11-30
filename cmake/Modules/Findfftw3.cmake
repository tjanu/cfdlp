# - Try to find fftw3
# Once done this will define
#
#  fftw3_FOUND - system has fftw3
#  fftw3_INCLUDE_DIR - the fftw3 include directory
#  fftw3_LIBRARIES - Link these to use fftw3

FIND_PATH(fftw3_INCLUDE_DIR fftw3.h)

FIND_LIBRARY(fftw3_LIBRARIES NAMES fftw3)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(fftw3 DEFAULT_MSG fftw3_LIBRARIES fftw3_INCLUDE_DIR)

MARK_AS_ADVANCED(fftw3_INCLUDE_DIR fftw3_LIBRARIES)
