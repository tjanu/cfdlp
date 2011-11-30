INCLUDE(ExternalProject)

ExternalProject_Add(libfftw
	DOWNLOAD_DIR .
	URL http://www.fftw.org/fftw-3.2.2.tar.gz
	SOURCE_DIR fftw-3.2.2
	CONFIGURE_COMMAND ./configure --prefix=${CMAKE_BINARY_DIR}
	BUILD_COMMAND make
	BUILD_IN_SOURCE 1
	)
