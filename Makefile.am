#########################################################################
# Copyright 2010 by Raytheon BBN Technologies Corp.  All Rights Reserved
#########################################################################

include $(top_srcdir)/Makefile-include.am
include $(top_srcdir)/Makefile-tested.am

AM_CFLAGS  +=  -O4 -std=c99 -Wall -I$(COTS_PATH)/build$(RATS_ARCH_SUFFIX)/$(FFTW)/api
AM_LDFLAGS   += -L$(COTS_PATH)/opt/$(FFTW)$(RATS_COMPILER_SUFFIX)$(RATS_ARCH_SUFFIX)/lib/ -l$(lib_FFTW) -lm


bin_PROGRAMS = cfdlp

cfdlp_SOURCES = cfdlp.c util.c icsilog.c adapt_m.c
cfdlp_HEADERS = util.h icsilog.h
cfdlpdir = fdlp


tested-local::
	rm -f example.01.out
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis mel -o example.01.out -print example.01.print -sr 16000) >& log
	$(scriptsdir)/diff_numeric.pl -e 0.0001 $(testdir)/example.01.print.ans example.01.print
	-diff log $(testdir)/log
#	$(CSH) $(srcdir)/test.sh $(srcdir) $(testdir)