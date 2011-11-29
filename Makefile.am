#########################################################################
# Copyright 2010 by Raytheon BBN Technologies Corp.  All Rights Reserved
#########################################################################

include $(top_srcdir)/Makefile-include.am
include $(top_srcdir)/Makefile-tested.am

AM_CFLAGS  +=  -O4 -std=c99 -Wall -lpthread -I$(COTS_PATH)/build$(RATS_ARCH_SUFFIX)/$(FFTW)/api
AM_LDFLAGS   += -L$(COTS_PATH)/opt/$(FFTW)$(RATS_COMPILER_SUFFIX)$(RATS_ARCH_SUFFIX)/lib/ -l$(lib_FFTW) -lm


bin_PROGRAMS = cfdlp

cfdlp_SOURCES = cfdlp.c util.c icsilog.c adapt_m.c
cfdlp_HEADERS = util.h icsilog.h
cfdlpdir = fdlp


tested-local::
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis mel -print example.01.print -sr 16000) >& log
	$(scriptsdir)/diff_numeric.pl -e 0.0001 $(testdir)/example.01.print.ans example.01.print
	-diff log $(testdir)/log
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis linear-mel -print example.01.linmel.print -sr 16000) >& log.linmel
	$(scriptsdir)/diff_numeric.pl -e 0.0001 $(testdir)/example.01.linmel.print.ans example.01.linmel.print
	-diff log.linmel $(testdir)/log.linmel
	(./cfdlp -i $(testdir)/example.02.in -gn 1 -spec 1 -axis mel -apply-wiener 1 -print example.02.wiener-mel.print -sr 8000) >& log.wiener-mel
	$(scriptsdir)/diff_numeric.pl -e 0.0001 $(testdir)/example.02.wiener-mel.print.ans example.02.wiener-mel.print
	-diff log.wiener-mel $(testdir)/log.wiener-mel
	(./cfdlp -i $(testdir)/example.02.in -gn 1 -spec 1 -axis linear-mel -apply-wiener 1 -print example.02.wiener-linmel.print -sr 8000) >& log.wiener-linmel
	$(scriptsdir)/diff_numeric.pl -e 0.0001 $(testdir)/example.02.wiener-linmel.print.ans example.02.wiener-linmel.print
	-diff log.wiener-linmel $(testdir)/log.wiener-linmel
#	$(CSH) $(srcdir)/test.sh $(srcdir) $(testdir)
