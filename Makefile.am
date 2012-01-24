#########################################################################
# Copyright 2010 by Raytheon BBN Technologies Corp.  All Rights Reserved
#########################################################################

include $(top_srcdir)/Makefile-include.am
include $(top_srcdir)/Makefile-tested.am

AM_CFLAGS  += -O4 -std=c99 -Wall -I$(COTS_PATH)/build$(RATS_ARCH_SUFFIX)/$(FFTW)/api ${CFDLP_EXTRA_CFLAGS}
AM_LDFLAGS   += -L$(COTS_PATH)/opt/$(FFTW)$(RATS_COMPILER_SUFFIX)$(RATS_ARCH_SUFFIX)/lib/ -l$(lib_FFTW)


bin_PROGRAMS = cfdlp

cfdlp_SOURCES = cfdlp.c util.c icsilog.c adapt_m.c
cfdlp_HEADERS = util.h icsilog.h
cfdlpdir = fdlp


tested-local::
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis mel -print example.01.print -sr 8000) >& log
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.print.ans example.01.print
	-diff log $(testdir)/log
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis bark -print example.01.bark.print -sr 8000) >& log.bark
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.bark.print.ans example.01.bark.print
	-diff log.bark $(testdir)/log.bark
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis linear-mel -print example.01.linmel.print -sr 8000) >& log.linmel
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.linmel.print.ans example.01.linmel.print
	-diff log.linmel $(testdir)/log.linmel
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis linear-bark -print example.01.linbark.print -sr 8000) >& log.linbark
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.linbark.print.ans example.01.linbark.print
	-diff log.linbark $(testdir)/log.linbark
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis mel -limit-range 1 -print example.01.rangelimit.print -sr 8000) >& log.rangelimit
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.rangelimit.print.ans example.01.rangelimit.print
	-diff log.rangelimit $(testdir)/log.rangelimit
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis mel -fdplpwin 8 -print example.01.fdplpwin.print -sr 8000) >& log.fdplpwin
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.fdplpwin.print.ans example.01.fdplpwin.print
	-diff log.fdplpwin $(testdir)/log.fdplpwin
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis mel -fdplpwin 8 -truncate-last 1 -print example.01.truncate.print -sr 8000) >& log.truncate
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.truncate.print.ans example.01.truncate.print
	-diff log.truncate $(testdir)/log.truncate
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -spec 1 -axis mel -skip-bands 2 -print example.01.skipbands.print -sr 8000) >& log.skipbands
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.skipbands.print.ans example.01.skipbands.print
	-diff log.skipbands $(testdir)/log.skipbands
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -axis mel -print example.01.modfeats.print -sr 8000) >& log.modfeats
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.modfeats.print.ans example.01.modfeats.print
	-diff log.modfeats $(testdir)/log.modfeats
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -axis bark -print example.01.modfeatsbark.print -sr 8000) >& log.modfeatsbark
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.modfeatsbark.print.ans example.01.modfeatsbark.print
	-diff log.modfeatsbark $(testdir)/log.modfeatsbark
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -feat 1 -axis bark -print example.01.nceps.print -sr 8000 -nceps 10) >& log.nceps
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.nceps.print.ans example.01.nceps.print
	-diff log.nceps $(testdir)/log.nceps
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -feat 1 -axis bark -print example.01.shortterm0.print -sr 8000 -shortterm-mode 0) >& log.shortterm0
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.shortterm0.print.ans example.01.shortterm0.print
	-diff log.shortterm0 $(testdir)/log.shortterm0
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -feat 1 -axis bark -print example.01.shortterm1.print -sr 8000 -shortterm-mode 1) >& log.shortterm1
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.shortterm1.print.ans example.01.shortterm1.print
	-diff log.shortterm1 $(testdir)/log.shortterm1
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -feat 1 -axis bark -print example.01.shortterm2.print -sr 8000 -shortterm-mode 2) >& log.shortterm2
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.shortterm2.print.ans example.01.shortterm2.print
	-diff log.shortterm2 $(testdir)/log.shortterm2
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -feat 1 -axis bark -print example.01.shortterm3.print -sr 8000 -shortterm-mode 3) >& log.shortterm3
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.shortterm3.print.ans example.01.shortterm3.print
	-diff log.shortterm3 $(testdir)/log.shortterm3
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -axis bark -print example.01.modulationmode0.print -sr 8000 -modulation-mode 0) >& log.modulationmode0
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.modulationmode0.print.ans example.01.modulationmode0.print
	-diff log.modulationmode0 $(testdir)/log.modulationmode0
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -axis bark -print example.01.modulationmode1.print -sr 8000 -modulation-mode 1) >& log.modulationmode1
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.modulationmode1.print.ans example.01.modulationmode1.print
	-diff log.modulationmode1 $(testdir)/log.modulationmode1
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -axis bark -print example.01.modulationmode2.print -sr 8000 -modulation-mode 2) >& log.modulationmode2
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.modulationmode2.print.ans example.01.modulationmode2.print
	-diff log.modulationmode2 $(testdir)/log.modulationmode2
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -axis bark -print example.01.dsfactor.print -sr 8000 -downsampling-factor 20) >& log.dsfactor
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.dsfactor.print.ans example.01.dsfactor.print
	-diff log.dsfactor $(testdir)/log.dsfactor
	(./cfdlp -i $(testdir)/example.01.in -gn 1 -axis bark -print example.01.dssr.print -sr 8000 -downsampling-sr 400) >& log.dssr
	$(scriptsdir)/diff_numeric_using_max_base.pl -e 0.01 $(testdir)/example.01.dssr.print.ans example.01.dssr.print
	-diff log.dssr $(testdir)/log.dssr
#	$(CSH) $(srcdir)/test.sh $(srcdir) $(testdir)
