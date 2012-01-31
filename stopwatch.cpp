#include "stopwatch.hpp"

#include <sstream>

std::ostream *cxxutil::stopwatch::outstream = &std::cerr;

namespace cxxutil {

    stopwatch::stopwatch(const char * name, bool print_on_destruction) : m_print_on_destruction(print_on_destruction) {
	if (name != NULL) {
	    m_name = std::string(name);
	}
	tick();
    }

    stopwatch::~stopwatch() {
	if (m_print_on_destruction) {
	    if (m_name.size() > 0) {
		(*outstream) << "[" << m_name << "] ";
	    }
	    (*outstream) << "Elapsed time: " << tock() << std::endl;
	}
    }

    void stopwatch::tick() {
	gettimeofday(&m_start, NULL);
    }

    std::string stopwatch::tock() {
	timeval end, diffval;
	gettimeofday(&end, NULL);
	get_timeval_diff(end, m_start, diffval);

	long int diff = diffval.tv_sec * 1000 + (diffval.tv_usec / 1000);
	float fsec = (float)diff / 1e3;

	std::stringstream sstr;

	if (diff < 1000) {
	    sstr << diff << " ms";
	} else {
	    sstr.precision(3);
	    if (diff < 60000) {
		sstr << fsec << " s";
	    } else {
		if (diff < 3600000) {
		    sstr << (int)(diff / 60000) << " m "
			<< (int)((diff % 60000) / 1000) << " s "
			<< "(or " << fsec << " s)";
		} else {
		    sstr << (int)(diff / 3600000) << " hr "
			<< (int)((diff % 3600000) / 60000) << " m "
			<< (int)(((diff % 3600000) % 60000) / 1000) << " s "
			<< "(or " << fsec << " s)";
		}
	    }
	}
	return sstr.str();
    }

    void stopwatch::tock_out(std::ostream &o) {
	if (m_name.size() > 0) {
	    o << "[" << m_name << "] ";
	}
	o << "Elapsed time: " << tock() << std::endl;
    }

    float stopwatch::elapsed_time_sec() {
	timeval end, diff;
	gettimeofday(&end, NULL);
	get_timeval_diff(end, m_start, diff);
	return ((float)diff.tv_sec) + ((float)diff.tv_usec / 1e6);
    }

    float stopwatch::elapsed_time_millis() {
	timeval end, diff;
	gettimeofday(&end, NULL);
	get_timeval_diff(end, m_start, diff);
	return ((float)diff.tv_sec * 1e3) + ((float)diff.tv_usec / 1e3);
    }

    void stopwatch::get_timeval_diff(timeval &end, timeval &start, timeval &diff) {
	diff.tv_sec = end.tv_sec - start.tv_sec - (start.tv_usec > end.tv_usec ? 1 : 0);
	diff.tv_usec = (start.tv_usec < end.tv_usec ? end.tv_usec - start.tv_usec : 1000000 - (start.tv_usec - end.tv_usec));
    }

    void stopwatch::set_outstream(std::ostream &s) {
	outstream = &s;
    }

    std::ostream &stopwatch::get_outstream() {
	return (*outstream);
    }
}; /* namespace cxxutil */
