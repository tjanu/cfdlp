#ifndef CXXUTIL_STOPWATCH_HPP
#define CXXUTIL_STOPWATCH_HPP

//#include <cxxutil/definitions.hpp>

#include <string>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <sys/time.h>

namespace cxxutil {

    /*!
     * Stopwatch class
     *
     * This class implements a stopwatch, optionally printing out the elapsed
     * time on destruction, which makes timing of single blocks of execution
     * really easy.
     * \author Thomas Janu
     */
    class stopwatch {
	public:
	    /*!
	     * Constructor
	     * \param print_on_destruction Whether or not to print out the
	     * elapsed type on stopwatch::outstream when destructed
	     */
	    stopwatch(const char * name = NULL, bool print_on_destruction = true);

	    /*!
	     * Destructor
	     */
	    virtual ~stopwatch(void);

	    /*!
	     * Reset the start time of this stopwatch to "now"
	     */
	    void tick(void);

	    /*!
	     * Get the elapsed time since the last call to stopwatch::tick()
	     * \returns a string representing the elapsed time so far
	     */
	    std::string tock(void);

	    /*!
	     * Directly print out the elapsed time to the given stream, or
	     * stopwatch::outstream if called without parameter
	     * \param o the std::ostream to use for output
	     */
	    void tock_out(std::ostream &o = (*outstream));

	    /*!
	     * Get the elapsed time, in seconds, as a float
	     * \returns the elapsed time in seconds
	     */
	    float elapsed_time_sec(void);

	    /*!
	     * Get the elapsed time in milliseconds, as a float
	     * \returns the elapsed time in milliseconds
	     */
	    float elapsed_time_millis(void);

	    /*!
	     * set the default output stream
	     * \param s the output stream to use
	     */
	    static void set_outstream(std::ostream &s);
	    static std::ostream &get_outstream(void);

	protected:
	    /*! The name of this stopwatch (used in output if set) */
	    std::string m_name;
	    /*! The start time */
	    timeval m_start;
	    /*! Whether or not to print out the elapsed time on destruction */
	    bool m_print_on_destruction;
	    /*! The default outstream to use. If not set by the user, std::cerr
	     * is used.
	     */
	    static std::ostream *outstream;
	    
	    /*!
	     * Little helper function to determine the difference between two
	     * time instances, since timersub(3) sadly  requires _BSD_SOURCE -> non-portable
	     * \param end The end time of the interval
	     * \param start the start time of the interval
	     * \param diff the difference between start and end. This is an
	     * output parameter
	     */
	    void get_timeval_diff(timeval &end, timeval &start, timeval &diff);

    }; /* class stopwatch */

    typedef boost::shared_ptr<stopwatch> stopwatch_ptr;

}; /* namespace util */

#endif/*CXXUTIL_STOPWATCH_HPP*/
