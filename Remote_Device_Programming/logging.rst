.. _logging:

==========
Logging
==========

UCS has logging infrastructure. logging is controlled by a single level:

 * fatal - stops the program
 * error - an error which does not stop the program and can be reported back to user.
 * warn - a warning which does not return error to the user.
 
info
 * debug - debugging messages, low volume, about initialization/cleanup.
 * trace - debugging messages, high volume, during runtime, for “special” events.
 * req - details of every send/receive request and tag matching.
 * data - headers of every packet being sent/received.
 * async - async notifications and progress thread.
 * func - function calls and arguments.
 * poll - same as func, but also for functions which do busy-polling.

Rules:

 * UCS_LOG_LEVEL controls the logging level.
 * When log level is selected, it enables all messages with this level or higher.
 * Logging messages can be sent to stdout (default) or to a file by setting UCS_LOG_FILE

Features:

 * It's possible to define maximal log level during compile time, to avoid checks in release mode.
 * It's possible to install custom log message handler.

