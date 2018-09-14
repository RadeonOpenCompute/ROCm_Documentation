.. _Async:

Infrastructure for asynchronous event delivery
************************************************
This infrastructure is for handling external events. There are 2 types of events:

 * File-descriptor events
 * Timer
Also, there are 2 methods to deliver the events:

 * Signal
 * Progress thread
A user may register for file/timer events and select which method to use. User can also have an optional "async context" which guarantees safe (synchronized) event delivery. The user can temporarily "block" events for an async context, in which case they will be put in a queue, and check for them later. If the events are not block, they will be dispatched in an asynchronous manner (either on progress thread or from signal handler).

Additional notes:

 * There can be only one handler for every file descriptor.
 * There is (practically) no limit on the amount of timers.
 * There is configurable limit on amount of events/timers per context, due to limited size of missed events queue. (missed = not     	handled because async context was blocked by the user)
 * If there are no timers/event handlers, there will not be a progress thread or signal handler installed.
 * The progress thread/signal handler will wake up in the interval of the minimal timer in the system, if such exists.
 * A timer with interval of less than one millisecond cannot be used with progress thread, because it uses epoll_wait.
 * Signal context can be used only from one thread. If there is no context, it can be used only from main thread.
 * If the system does not support F_SETOWN_EX (redirecting fd signals to particular thread), signals can be used only for main thread.

Further improvements (TODO):

 * Use timerfd in progress thread to handle smaller timer resolution (if kernel supports).
 * Use eventfd instead of pipe() for notifications (if kernel supports). Update: Cannot do it because eventfd does not support SIGIO.

