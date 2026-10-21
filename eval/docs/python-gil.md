# The Python global interpreter lock

CPython uses a global interpreter lock, usually called the GIL. The GIL allows only one thread to execute Python bytecode at a time. This keeps reference counting simple and makes single threaded code fast.

Because of the GIL, CPU bound work does not get faster with threads. The multiprocessing module avoids the problem by running separate interpreter processes. Threads are still useful for IO bound work, since the lock is released while waiting on the network or disk.

Python 3.13 added an experimental free threaded build that can run without the GIL.
