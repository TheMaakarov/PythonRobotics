from enum import IntEnum

class RouteType(IntEnum):
    Straight = 0
    Serpent = 1
    SerpentRev = 2
    Forward = 3
    SwitchBack = 4

class LogLevel(IntEnum):
    Trace = 0
    Debug = 1
    Info = 2
    Warn = 3
    Error = 4
    Fatal = 5