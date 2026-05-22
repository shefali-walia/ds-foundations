"""
STRFTIME FORMAT CODES — for DateFormatter and any date formatting in Python

%Y   -- 4-digit year        : 2011, 2024
%y   -- 2-digit year        : 11, 24

%m   -- month as number     : 01, 06, 12
%b   -- month abbreviated   : Jan, Jun, Dec
%B   -- month full name     : January, June, December

%d   -- day as number       : 01, 15, 31

%H   -- hour (24hr)         : 00, 13, 23
%M   -- minute              : 00, 30, 59    (capital M = minute, not month)
%S   -- second              : 00, 45, 59

COMBINING THEM:
'%Y-%m'       -- 2011-01
'%b %Y'       -- Jan 2011
'%d/%m/%Y'    -- 01/01/2011
'%B %d, %Y'   -- January 01, 2011
'%d %b'       -- 01 Jan

EASY TRAP: %m (lowercase) = month number
           %M (uppercase) = minute
these will silently give wrong output and confuse you, remember this

"""