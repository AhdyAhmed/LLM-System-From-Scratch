@ECHO OFF
SETLOCAL
IF EXIST "%ProgramData%\chocolatey\lib\maven\apache-maven-3.9.16\bin\mvn.cmd" (
    CALL "%ProgramData%\chocolatey\lib\maven\apache-maven-3.9.16\bin\mvn.cmd" %*
    EXIT /B %ERRORLEVEL%
)
CALL mvn %*
EXIT /B %ERRORLEVEL%
