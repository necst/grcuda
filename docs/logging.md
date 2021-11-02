#LOGGING

###Different levels of logging are provided by the Level class, in order to differenciate the importance of occurring errors or warnings. This gives the possibility to decide up to which severity it is convenient to have them printed on the terminal. It is also possible to print them on a file.

##LEVELS

####The logging Level objects are ordered and are specified by ordered integers. Enabling logging at a given level also enables logging at all higher levels.
The levels in descending order are:
- **SEVERE** (highest value)
- **WARNING**
- **INFO**
- **CONFIG**
- **FINE**
- **FINER**
- **FINEST** (lowest value)
In addition there is a level **OFF** that can be used to turn off logging, and a level **ALL** that can be used to enable logging of all messages.
The Level class is already implemented in Java, visit https://docs.oracle.com/javase/7/docs/api/java/util/logging/Level.html for further information.

##LOGGERS IN THE CODE

###AVAILABLE LOGGERS

####GrCUDA is characterized by the presence of several different types of loggers, each one with its own functionality. The GrCUDALogger class is implemented in order to have access to loggers of interest when specific features are needed.
Main examples of loggers in GrCUDALogger follow:
- **MAIN_LOGGER** : all the logging action in grcuda can refer to this principal logger;

```javascript
public static final String MAIN_LOGGER = "com.nvidia.grcuda";
```

- **RUNTIME_LOGGER** : referral for each logging action in runtime project of grcuda;

```javascript
public static final String RUNTIME_LOGGER = "com.nvidia.grcuda.runtime";
```

- **EXECUTIONCONTEXT_LOGGER** : referral for each logging action in exectution context project of runtime;

```javascript
public static final String EXECUTIONCONTEXT_LOGGER = "com.nvidia.grcuda.runtime.executioncontext";
```

####If further loggers are needed to be implemented, it can be easily done by adding them to the GrCUDALogger class, being sure of respecting the name convention, like in the following example.

```javascript
public static final String LOGGER_NAME = "com.nvidia.grcuda.file_name";
```

###USING AVAILABLE LOGGERS

####To use the available loggers in the code, follow the instructions below:
1. create the specific logger in the project's class as TruffleLogger type object:

```javascript
public static final TruffleLogger LOGGER_NAME = GrCUDALogger.getLogger(GrCUDALogger.LOGGER_NAME);
```
2. set logging level to the message with:

```javascript
LOGGER_NAME.logger_level("message");
```

####As alternative of step 2. it is also possible to directly associate logging level to messages by using the following form:

```javascript
GrCUDALogger.getLogger(GrCUDALogger.LOGGER_NAME).logger_level("*message*");
```

##LOGGERS CONFIGURATION

####All loggers are set to level INFO by default.
It is possible to modify the level of all the messages in a file with graal options from the command line.
In particular, it is possible to specify a unique output file for all the logger messages.

```javascript
-- log.file=path_to_file
```

####Is it also possible to specify the logging level for each logger.

```javascript
--log.grcuda.com.nvidia.grcuda.file_name.level=logger_level
```

