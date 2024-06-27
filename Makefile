# Define the compiler
CXX = g++

# Define compiler flags for release and debug builds
CXXFLAGS_RELEASE = -Wall -Werror -Wshadow -std=c++17 -O2
CXXFLAGS_DEBUG = -Wall -Werror -Wshadow -std=c++17 -O2 -g -fsanitize=address

# Define the output executable name
TARGET = stab

# Define the source directory
SRC_DIR = src

# Automatically find all .cpp and .h/.hpp files in the source directory
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
HEADERS = $(wildcard $(SRC_DIR)/*.h $(SRC_DIR)/*.hpp)

# Define object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: release

# Release build target
release: CXXFLAGS = $(CXXFLAGS_RELEASE)
release: $(TARGET)

# Debug build target
debug: CXXFLAGS = $(CXXFLAGS_DEBUG)
debug: clean $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files into object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean the build files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all release debug clean
