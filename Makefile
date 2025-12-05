CC      = gcc
TARGET  = sorter
C_FILES = parallel_merge.c merge.c
OBJS    = parallel_merge.o merge.o
CFLAGS  = -g -Wall -Werror -pedantic-errors -O3
LDFLAGS = -pthread

# Detect OS and set cleanup command accordingly
ifdef OS
   # Windows
   RM = del /Q /F
   EXT = .exe
else
   # Linux/Unix
   RM = rm -f
   EXT =
endif

.PHONY: all clean
all: $(TARGET)
$(TARGET): $(OBJS) $(TARGET).c
	$(CC) $(OBJS) $(TARGET).c -o $(TARGET) $(LDFLAGS)
%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<
clean:
	$(RM) $(OBJS) $(TARGET)$(EXT)

