CC = gcc
CFLAGS = -fPIC -c
LDFLAGS = -shared
TARGET = my_program.so  # Change .so to .dll
SOURCES = main.c read_file.c pruning.c helpers/cJSON.c path_generation.c
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS) -Wl,--out-implib,libmy_program.a

clean:
	rm -f $(OBJECTS) $(TARGET) libmy_program.a
