CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -pthread
LDFLAGS ?=
LDLIBS  ?= -pthread

all: rowserverd

rowserverd: rowserverd.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

# Static build for the container image: the daemon resolves no names and loads no
# plugins, so it needs nothing from the runtime filesystem and the image can be
# FROM scratch.
static:
	$(CC) $(CFLAGS) -static $(LDFLAGS) -o rowserverd rowserverd.c $(LDLIBS)

clean:
	rm -f rowserverd

.PHONY: all static clean
