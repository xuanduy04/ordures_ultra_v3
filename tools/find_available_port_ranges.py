

import socket


def is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


# Print header
print("Size\tRange")
print("-" * 20)

start: int | None = None
for port in range(1024, 65536):
    if is_free(port):
        if start is None:
            start = port
    else:
        if start is not None:
            if start == port - 1:
                size = 1
                print(f"{size:4d}\t{start}")
            else:
                size = port - start
                print(f"{size:4d}\t{start}-{port - 1}")
            start = None

# If it ends on a free range, print it
if start is not None:
    if start == 65535:
        size = 1
        print(f"{size:4d}\t{start}")
    else:
        size = 65536 - start
        print(f"{size:4d}\t{start}-65535")
