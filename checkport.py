import socket


def check_port(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        print(f"Connection successful to {host}:{port}!")
    except Exception as e:
        print(f"Connection failed to {host}:{port}: {e}")
    finally:
        s.close()


if __name__ == "__main__":
    host = "127.0.0.1"  # or "localhost"
    port = 51350
    check_port(host, port)
