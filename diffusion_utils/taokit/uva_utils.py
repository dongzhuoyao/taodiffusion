
import socket

def check_host_type():
    """Check the host type."""
    host_name = socket.gethostname()
    if "ivi" in host_name:
        return "ivi"
    elif  host_name in ["node401", "node402", "node403", "node404",
    "node405", "node406", "node407", "node408",
    "node409", "node410", "node411", "node412"]:
        return "das6"
    else:
        return "unknown"