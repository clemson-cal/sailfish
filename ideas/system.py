__version__ = "0.6.0"


def system_info():
    from platform import node, machine, processor, platform, system, release, version
    from os import cpu_count
    from subprocess import check_output
    from datetime import datetime

    host = dict()
    host["node"] = node()
    host["machine"] = machine()
    host["processor"] = processor()
    host["cpu_count"] = cpu_count()
    host["platform"] = platform()
    host["system"] = system()
    host["release"] = release()
    host["version"] = version()

    code = dict()
    code["version"] = __version__
    code["commit"] = str(check_output(["git", "rev-parse", "HEAD"]), "utf-8").strip()

    try:
        from cupy.cuda.device import getDeviceProperties, getDeviceCount

        gpu_info = list(getDeviceProperties(i) for i in range(getDeviceCount()))

    except ImportError:
        gpu_info = None

    return dict(host=host, code=code, gpu_info=gpu_info, datetime=str(datetime.now()))
