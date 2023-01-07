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
        from cpuinfo import get_cpu_info

        cpu_info = get_cpu_info()
    except ImportError:
        cpu_info = None

    try:
        from cupy.cuda.runtime import getDeviceProperties, getDeviceCount

        def strkey(v):
            if type(v) is bytes:
                try:
                    return str(v, "utf-8")
                except UnicodeDecodeError:
                    return None
            else:
                return v

        gpu_info = list(
            {k: strkey(v) for k, v in getDeviceProperties(i).items()}
            for i in range(getDeviceCount())
        )

    except ImportError:
        gpu_info = None

    return dict(
        host=host,
        code=code,
        cpu_info=cpu_info,
        gpu_info=gpu_info,
        datetime=str(datetime.now()),
    )
