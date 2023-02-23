from inspect import getfullargspec


PRESET_FUNCTIONS = dict()


def preset(func):
    """
    Decorator function to register a preset function

    A preset is a dictionary of config items, at least specifying an
    initial_data field but probably also some physics, a domain, a boundary
    condition, and a driver.tfinal.

    A preset function returns a preset. Note that preset functions cannot take
    any arguments; if the preset function is defined as part of a class, it
    cannot take a self parameter.

    NOTE: in sailfish versions before 0.6, a preset was also called a "setup."
    This terminology should be avoided, since files and functions with names
    like setup could be confused with installation or package distribution
    scripts.
    """
    if getfullargspec(func).args:
        raise ValueError("preset function cannot take any arguments")

    func.__preset_function__ = True
    PRESET_FUNCTIONS[func.__name__.replace("_", "-")] = func
    return func


def is_preset_function(func):
    """
    Return true if the given function is a preset function
    """
    return getattr(func, "__preset_function__", False)


def get_preset_functions():
    """
    Return a copy of registered preset functions
    """
    return dict(**PRESET_FUNCTIONS)
