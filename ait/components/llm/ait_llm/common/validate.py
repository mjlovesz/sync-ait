import functools


def validate_parameters_by_func(parameter_constraints, in_class=False):
    if not isinstance(parameter_constraints, dict):
        raise TypeError(f"Parameter constraints expects dict, but got {type(parameter_constraints).__name__} instead.")

    if not parameter_constraints:
        raise ValueError(f"Parameter constraints should not be empty.")

    if not all(isinstance(key, str) for key in parameter_constraints.keys()):
        raise ValueError(f"Key of the parameter constraints only supports string.")

    if not all(isinstance(val, (tuple, list)) for val in parameter_constraints.values()):
        raise ValueError(f"Value of the parameter constraints only supports tuple or list.")

    def decorator(func):

        def _check_constraint(arg, constraint_name, constraint):
            if not constraint:
                return

            for check_item in constraint:

                if callable(check_item):
                    check_name = check_item.__name__
                    
                    test_result = None
                    try:
                        test_result = check_item(arg)
                    except Exception as e:
                        raise RuntimeError(
                            f"In the running function `{func.__name__}`, the argument `{constraint_name}`, whose value is `{arg}`, " 
                            f"is invalid as it has not been passed through the designated constraints.") from e

                    try:
                        test_result = True if test_result is None else bool(test_result)
                    except Exception as e:
                        raise RuntimeError(
                            f"The result from the designated constraints `{check_name}` can not be interpreted as bool.") from e

                    if not test_result:
                        raise RuntimeError(
                            f"In the running function `{func.__name__}`, the argument `{constraint_name}`, whose value is `{arg}`, " 
                            f"is invalid as it has not been passed through the designated constraints.")

                else:
                    raise TypeError(
                        f"Provided `{check_item}` that associated with key `{constraint_name}` is invalid: Not callable.")
            
        if in_class:
            @functools.wraps(func)
            def wrapper(self_or_cls, *args, **kwargs):
                constraints_iterator = iter(parameter_constraints.items())

                if args:
                    for arg, (constraint_name, constraint) in zip(args, constraints_iterator):
                        _check_constraint(arg, constraint_name, constraint)

                if kwargs:
                    for (_, arg_val), (constraint_name, constraint) in zip(kwargs.items(), constraints_iterator):
                        _check_constraint(arg_val, constraint_name, constraint)

                return func(self_or_cls, *args, **kwargs)

            return wrapper
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            constraints_iterator = iter(parameter_constraints.items())

            if args:
                for arg, (constraint_name, constraint) in zip(args, constraints_iterator):
                    _check_constraint(arg, constraint_name, constraint)

            if kwargs:
                for (_, arg_val), (constraint_name, constraint) in zip(kwargs.items(), constraints_iterator):
                    _check_constraint(arg_val, constraint_name, constraint)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# if no type matches, the function should also check type again, which is not desired
def validate_parameters_by_type(parameter_constraints, in_class=False):
    """Easy type checking

    Automatically check the arguments of a function based on given constraints `parameter_constraints` in the runtime.
    `parameter_constraints` is a user-defined dictionary and users should be responsible for the match of key and the
    real argument name.

    Self-defined functions should only take one argument. By default, if the function is not passed, it will raise an
    exception. If the test is complicated, or related to other variable, or user wants to other exception messages, 
    user should take care it on their own.

    in_class: take care of self and cls, static method should not use this option.

    Users also should take care of `*arg` and `**kwargs` on their own. This is only designed for explicit argument
    cases.
    """

    if not isinstance(parameter_constraints, dict):
        raise TypeError(f"Parameter constraints expects dict, but got {type(parameter_constraints).__name__} instead.")

    if not parameter_constraints:
        raise ValueError(f"Parameter constraints should not be empty.")

    if not all(isinstance(key, str) for key in parameter_constraints.keys()):
        raise ValueError(f"Key of the parameter constraints only supports string.")

    if not all(isinstance(val, (tuple, list)) for val in parameter_constraints.values()):
        raise ValueError(f"Value of the parameter constraints only supports tuple or list.")

    def decorator(func):

        def _check_constraint(arg, constraint_name, constraint):
            if not constraint:
                return

            type_list = []

            for check_item in constraint:
                if isinstance(check_item, type):
                    check_name = check_item.__name__
                    type_list.append(check_name)

                    # if that is the one
                    if isinstance(arg, check_item):
                        return

                elif isinstance(check_item, type(None)):
                    type_list.append(None)

                    if arg is None:
                        return
                        
                else:
                    raise TypeError(
                        f"Provided `{check_item}` that associated with key `{constraint_name}` is invalid. Only types "
                        f"are allowed.")

            raise TypeError(
                        f"In the running function `{func.__name__}`, the argument `{constraint_name}`, whose value is `{arg}`, "
                        f"is invalid, where `{type_list}` is expected, but got `{type(arg).__name__}` instead.")

        if in_class:
            @functools.wraps(func)
            def wrapper(self_or_cls, *args, **kwargs):
                constraints_iterator = iter(parameter_constraints.items())

                if args:
                    for arg, (constraint_name, constraint) in zip(args, constraints_iterator):
                        _check_constraint(arg, constraint_name, constraint)

                if kwargs:
                    for (_, arg_val), (constraint_name, constraint) in zip(kwargs.items(), constraints_iterator):
                        _check_constraint(arg_val, constraint_name, constraint)

                return func(self_or_cls, *args, **kwargs)

            return wrapper
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            constraints_iterator = iter(parameter_constraints.items())

            if args:
                for arg, (constraint_name, constraint) in zip(args, constraints_iterator):
                    _check_constraint(arg, constraint_name, constraint)

            if kwargs:
                for (_, arg_val), (constraint_name, constraint) in zip(kwargs.items(), constraints_iterator):
                    _check_constraint(arg_val, constraint_name, constraint)

            return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == '__main__':
    def test(a):
        if a == 0:
            raise ZeroDivisionError("abc")
        
        if a > -1:
            raise ValueError("bcd")
        
        if a < -1:
            raise RuntimeError("qwe")


    @validate_parameters_by_type(
        {
            "a": (float, str, int, ),
            "b": (int, float, str),
            "c": (str, int, float),
        }
    )
    @validate_parameters_by_func(
        {
            "a": (len, sum),
            "b": (),
            "c": [sum],
        }
    )
    def foo(a, b, c):
        print(a, b, c)

    foo([1, 2, 3,], [], [])
