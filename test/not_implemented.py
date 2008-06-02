fail_incomplete_tests = 0

def not_completed():
    if fail_incomplete_tests:
        return False
    return True

def not_completed_raises():
    raise NotImplementedError, 'Test need to be written!'
