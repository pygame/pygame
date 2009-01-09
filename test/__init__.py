def run():
    if __name__ == 'pygame.tests':
        from pygame.tests.test_utils.run_tests import run
    else:
        from test.test_utils.run_tests import run
    run()
