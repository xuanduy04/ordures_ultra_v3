

import os
import subprocess
import signal

def run_test_script(folder, test_filename):
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    test_file_path = os.path.join(dir_path, 'functional_tests', folder, test_filename)
    p = subprocess.Popen(
        ["bash", test_file_path],
        preexec_fn=os.setsid          # On Unix: puts it in a new session/process group
    )

    try:
        assert p.wait() == 0
    finally:
        # Kill the entire process group, not just p
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
