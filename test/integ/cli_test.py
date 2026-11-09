##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of H5Serv (HDF5 REST Server) Service, Libraries and      #
# Utilities.  The full HDF5 REST Server copyright notice, including          #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

import h5py

top_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
src_dir = os.path.join(top_dir, "src")
tall_h5 = os.path.join(top_dir, "data", "hdf5", "tall.h5")
tall_json = os.path.join(top_dir, "data", "json", "tall.json")


def run_app(app, args, cwd):
    """Run an h5json CLI module from cwd and return the CompletedProcess."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_dir, env.get("PYTHONPATH")) if p)
    cmd = [sys.executable, "-m", f"h5json.apps.{app}"] + list(args)
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


class CliTest(unittest.TestCase):

    def make_tempdir(self):
        """Return a temp dir that is removed when the test ends."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name

    # --- logging ---

    def testNoLogFileByDefault(self):
        tmp = self.make_tempdir()
        proc = run_app("h5tojson", [tall_h5], tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("root", json.loads(proc.stdout))
        self.assertNotIn("DEBUG ", proc.stderr)

        out_h5 = os.path.join(tmp, "out.h5")
        proc = run_app("jsontoh5", [tall_json, out_h5], tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with h5py.File(out_h5, "r") as f:
            self.assertIn("g1", f)
        self.assertEqual(os.listdir(tmp), ["out.h5"])

    def testLogOption(self):
        tmp = self.make_tempdir()
        log_path = os.path.join(tmp, "h5tojson-debug.log")
        proc = run_app("h5tojson", ["--log", log_path, tall_h5], tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("root", json.loads(proc.stdout))
        with open(log_path) as f:
            log_text = f.read()
        self.assertIn("INFO ", log_text)
        self.assertIn(f"h5tojson {tall_h5}", log_text)
        self.assertIn("DEBUG ", log_text)

        log_path = os.path.join(tmp, "jsontoh5-debug.log")
        out_h5 = os.path.join(tmp, "out.h5")
        proc = run_app("jsontoh5", [tall_json, out_h5, "--log", log_path], tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(log_path) as f:
            log_text = f.read()
        self.assertIn(f"jsontoh5 {tall_json} to {out_h5}", log_text)
        self.assertIn("DEBUG ", log_text)

    @unittest.skipIf(os.name == "nt" or os.geteuid() == 0, "needs POSIX permissions and a non-root user")
    def testReadOnlyCwd(self):
        tmp = self.make_tempdir()
        ro_dir = os.path.join(tmp, "ro")
        os.mkdir(ro_dir)
        os.chmod(ro_dir, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        self.addCleanup(os.chmod, ro_dir, stat.S_IRWXU)

        proc = run_app("h5tojson", [tall_h5], ro_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("root", json.loads(proc.stdout))

        out_h5 = os.path.join(tmp, "tall.h5")
        proc = run_app("jsontoh5", [tall_json, out_h5], ro_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with h5py.File(out_h5, "r") as f:
            self.assertIn("g1", f)

        # a --log file that can't be created falls back to stderr
        proc = run_app("h5tojson", ["--log", "h5tojson.log", tall_h5], ro_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("root", json.loads(proc.stdout))
        self.assertIn("not logging to 'h5tojson.log'", proc.stderr)
        self.assertEqual(os.listdir(ro_dir), [])

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def testLogSymlinkNotFollowed(self):
        tmp = self.make_tempdir()
        canary = os.path.join(tmp, "canary.txt")
        with open(canary, "w") as f:
            f.write("canary\n")

        link = os.path.join(tmp, "link.log")
        os.symlink(canary, link)
        proc = run_app("h5tojson", ["--log", link, tall_h5], tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("root", json.loads(proc.stdout))
        self.assertIn(f"not logging to '{link}'", proc.stderr)

        out_h5 = os.path.join(tmp, "out.h5")
        proc = run_app("jsontoh5", ["--log", link, tall_json, out_h5], tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(f"not logging to '{link}'", proc.stderr)
        with h5py.File(out_h5, "r") as f:
            self.assertIn("g1", f)

        with open(canary) as f:
            self.assertEqual(f.read(), "canary\n")

    def testLogEscapesControlCharacters(self):
        tmp = self.make_tempdir()
        h5_path = os.path.join(tmp, "newline.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_group("g\nINJECTED-LINE\r#")
        log_path = os.path.join(tmp, "h5tojson.log")
        proc = run_app("h5tojson", ["--log", log_path, h5_path], tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("root", json.loads(proc.stdout))
        with open(log_path) as f:
            log_text = f.read()
        self.assertIn("g\\nINJECTED-LINE\\r#", log_text)
        for line in log_text.splitlines():
            self.assertFalse(line.startswith("INJECTED-LINE"), line)

    # --- argument handling ---

    def testDataLimitAfterFilename(self):
        tmp = self.make_tempdir()
        with h5py.File(os.path.join(tmp, "f.h5"), "w") as f:
            f.attrs["marker"] = "real-input"
        with h5py.File(os.path.join(tmp, "5"), "w") as f:
            f.attrs["marker"] = "decoy-input"
        proc = run_app("h5tojson", ["f.h5", "--data-limit", "5"], tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("real-input", proc.stdout)
        self.assertNotIn("decoy-input", proc.stdout)

    def testBadDataLimit(self):
        tmp = self.make_tempdir()
        for args in (["--data-limit", "-1", tall_h5], [tall_h5, "--data-limit", "x"], [tall_h5, "--data-limit"]):
            proc = run_app("h5tojson", args, tmp)
            self.assertEqual(proc.returncode, 2, args)
            self.assertEqual(proc.stdout, "")

    def testH5tojsonMissingInput(self):
        tmp = self.make_tempdir()
        proc = run_app("h5tojson", ["--nodata"], tmp)
        self.assertEqual(proc.returncode, 2)

    def testJsontoh5ExtraPositional(self):
        tmp = self.make_tempdir()
        proc = run_app("jsontoh5", [tall_json, "b.h5", "c.h5"], tmp)
        self.assertEqual(proc.returncode, 2)
        self.assertFalse(os.path.exists(os.path.join(tmp, "b.h5")))
        self.assertFalse(os.path.exists(os.path.join(tmp, "c.h5")))

    def testJsontoh5MissingOutput(self):
        tmp = self.make_tempdir()
        proc = run_app("jsontoh5", ["--nodata", tall_json], tmp)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(os.listdir(tmp), [])

    def testJsontoh5SameInputAndOutput(self):
        tmp = self.make_tempdir()
        src = os.path.join(tmp, "in.json")
        with open(tall_json) as f_in, open(src, "w") as f_out:
            f_out.write(f_in.read())
        with open(src) as f:
            before = f.read()
        link = os.path.join(tmp, "link.h5")
        os.link(src, link)
        for out in (src, link):
            proc = run_app("jsontoh5", [src, out], tmp)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("same file", proc.stderr)
            with open(src) as f:
                self.assertEqual(f.read(), before)

    def testValidatorContinuesPastMalformedFile(self):
        tmp = self.make_tempdir()
        bad = os.path.join(tmp, "bad.json")
        with open(bad, "w") as f:
            f.write("{not json")
        deep = os.path.join(tmp, "deep.json")
        with open(deep, "w") as f:
            f.write("[" * 100000 + "]" * 100000)
        proc = run_app("validator", [bad, deep, tall_json], tmp)
        self.assertEqual(proc.returncode, 1)
        self.assertIn(f"Validating {bad} ... FAIL", proc.stdout)
        self.assertIn(f"Validating {deep} ... FAIL", proc.stdout)
        self.assertIn(f"Validating {tall_json} ... pass", proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main()
