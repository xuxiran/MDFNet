"""Regression checks for repository-root CLI imports."""
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliImportTests(unittest.TestCase):
    def test_help_commands_run_from_repository_root(self):
        commands = (
            ("main.py", "--help"),
            ("run_loso.py", "--help"),
            ("analysis/run_fig2_experiment.py", "--help"),
        )
        for command in commands:
            with self.subTest(command=command[0]):
                result = subprocess.run(
                    [sys.executable, *command],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"{command[0]} --help failed:\n{result.stdout}\n{result.stderr}",
                )
                self.assertIn("usage:", result.stdout.lower())
                self.assertNotIn("BASEmodels.architectures", result.stderr)


if __name__ == "__main__":
    unittest.main()
