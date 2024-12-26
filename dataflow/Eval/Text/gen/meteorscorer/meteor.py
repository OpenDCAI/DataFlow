import subprocess
import threading
import os
import psutil
import logging
import atexit
import re

# Assumes meteor-1.5.jar is in the same directory as meteor.py. Change the path if needed.
METEOR_JAR = 'meteor-1.5.jar'


import logging

class Meteor:
    def __init__(self, language="en", mem_limit="2G"):
        """Initialize the Meteor object, setup the subprocess to run METEOR"""
        self.lock = threading.Lock()
        self.language = language
        self.mem_limit = mem_limit

        # Check available memory and adjust accordingly
        mem_available_G = psutil.virtual_memory().available / 1E9
        if mem_available_G < 2:
            logging.warning("There is less than 2GB of available memory. Using 1GB instead.")
            self.mem_limit = "1G"

        meteor_cmd = [
            'java', '-jar', '-Xmx{}'.format(self.mem_limit), METEOR_JAR, '-', '-', '-stdio', '-l', self.language, '-norm'
        ]
        env = os.environ.copy()
        env['LC_ALL'] = "C"

        # Start the Meteor subprocess
        self.meteor_p = subprocess.Popen(meteor_cmd,
                                          cwd=os.path.dirname(os.path.abspath(__file__)),
                                          env=env,
                                          stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)

        atexit.register(self.close)

    def close(self):
        """Close the Meteor process properly"""
        with self.lock:
            if hasattr(self, 'meteor_p') and self.meteor_p:
                try:
                    self.meteor_p.kill()
                    self.meteor_p.wait()
                except Exception as e:
                    logging.error(f"Error during Meteor process termination: {str(e)}")
                self.meteor_p = None

    def compute_score(self, hypothesis, references):
        """
        Compute METEOR score for the given hypothesis and references.
        :param hypothesis: The generated hypothesis string.
        :param references: A list of reference strings.
        :return: METEOR score (float).
        """
        assert isinstance(references, list)
        assert len(references) > 0, "References must be a non-empty list"

        eval_line = 'EVAL'
        with self.lock:
            if self.meteor_p.poll() is not None:  # Check if the process is alive
                logging.error("Meteor subprocess has terminated unexpectedly.")
                return None

            # Format the evaluation line for Meteor
            for ref in references:
                stat = self._stat(hypothesis, ref)
                eval_line += ' ||| {}'.format(stat)

            try:
                self.meteor_p.stdin.write(self._enc('{}\n'.format(eval_line)))
                self.meteor_p.stdin.flush()
                score = float(self._dec(self.meteor_p.stdout.readline()).strip())
            except BrokenPipeError as e:
                logging.error(f"Broken pipe error: {str(e)}")
                self.close()  # Close and reopen the subprocess
                return None
            except Exception as e:
                logging.error(f"Error during score calculation: {str(e)}")
                return None

        return score

    def _stat(self, hypothesis_str, reference_str):
        """
        Sends a line to the Meteor process and returns the result.
        :param hypothesis_str: Hypothesis string.
        :param reference_str: Reference string.
        :return: METEOR statistics.
        """
        # Clean hypothesis and reference strings
        hypothesis_str = hypothesis_str.replace('|||', '')
        score_line = ' ||| '.join(('SCORE', reference_str, hypothesis_str))
        score_line = re.sub(r'\s+', ' ', score_line)

        # Send the formatted line to the Meteor subprocess
        self.meteor_p.stdin.write(self._enc(score_line))
        self.meteor_p.stdin.write(self._enc('\n'))
        self.meteor_p.stdin.flush()

        # Read and return the score from Meteor
        return self._dec(self.meteor_p.stdout.readline()).strip()

    def _enc(self, s):
        """Encodes string into UTF-8 bytes."""
        return s.encode('utf-8')

    def _dec(self, s):
        """Decodes bytes into string."""
        return s.decode('utf-8')

    def __del__(self):
        """Ensure proper cleanup when the object is deleted."""
        self.close()
